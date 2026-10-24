"""Compare native Fibo transformer against independent PyTorch equations.
Uses local checkpoint files only; requires torch and safetensors.
"""
import argparse
from pathlib import Path
import subprocess
import torch
from torch.nn import functional as F
from safetensors.torch import save_file, load_file
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from moss_reference import Weights

class FiboWeights(Weights):
    def norm(self,x,name):
        return (x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-5)).to(x.dtype)*self.get(name+'.weight')
    def release(self,prefix):
        self.cache={k:v for k,v in self.cache.items() if not k.startswith(prefix)}

def attention(q,k,v):
    heads=28
    q,k,v=[x.reshape(1,-1,heads,128).transpose(1,2) for x in (q,k,v)]
    return F.scaled_dot_product_attention(q,k,v).transpose(1,2).reshape(-1,3584)

def cross(w,x,context,p):
    q=w.norm(w.linear(x,p+'.to_q').reshape(-1,28,128),p+'.norm_q').reshape(-1,3584)
    k,v=w.linear(context,p+'.to_kv').chunk(2,-1)
    k=w.norm(k.reshape(-1,28,128),p+'.norm_k').reshape(-1,3584)
    return w.linear(attention(q,k,v),p+'.to_out')

def ff(w,x,p):
    a,b=w.linear(x,p+'.w13').chunk(2,-1)
    return w.linear(F.silu(a)*b,p+'.w2')

def perceiver(w,context):
    x=w.get('perceiver.latent_queries')[:256].clone()
    for i in range(2):
        p=f'perceiver.layers.{i}'
        x=x+cross(w,w.norm(x,p+'.cross_attn_norm'),context,p+'.cross_attention')
        y=w.norm(x,p+'.self_attn_norm');x=x+cross(w,y,y,p+'.self_attention')
        x=x+ff(w,w.norm(x,p+'.ffn_norm'),p+'.ffn');w.release(p+'.')
    return w.norm(x,'perceiver.norm_out')

def rope_table(plane,h,w):
    positions=torch.stack(torch.meshgrid(torch.arange(h),torch.arange(w),indexing='ij'),-1).reshape(-1,2)
    positions=torch.cat((torch.full((h*w,1),plane),positions),-1)
    phases=torch.cat([positions[:,i,None].double()*256**(-torch.arange(0,d,2).double()/d) for i,d in enumerate((32,48,48))],-1).float().cuda()
    return phases.cos(),phases.sin()

def rope(x,c,s):
    original=x.shape;x=x.float().reshape(-1,28,64,2);a,b=x.unbind(-1)
    return torch.stack((a*c[:,None]-b*s[:,None],b*c[:,None]+a*s[:,None]),-1).reshape(original).bfloat16()

def block(w,x,p,time,start,end,c,s,context=None):
    m=w.linear(time,p+'.adaLN_modulation.0');scale,gate,scaleff,gateff=m.chunk(4,-1)
    idx=torch.zeros(x.shape[0],dtype=torch.long,device=x.device);idx[start:end]=1
    y=w.norm(x,p+'.attention_norm1')*(1+scale[idx]);q,k,v=w.linear(y,p+'.attention.qkv').chunk(3,-1)
    q=w.norm(q.reshape(-1,28,128),p+'.attention.q_norm').reshape(-1,3584)
    k=w.norm(k.reshape(-1,28,128),p+'.attention.k_norm').reshape(-1,3584)
    y=w.norm(w.linear(attention(rope(q,c,s),rope(k,c,s),v),p+'.attention.out'),p+'.attention_norm2')
    x=x+y*gate[idx].tanh()
    if context is not None:
        y=w.norm(cross(w,w.norm(x,p+'.cross_attn_norm1'),context,p+'.cross_attention'),p+'.cross_attn_norm2')
        x=torch.addcmul(x,y,w.get(p+'.cross_attn_gate').tanh())
    y=w.norm(x,p+'.ffn_norm1')*(1+scaleff[idx]);y=w.norm(ff(w,y,p+'.feed_forward'),p+'.ffn_norm2')
    return x+y*gateff[idx].tanh()

@torch.no_grad()
def reference(model,data,mode):
    w=FiboWeights(model);context=data['prompt_embedding'].cuda().bfloat16().chunk(6,-1)
    gist=perceiver(w,context[0])
    if mode=='perceiver':return gist.float().cpu()
    latent=data['image_latents'];h,ww=latent.shape[:2];n=h*ww
    ref=data['reference_latents'];rh,rw=ref.shape[:2]
    sigma=float(data['sigma'][0]);freq=torch.exp(-torch.log(torch.tensor(10000.))*torch.arange(128)/128)
    phase=torch.tensor([0,sigma*1000])[:,None]*freq[None]
    time=w.linear(F.silu(w.linear(torch.cat((phase.cos(),phase.sin()),-1).cuda().bfloat16(),'t_embedder.mlp.0')),'t_embedder.mlp.2')
    img=w.linear(torch.cat((latent.reshape(-1,128),ref.reshape(-1,128))).cuda().bfloat16(),'x_embedder')
    a,b=rope_table(513,h,ww);cc,ss=rope_table(514,rh,rw);c=torch.cat((a,cc));s=torch.cat((b,ss))
    for i in range(2):
        p=f'noise_refiner.{i}';img=block(w,img,p,time,0,n,c,s);w.release(p+'.')
    gc,gs=zip(*[rope_table(i+1,1,1) for i in range(256)]);c=torch.cat((*gc,c));s=torch.cat((*gs,s));x=torch.cat((gist,img))
    for i in range(44):
        p=f'layers.{i}';ctx=context[(i-6)//8+1] if i in (6,14,22,30,38) else None
        x=block(w,x,p,time,256,256+n,c,s,ctx);w.release(p+'.')
    x=F.layer_norm(x[256:256+n],(3584,),eps=1e-6)*(1+w.linear(F.silu(time[1:]),'final_layer.adaLN_modulation.1'))
    return w.linear(x,'final_layer.linear').float().reshape(h,ww,128).cpu()

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--probe',required=True);parser.add_argument('--model',required=True);parser.add_argument('--mode',choices=['perceiver','transformer'],default='transformer');parser.add_argument('--embedding');args=parser.parse_args()
    torch.manual_seed(41);torch.backends.cuda.matmul.allow_tf32=False
    out=Path('build/fibo-support');data={'prompt_embedding':torch.randn(17,46080),'image_latents':torch.randn(3,4,128),'reference_latents':torch.randn(2,3,128),'sigma':torch.tensor([.75])}
    if args.embedding:data['prompt_embedding']=load_file(args.embedding)['output']
    expected=reference(args.model,data,args.mode);torch.cuda.empty_cache()
    fixture=out/(args.mode+'-input.safetensors');actual=out/(args.mode+'-native.safetensors');save_file(data,str(fixture));save_file({'output':expected.contiguous()},str(out/(args.mode+'-expected.safetensors')))
    subprocess.run([args.probe,args.mode,args.model,str(fixture),str(actual)],check=True)
    observed=load_file(str(actual))['output'];error=observed-expected;relative=float(error.square().mean().sqrt()/expected.square().mean().sqrt())
    print(args.mode,'max_abs',float(error.abs().max()),'relative_rms',relative,flush=True)
    if not torch.isfinite(observed).all() or relative>.04:raise SystemExit('Fibo comparison exceeds 4% relative RMS')
if __name__=='__main__':main()
