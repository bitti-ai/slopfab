"""Check real packed W4A8/W6A8 transformer projections against independent PyTorch INT8 equations."""
import argparse,sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch
from safetensors import safe_open
from safetensors.torch import load_file,save_file
import subprocess
from moss_reference import rotate
@torch.no_grad()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',required=True);p.add_argument('--probe',required=True);a=p.parse_args()
 directory=Path('build/w4a8-support');directory.mkdir(exist_ok=True)
 with safe_open(a.model,framework='pt',device='cpu') as st:
  for name in ('blocks.0.attn.qkv_proj','blocks.0.attn.out_proj','blocks.0.mlp.fc1','blocks.0.mlp.fc2'):
   local=name+'.comfy_quant'
   state=json.loads(bytes(st.get_tensor(local).tolist())) if local in st.keys() else json.loads(st.metadata()['_quantization_metadata'])['layers'][name]
   params=state.get('params',state);group_size=params['group_size'];bits=6 if state['format']=='w6a8_int8' else 4
   packed=st.get_tensor(name+'.weight').cuda().view(torch.uint8);out,width=packed.shape;k=width*8//bits
   low=packed[:,:k//2];codes=torch.stack((low&15,low>>4),-1).reshape(out,k).long()
   if bits==6:
    high=packed[:,k//2:].long();high=torch.stack(tuple((high>>shift)&3 for shift in (0,2,4,6)),-1).reshape(out,k)
    codes|=high<<4;values=codes.float()-32;del high
   else:
    book=st.get_tensor(name+'.weight_codebook').cuda();values=book[codes]
   groups=st.get_tensor(name+'.weight_s_rel').cuda().float().repeat_interleave(group_size,-1)
   weights=(values*groups).round().clamp(-127,127).to(torch.int8);del codes,groups,packed,values,low
   channel=st.get_tensor(name+'.weight_s_channel').cuda()
   torch.manual_seed(24);x=torch.randn(32,k).cuda().bfloat16();x[0]=0;x[1,0]=16
   rotated=rotate(x);scale=rotated.float().abs().amax(-1).div(127).clamp_min(1e-30)
   quant=(rotated/scale.bfloat16()[:,None]).round().clamp(-128,127).to(torch.int8)
   accum=torch._int_mm(quant,weights.T.contiguous())
   expected=(accum.float()*scale[:,None]*channel[None]).bfloat16().float().cpu()
   fixture=directory/(name+'-input.safetensors');result=directory/(name+'-output.safetensors');save_file({'input':x.float().cpu()},str(fixture))
   subprocess.run([a.probe,a.model,name,str(fixture),str(result)],check=True)
   actual=load_file(str(result))['output'];error=actual-expected;relative=float(error.square().mean().sqrt()/expected.square().mean().sqrt())
   print(name,'relative_rms',relative,'max_abs',float(error.abs().max()),flush=True)
   assert torch.isfinite(actual).all() and relative<1e-5
   del weights,x,rotated,quant,accum;torch.cuda.empty_cache()
if __name__=='__main__':main()
