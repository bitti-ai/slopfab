"""Numerical YuE2 validation against PyTorch and the upstream Oobleck decoder.

Development only: requires torch, safetensors and transformers. No downloads.
"""
import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from torch.nn import functional as F
from safetensors.torch import load_file, save_file
from moss_reference import Weights, rotate

AR = "text_encoders.model."
NAR = "model.diffusion_model."

def norm(w, x, key):
    return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6).to(x.dtype) * w.get(key).to(x.dtype)

def layers(w, x, prefix, cache=None):
    def check(t,key):
        if not getattr(w,"trace",None):return
        path=w.trace/(("ar-" if cache is None else "nar-")+key+".safetensors")
        a=load_file(str(path))["output"].reshape(t.shape)
        b=t.detach().float().cpu();print(path.stem,"max",(a-b).abs().max().item(),"rms",(a-b).square().mean().sqrt().item(),flush=True)
    check(x,"input")
    count = x.shape[0]
    offset = 0 if cache is None else cache[0][0].shape[0]
    phase = torch.arange(offset, offset+count, device="cuda").float()[:, None] / 1e6**(torch.arange(64,device="cuda").float()/64)
    cs, sn = phase.cos().bfloat16()[:,None], phase.sin().bfloat16()[:,None]
    def rope(t):
        a,b=t.chunk(2,-1)
        return torch.cat((a*cs-b*sn,b*cs+a*sn),-1)
    saved=[]
    for i in range(28):
        p=f"{prefix}layers.{i}."
        q,k,v=w.linear(norm(w,x,p+"input_layernorm.weight"),p+"self_attn.qkv_proj").split((2048,1024,1024),-1)
        if i in (0,11):check(torch.cat((q,k,v),-1),"qkv"+str(i))
        q=rope(norm(w,q.reshape(count,16,128),p+"self_attn.q_norm.weight"))
        k=rope(norm(w,k.reshape(count,8,128),p+"self_attn.k_norm.weight"))
        if i in (0,11):check(q,"q"+str(i));check(k,"k"+str(i))
        v=v.reshape(count,8,128)
        saved.append((k,v))
        if cache is not None:
            k,v=torch.cat((cache[i][0],k)),torch.cat((cache[i][1],v))
        q,k,v=(t.transpose(0,1) for t in (q,k,v))
        k,v=k.repeat_interleave(2,0),v.repeat_interleave(2,0)
        scores=(q.float()@k.float().transpose(-1,-2))/128**.5
        if cache is None:
            scores=scores.masked_fill(torch.ones(count,count,device="cuda",dtype=torch.bool).triu(1),-torch.inf)
        # The native shared attention implementation materializes BF16 probabilities.
        attn=(scores.softmax(-1).bfloat16()@v).transpose(0,1).reshape(count,2048)
        x=x+w.linear(attn,p+"self_attn.o_proj")
        if i in (0,11):check(attn,"attn"+str(i));check(x,"residual"+str(i))
        g,u=w.linear(norm(w,x,p+"post_attention_layernorm.weight"),p+"mlp.gate_up_proj").chunk(2,-1)
        x=x+w.linear(F.silu(g)*u,p+"mlp.down_proj")
        check(x,"layer"+str(i))
    return x,saved

def reference(w,tokens,state,flow):
    table=w.file.get_tensor(AR+"embed_tokens.weight")[tokens].cuda().float()
    scales=w.file.get_tensor(AR+"embed_tokens.weight_scale")[tokens].cuda()
    x=(rotate(table)*scales).bfloat16()
    h,cache=layers(w,x,AR)
    if not flow:
        return w.linear(norm(w,h,AR+"norm.weight"),AR+"lm_head")
    state=state.cuda().bfloat16()
    h=w.linear(F.pad(state,(0,0,1,1)),NAR+"vae2llm")
    freqs=torch.exp(-torch.log(torch.tensor(10000.,device="cuda"))*torch.arange(128,device="cuda").float()/128)
    args=.5*freqs
    t=w.linear(F.silu(w.linear(torch.cat((args.cos(),args.sin()))[None].bfloat16(),NAR+"time_embedder.mlp.0")),NAR+"time_embedder.mlp.2")
    h=h+t
    h=h+w.get(NAR+"latent_pos_embed.pe")[:h.shape[0]]
    h,_=layers(w,h,NAR+"model.",cache)
    return w.linear(norm(w,h,NAR+"model.norm.weight"),NAR+"llm2vae")[1:-1]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",required=True);p.add_argument("--probe",required=True)
    p.add_argument("--vae-source",required=True);p.add_argument("--out",required=True)
    p.add_argument("--trace",action="store_true")
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(7)
    tokens=[151643,9707,151847,151848,151851,151853,151852]
    state=torch.randn(3,64)*.1
    fixture=out/"input.safetensors"
    save_file({"tokens":torch.tensor(tokens,dtype=torch.float32),"audio_latents":state},str(fixture))
    for mode in ("ar","flow","vae"):
        dest=out/(mode+".safetensors")
        subprocess.run([a.probe,mode,a.model,str(fixture),str(dest)]+([str(out)] if a.trace else []),check=True)
        actual=load_file(str(dest))["output"]
        if mode=="vae":
            spec=importlib.util.spec_from_file_location("yue_vae",a.vae_source)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            decoder=module.OobleckDecoder(out_channels=2,channels=64,latent_dim=64,c_mults=[1,2,4,8,16,32],strides=[2,2,4,4,5,6],use_snake=True,final_tanh=False).cuda().float().eval()
            from safetensors import safe_open
            with safe_open(a.model,framework="pt") as f:
                decoder.load_state_dict({k[len("vae.decoder."):]:f.get_tensor(k).float() for k in f.keys() if k.startswith("vae.decoder.")})
            with torch.no_grad():expected=decoder(state.T[None].cuda())[0].T.cpu()
            del decoder
        else:
            w=Weights(a.model)
            if a.trace:w.trace=out
            with torch.no_grad():expected=reference(w,tokens,state,mode=="flow").float().cpu()
            del w
        diff=(expected-actual).float()
        relative=diff.square().mean().sqrt()/expected.square().mean().sqrt().clamp_min(1e-9)
        cosine=F.cosine_similarity(expected.flatten()[None],actual.flatten()[None]).item()
        print(mode,"max",diff.abs().max().item(),"relative RMS",relative.item(),"cosine",cosine,flush=True)
        assert relative < (.03 if mode!="vae" else .002), (mode,relative)
        assert torch.isfinite(actual).all()
        if mode=="ar":
            expected_top=expected[-1].topk(5)
            actual_top=actual[-1].topk(5)
            print("AR top logits",expected_top,actual_top,flush=True)
            # BF16 reduction order can swap near-tied logits. The relative RMS
            # check above covers the entire vocabulary; also inspect the head.
            assert actual_top.indices[0] in expected_top.indices
        torch.cuda.empty_cache()

if __name__=="__main__":main()
