"""Validate native MiniMax Music3 stages with PyTorch and upstream DAV.

Development only; needs torch, safetensors and the downloaded ComfyUI dav.py.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types
sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from safetensors import safe_open
from safetensors.torch import load_file,save_file
from moss_reference import Weights

def qnorm(w,x,key):
    return (x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-6)).to(x.dtype)*w.get(key).to(x.dtype)

def attention(q,k,v,heads,kvheads,causal=False):
    n=q.shape[0];dim=q.shape[1]//heads
    q=q.reshape(n,heads,dim).transpose(0,1)[None]
    k,v=(t.reshape(t.shape[0],kvheads,dim).transpose(0,1)[None].repeat_interleave(heads//kvheads,1) for t in (k,v))
    return F.scaled_dot_product_attention(q,k,v,is_causal=causal)[0].transpose(0,1).reshape(n,-1)

def global_reference(w,tokens):
    x=w.get("model.embed_tokens_prefill.weight")[tokens].bfloat16()
    n=len(tokens);phase=torch.arange(n,device="cuda").float()[:,None]/1e6**(torch.arange(64,device="cuda").float()/64)
    cs,sn=phase.cos().bfloat16()[:,None],phase.sin().bfloat16()[:,None]
    def rope(x):
        a,b=x.chunk(2,-1);return torch.cat((a*cs-b*sn,b*cs+a*sn),-1)
    for i in range(36):
        p=f"model.layers.{i}."
        q,k,v=w.linear(qnorm(w,x,p+"input_layernorm.weight"),p+"self_attn.qkv_proj").split((4096,1024,1024),-1)
        q=rope(qnorm(w,q.reshape(n,32,128),p+"self_attn.q_norm.weight")).reshape(n,-1)
        k=rope(qnorm(w,k.reshape(n,8,128),p+"self_attn.k_norm.weight")).reshape(n,-1)
        x=x+w.linear(attention(q,k,v,32,8,True),p+"self_attn.o_proj")
        g,u=w.linear(qnorm(w,x,p+"post_attention_layernorm.weight"),p+"mlp.gate_up_proj").chunk(2,-1)
        x=x+w.linear(F.silu(g)*u,p+"mlp.down_proj")
    return qnorm(w,x[-1:],"model.norm.weight")

def depth_reference(w,sequence):
    x=sequence.cuda().bfloat16()+w.get("model.audio_decoder.pos_embedding.weight")[:len(sequence)]
    def norm(x,key):return F.rms_norm(x,(4096,),w.get(key),1e-6)
    for i in range(4):
        p=f"model.audio_decoder.layers.{i}."
        q,k,v=w.linear(norm(x,p+"input_layernorm.weight"),p+"self_attn.qkv_proj").chunk(3,-1)
        x=x+w.linear(attention(q,k,v,16,16,True),p+"self_attn.o_proj")
        g,u=w.linear(norm(x,p+"post_attention_layernorm.weight"),p+"mlp.gate_up_proj").chunk(2,-1)
        x=x+w.linear(F.silu(g)*u,p+"mlp.down_proj")
    return norm(x[-1:],"model.audio_decoder.norm.weight")

def condition_reference(w,hidden):
    x=hidden.cuda().bfloat16().reshape(-1,8,4096)
    weights=w.get("cond_layer_logits").bfloat16().softmax(0)
    x=torch.einsum("tlc,l->tc",x,weights)*w.get("cond_layer_scale").bfloat16()
    # Express the convolution as its im2col GEMM. cuDNN selects a different
    # BF16 accumulator order, which is a separate precision comparison.
    columns=F.pad(x.T[None],(1,1)).unfold(2,3,1).permute(0,2,1,3).reshape(len(hidden),12288)
    x=F.linear(columns,w.get("latent_conditioners.0.weight").bfloat16().reshape(2048,12288))
    x=x+w.get("latent_conditioners.0.bias").bfloat16()
    return F.interpolate(x.T[None],size=max(1,len(hidden)*441//128),mode="nearest")[0].T

def flow_reference(w,state,condition,math_attention=True):
    def check(x,key):
        if not getattr(w,"trace",None):return
        a=load_file(str(w.trace/("flow-"+key+".safetensors")))["output"];b=x.detach().float().cpu()
        print(key,"max",(a-b).abs().max().item(),"rms",(a-b).square().mean().sqrt().item(),flush=True)
    p="diffusion_transformer.";t=p+"transformer."
    x=torch.cat((state.cuda().bfloat16(),torch.zeros_like(state,device="cuda",dtype=torch.bfloat16),condition),-1)
    x=F.linear(x,w.get(p+"preprocess_conv.weight").bfloat16()[:,:,0])+x
    check(x,"preprocess")
    time=torch.tensor([.5],device="cuda",dtype=torch.bfloat16)
    angles=2*torch.pi*time[:,None]@w.get(p+"timestep_features.weight").bfloat16().T
    emb=w.linear(F.silu(w.linear(torch.cat((angles.cos(),angles.sin()),-1),p+"to_timestep_embed.0")),p+"to_timestep_embed.2")
    x=torch.cat((emb,w.linear(x,t+"project_in")))
    check(x,"input")
    n=x.shape[0];phase=(torch.arange(n,device="cuda").float()[:,None]*w.get(t+"rotary_pos_emb.inv_freq").float()).bfloat16()
    cs,sn=phase.cos()[:,None],phase.sin()[:,None]
    def rope(q):
        q=q.reshape(n,32,64);a,b=q[...,:16],q[...,16:32]
        return torch.cat((a*cs-b*sn,b*cs+a*sn,q[...,32:]),-1).reshape(n,-1)
    def norm(x,key):return F.layer_norm(x,(2048,),w.get(key+".gamma").bfloat16(),w.get(key+".beta").bfloat16(),1e-5)
    for i in range(36):
        b=f"{t}layers.{i}."
        normalized=norm(x,b+"pre_norm")
        if i==0:check(normalized,"norm0")
        qkv=w.linear(normalized,b+"self_attn.to_qkv");q,k,v=qkv.chunk(3,-1)
        # The native flow path keeps softmax probabilities and PV in FP32.
        # cuDNN's BF16 SDPA accumulates different rounding over 36 layers.
        if math_attention:
            with sdpa_kernel(SDPBackend.MATH):
                attn=attention(rope(q),rope(k),v,32,32)
        else:
            attn=attention(rope(q),rope(k),v,32,32)
        if i==0:check(qkv,"qkv0");check(rope(q),"q0");check(attn,"attn0")
        x=x+w.linear(attn,b+"self_attn.to_out")
        value,gate=w.linear(norm(x,b+"ff_norm"),b+"ff.ff.0.proj").chunk(2,-1)
        x=x+w.linear(value*F.silu(gate),b+"ff.ff.2")
        check(x,"layer"+str(i))
    x=w.linear(x[1:],t+"project_out")
    return F.linear(x,w.get(p+"postprocess_conv.weight").bfloat16()[:,:,0])+x

def vae_reference(path,source,state):
    comfy=types.ModuleType("comfy");ops=types.ModuleType("comfy.ops")
    ops.cast_to_input=lambda value,x:value.to(device=x.device,dtype=x.dtype)
    comfy.ops=ops;sys.modules["comfy"]=comfy;sys.modules["comfy.ops"]=ops
    spec=importlib.util.spec_from_file_location("music3_dav",source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    operations=types.SimpleNamespace(Conv1d=nn.Conv1d,ConvTranspose1d=nn.ConvTranspose1d)
    model=module.MiniMaxMusic3DAV(dtype=torch.float32,device="cuda",operations=operations).eval()
    with safe_open(path,framework="pt") as f:
        weights={k.replace(".weight_v",".parametrizations.weight.original1").replace(".weight_g",".parametrizations.weight.original0"):f.get_tensor(k).float() for k in f.keys()}
    model.load_state_dict(weights)
    return model(state.T[None].cuda().float())[0].T

def main():
    p=argparse.ArgumentParser()
    for arg in ("ar","dit","vae","vae-source","probe","out"):p.add_argument("--"+arg,required=True)
    p.add_argument("--modes",default="prompt,global,depth,condition,flow,vae")
    p.add_argument("--trace",action="store_true")
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.manual_seed(13)
    tokens=[151644,151671,9707,151672,151673,151674,151645,151669]
    hidden=torch.randn(4,32768)*.2;sequence=torch.randn(4,4096)*.2;state=torch.randn(13,128)*.1
    fixture=out/"input.safetensors";save_file({"tokens":torch.tensor(tokens,dtype=torch.float32),"hidden":hidden,"sequence":sequence,"audio_latents":state},str(fixture))
    stats={}
    for mode in a.modes.split(","):
        model=a.ar if mode in ("prompt","global","depth") else a.vae if mode=="vae" else a.dit
        dest=out/(mode+".safetensors");subprocess.run([a.probe,"mm-"+mode,model,str(fixture),str(dest)]+([str(out)] if a.trace else []),check=True)
        if mode=="prompt":
            from tokenizers import Tokenizer
            with safe_open(model,framework="pt") as f:
                tokenizer=Tokenizer.from_str(bytes(f.get_tensor("tokenizer_json").numpy()).decode())
            spec=importlib.util.spec_from_file_location("music3_prompt",Path(a.vae_source).with_name("prompt.py"))
            prompt=importlib.util.module_from_spec(spec);spec.loader.exec_module(prompt)
            expected=tokenizer.encode(prompt.build_prompt("# Warm **piano**\n<|BPM 90|>","[Verse] Hello\n [Chorus] World")).ids
            actual=load_file(str(dest))["output"].int().tolist()
            assert actual==expected,(actual,expected)
            stats[mode]={"token_count":len(actual),"exact_match":True}
            print(mode,stats[mode],flush=True)
            continue
        with torch.no_grad():
            if mode=="vae":expected=vae_reference(model,a.vae_source,state).cpu()
            else:
                w=Weights(model)
                if a.trace:w.trace=out
                if mode=="global":expected=global_reference(w,tokens)
                elif mode=="depth":expected=depth_reference(w,sequence)
                elif mode=="condition":expected=condition_reference(w,hidden)
                else:
                    condition=condition_reference(w,hidden)
                    expected=flow_reference(w,state,condition)
                    w.trace=None
                    alternate=flow_reference(w,state,condition,False)
                    drift=((expected.float()-alternate.float()).square().mean()/expected.float().square().mean()).sqrt().item()
                    stats["pytorch_flow_attention_backend_relative_rms"]=drift
                    print("PyTorch math versus default SDPA flow relative RMS",drift,flush=True)
                expected=expected.float().cpu();del w
        actual=load_file(str(dest))["output"];diff=expected-actual
        relative=(diff.square().mean().sqrt()/expected.square().mean().sqrt().clamp_min(1e-9)).item()
        stats[mode]={"maximum_error":diff.abs().max().item(),"relative_rms":relative,"cosine":F.cosine_similarity(expected.flatten()[None],actual.flatten()[None]).item()}
        print(mode,stats[mode],flush=True)
        # BF16 rounding is amplified through the 36-layer flow stack. Compare
        # both independent PyTorch attention backends above as a precision baseline.
        assert relative<(.002 if mode=="vae" else .05 if mode=="flow" else .03),(mode,stats[mode])
        assert stats[mode]["cosine"]>.998
        assert torch.isfinite(actual).all()
        torch.cuda.empty_cache()
    (out/"metrics.json").write_text(json.dumps(stats,indent=2))

if __name__=="__main__":main()
