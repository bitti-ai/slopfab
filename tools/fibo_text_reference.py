"""Check Fibo Qwen3-VL conditioning with the Hugging Face Transformers model.
No downloads: tokenizer, config, input image and model are all local.
"""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch
from torch import nn
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
from tokenizers import Tokenizer
from safetensors.torch import load_file,save_file
from moss_reference import Weights,rotate

@torch.no_grad()
def main():
    root=Path('build/fibo-support');w=Weights('weights/text_encoder/qwen3vl_4b_int8_convrot.safetensors')
    cfg=Qwen3VLConfig.from_dict(json.loads((root/'qwenconfig.json').read_text()));cfg._attn_implementation='sdpa'
    with torch.device('meta'):model=Qwen3VLModel(cfg)
    class Linear(nn.Module):
        def __init__(self,name):super().__init__();self.name=name
        def forward(self,x):
            out=w.linear(x,self.name)
            w.cache={k:v for k,v in w.cache.items() if not k.startswith(self.name+'.')}
            return out
    for name,module in list(model.named_modules()):
        if isinstance(module,nn.Linear):
            parent,_,leaf=name.rpartition('.');setattr(model.get_submodule(parent),leaf,Linear('model.'+name))
    class Embedding(nn.Module):
        def forward(self,ids):
            key='model.language_model.embed_tokens'
            value=w.file.get_tensor(key+'.weight')[ids.cpu()].float()
            scale=w.file.get_tensor(key+'.weight_scale')[ids.cpu()]
            return (rotate(value,256)*scale).cuda().bfloat16()
    model.language_model.embed_tokens=Embedding()
    for name,param in list(model.named_parameters()):
        key='model.'+name
        value=w.file.get_tensor(key)
        if name.endswith('embed_tokens.weight'):
            scale=w.file.get_tensor(key[:-7]+'.weight_scale');config=json.loads(bytes(w.file.get_tensor(key[:-7]+'.comfy_quant').tolist()))
            value=rotate(value.float(),config.get('convrot_groupsize',256))*scale
        parent,_,leaf=name.rpartition('.');setattr(model.get_submodule(parent),leaf,nn.Parameter(value.cuda().bfloat16(),requires_grad=False))
    # Meta construction leaves rotary buffers on meta; initialize those modules normally.
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding,Qwen3VLVisionRotaryEmbedding
    model.language_model.rotary_emb=Qwen3VLTextRotaryEmbedding(cfg.text_config,device='cuda')
    model.visual.rotary_pos_emb=Qwen3VLVisionRotaryEmbedding(32).cuda()
    tokenizer=Tokenizer.from_file('ref/text_encoder/tokenizer.json')
    prompt='<|im_start|>user\n'+json.dumps({'image':'<|vision_start|><|image_pad|><|vision_end|>','edit_instruction':'Make the object blue'},separators=(',',':'))+'<|im_end|>\n<|im_start|>assistant\n'
    ids=[]
    for token in tokenizer.encode(prompt).ids:ids.extend([token]*4 if token==151655 else [token])
    from PIL import Image
    import numpy as np
    pixels=torch.from_numpy(np.array(Image.open(root/'source.ppm'))).float()/255
    pixels=((pixels-.5)/.5).permute(2,0,1)[None].repeat(2,1,1,1)
    patches=pixels.reshape(1,2,3,2,2,16,2,2,16).permute(0,3,6,4,7,2,1,5,8).reshape(16,1536).cuda().bfloat16()
    vision=model.visual(patches,grid_thw=torch.tensor([[1,4,4]],device='cuda'))
    save_file({'output':torch.cat((vision[0],*vision[1]),-1).float().cpu().contiguous()},str(root/'vision-expected.safetensors'))
    # Supplying the padding mask prevents Transformers 4.57 from treating
    # repeated temporal image positions as separate packed text sequences.
    out=model(input_ids=torch.tensor([ids],device='cuda'),attention_mask=torch.ones((1,len(ids)),device='cuda',dtype=torch.long),pixel_values=patches,image_grid_thw=torch.tensor([[1,4,4]],device='cuda'),output_hidden_states=True,use_cache=False)
    save_file({str(i):x.float().cpu().contiguous() for i,x in enumerate(out.hidden_states)},str(root/'text-hidden.safetensors'))
    bundles=((9,20,31),(5,16,27),(7,18,29),(9,20,31),(11,22,33),(13,24,35))
    expected=torch.cat([torch.cat([(out.hidden_states[i].float()*torch.rsqrt(out.hidden_states[i].float().square().mean(-1,keepdim=True)+1e-6)).bfloat16() for i in b],-1) for b in bundles],-1)[0].float().cpu()
    save_file({'output':expected.contiguous()},str(root/'text-expected.safetensors'))
    observed=load_file(str(root/'text-native.safetensors'))['output'];error=observed-expected
    relative=float(error.square().mean().sqrt()/expected.square().mean().sqrt());print('Qwen3-VL relative RMS',relative,'max_abs',float(error.abs().max()),flush=True)
    if relative>.04:raise SystemExit('Qwen3-VL comparison exceeds 4% relative RMS')
if __name__=='__main__':main()
