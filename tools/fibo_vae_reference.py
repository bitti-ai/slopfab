"""Compare local FLUX.2 VAE encode/decode with Diffusers FP32 modules."""
from pathlib import Path
import re,subprocess
import torch
from torch.nn import functional as F
from safetensors.torch import load_file,save_file
from diffusers.models.autoencoders.vae import Encoder,Decoder
from PIL import Image
import numpy as np

def rename(name):
    name=name.replace('encoder.quant_conv','quant_conv').replace('decoder.post_quant_conv','post_quant_conv')
    name=name.replace('.mid.block_1.','.mid_block.resnets.0.').replace('.mid.block_2.','.mid_block.resnets.1.').replace('.mid.attn_1.','.mid_block.attentions.0.')
    name=re.sub(r'decoder.up.(\d+).',lambda m:f'decoder.up_blocks.{3-int(m[1])}.',name)
    name=re.sub(r'encoder.down.(\d+).',lambda m:f'encoder.down_blocks.{m[1]}.',name)
    name=name.replace('.block.','.resnets.').replace('.upsample.','.upsamplers.0.').replace('.downsample.','.downsamplers.0.').replace('.norm_out.','.conv_norm_out.').replace('.nin_shortcut.','.conv_shortcut.')
    if '.attentions.' in name:
        for a,b in (('.norm.','.group_norm.'),('.q.','.to_q.'),('.k.','.to_k.'),('.v.','.to_v.'),('.proj_out.','.to_out.0.')):name=name.replace(a,b)
    return name
@torch.no_grad()
def main():
    root=Path('build/fibo-support');model='weights/vae/flux2-vae.safetensors';state={rename(k):v.cuda().float() for k,v in load_file(model).items()}
    for k,v in list(state.items()):
        if '.attentions.' in k and v.ndim==4:state[k]=v[:,:,0,0]
    widths=tuple(state[f'encoder.down_blocks.{i}.resnets.0.conv1.weight'].shape[0] for i in range(4))
    encoder=Encoder(in_channels=3,out_channels=32,down_block_types=('DownEncoderBlock2D',)*4,block_out_channels=widths,layers_per_block=2,norm_num_groups=32,double_z=True).cuda().float()
    decoder=Decoder(in_channels=32,out_channels=3,up_block_types=('UpDecoderBlock2D',)*4,block_out_channels=widths,layers_per_block=2,norm_num_groups=32).cuda().float()
    encoder.load_state_dict({k[8:]:v for k,v in state.items() if k.startswith('encoder.')});decoder.load_state_dict({k[8:]:v for k,v in state.items() if k.startswith('decoder.')})
    image=torch.from_numpy(np.array(Image.open(root/'source.ppm'))).permute(2,0,1)[None].cuda().float()/127.5-1
    z=F.conv2d(encoder(image),state['quant_conv.weight'],state['quant_conv.bias'])[:,:32]
    b,c,h,w=z.shape;packed=z.reshape(b,c,h//2,2,w//2,2).permute(0,1,3,5,2,4).reshape(b,128,h//2,w//2)
    packed=(packed-state['bn.running_mean'][None,:,None,None])/(state['bn.running_var'][None,:,None,None]+1e-4).sqrt()
    expected=packed[0].permute(1,2,0).contiguous().cpu();observed=load_file(str(root/'encoded.safetensors'))['output']
    error=float((observed-expected).square().mean().sqrt()/expected.square().mean().sqrt());print('VAE encode relative RMS',error,flush=True);assert error<.001
    save_file({'image_latents':expected},str(root/'vae-input.safetensors'))
    pixels=(decoder(F.conv2d(z,state['post_quant_conv.weight'],state['post_quant_conv.bias']))[0].clamp(-1,1)*.5+.5).cpu()
    subprocess.run(['build/Release/slopfab_fiboprobe.exe','vae',model,str(root/'vae-input.safetensors'),str(root/'vae-native.safetensors')],check=True)
    observed=load_file(str(root/'vae-native.safetensors'))['output'];error=float((observed-pixels).square().mean().sqrt()/pixels.square().mean().sqrt());print('VAE decode relative RMS',error,flush=True);assert error<.001
if __name__=='__main__':main()
