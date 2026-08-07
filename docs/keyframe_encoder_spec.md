# MiniMax H3 single-image encoder

The Ref2VA image path is the causal H3 video VAE encoder specialized to one
frame. Its checkpoint graph is `encoder.conv_in`, six two-block residual
levels with channels `128,256,256,512,512,1024`, `encoder.norm_out`,
`encoder.conv_out`, then `quant_conv`. Spatial strides are `2,2,2,2,1,1`;
temporal strides do not reduce a single frame. GroupNorm is time-isolated,
with 32 groups, affine parameters, epsilon `1e-6`, followed by SiLU.

The encoder produces 48 posterior-moment channels at `H/16 x W/16`.
Log-variance is clamped to `[-30,20]`, a diagonal Gaussian sample is taken,
the result is rounded through FP16, and each of the 24 channels is normalized
with the constants in the bundled VAE `config.json`. The one-frame latent is
then patchified in channel-major `1x2x2` order to 96-wide Ref2VA rows.

`vidfab_vaeprobe <checkpoint>` validates every encoder and `quant_conv`
tensor without paging the 5.2 GB archive into RAM.

## Remaining forward implementation

The repository currently has decoder-specific linear/attention CUDA kernels
and 1-D audio convolution, but no Conv3D or GroupNorm3D primitive. A usable
2048-pixel reference path cannot run the graph as scalar host convolution: its
512/1024-channel 3x3 spatial stages require a CUDA convolution backend. The
remaining implementation is therefore a CUDA FP16 Conv3D cross-correlation
(reflective spatial and causal temporal padding, asymmetric right/bottom pad
for stride-2 downsample), time-isolated GroupNorm+SiLU, residual add, and a
streaming weight uploader. The graph/weight and posterior/patch layout
contracts above are implemented and independently testable.

The reference distribution creates `torch.randn(mean.shape)` on the CPU and
then copies it to the parameter device. Bit-identical seed-42 sampling also
requires PyTorch's CPU normal-generator stream; the project's sampler
explicitly documents that its counter-based generator is not PyTorch-compatible.
`sample_keyframe_latents` therefore accepts the externally generated normal
field instead of silently substituting the diffusion-noise generator.
