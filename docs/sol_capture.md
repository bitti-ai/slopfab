# Sol-Attn Q/K/V capture and replay

Capture is disabled unless `VIDFAB_SOL_CAPTURE` names an output file. The
transformer checks the environment only at an attention call; it performs no
copies or synchronization on the normal path.

```powershell
$env:VIDFAB_SOL_CAPTURE = 'h3-step0-layer2.solqkv'
$env:VIDFAB_SOL_CAPTURE_STEP = '0'
$env:VIDFAB_SOL_CAPTURE_LAYER = '2'
# Run vidfab normally.
build-sol/Release/vidfab_solbench.exe --input h3-step0-layer2.solqkv --iters 10
```

Step and layer default to zero. Only the first matching call is written. Q and
K are captured after head RMSNorm and RoPE, immediately before attention; V is
the projection output consumed by attention.

The little-endian binary format begins with the 64-byte `SolCaptureHeader` from
`include/vidfab/sol_capture.h`, followed by three equally sized arrays: Q, K,
and V. Arrays are raw BF16 words in `[sequence][head][channel]` order. The
header records sequence length, heads, head dimension, Sol exact-prefix row,
denoise step, layer, and per-tensor element count. Version 1 replay currently
requires head dimension 128, matching the Sol kernel.

For a small format/replay smoke test without loading H3, `solbench --seq 128
--heads 2 --prefix 16 --save-input synthetic.solqkv` writes its deterministic
synthetic tensors in the same format. Replay them with `--input
synthetic.solqkv`.
