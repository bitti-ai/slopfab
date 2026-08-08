# Sol correction semantics audit

The native Sol path was compared with NVlabs/Sana `sol_attn` preprocessing,
the Triton reference forward kernel, and the released SM120 mainloop.

| Operation | Released implementation | Native implementation | Result |
|---|---|---|---|
| K summary | valid-token block mean | `pool_kv`: valid-token mean | Match |
| V summary | valid-token block sum | `pool_kv`: FP32 sum, BF16 for pipeline MMA | Match |
| Route score | mean query-row score against KC | `qmean dot km` | Match |
| Score units | score and threshold multiplied by `scale * log2(e)` | both multiplied by natural softmax `scale` | Equivalent |
| Threshold | mean plus beta times diagonal standard deviation | same diagonal population estimator | Match, except released code adds `1e-6` in log2 variance units |
| Local route | query/key block distance at most one | same | Match |
| Approximate numerator | `exp(score) * VC` once | same | Match |
| Approximate denominator | `exp(score) * valid_block_length` | same, including ragged tail | Match |
| Exact route | remove approximate contribution and evaluate full K/V block | exact and approximate IDs are disjoint | Match |
| Online softmax | common row max/sum across approximate and exact groups | common FP32 row max/sum; groups reordered | Mathematically equivalent; all-exact error is 0.044% rel-L2 |

`exact_prefix` is implemented slightly more conservatively than Sana's sink
range: native code makes prefix K blocks exact and also makes prefix Q blocks
fully exact. This increases work and accuracy and cannot explain the observed
error. H3 Q/K normalization occurs before capture and both compared backends
consume identical captured tensors.

On `h3-step10-layer2.solqkv`, beta 1 gives the same error in both native
implementations:

- pipeline: rel-L2 0.04166411, cosine 0.999156451, max error 11.98438;
- scalar fallback: rel-L2 0.04166197, cosine 0.999156465, max error 11.98438;
- all-exact control: rel-L2 0.00043606, cosine 0.999999905, max error 0.25.

Therefore the material error is the intended zeroth-order centroid
approximation, not grouped correction, persistent fragments, or normalization.
The beta-1 output itself is finite, but repeated 4% layer errors are not a safe
quality contract for iterative denoising. The backend must remain disabled by
default until an end-to-end-validated hybrid policy exists.
