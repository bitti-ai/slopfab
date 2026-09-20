# Backend contracts and modules

Attention shape and semantics are described by `AttentionDescriptor` in
`slopfab/attention_descriptor.h`: separate query/KV token and head counts,
head width, token-major layout, mask, arithmetic requirement and scale.
Validation distinguishes a valid mathematical shape from an implemented
backend capability. The shared exact text contract still requires 64 query
heads, 8 KV heads and width 128; CUDA and Vulkan retain their additional
indexing, device, driver, format and sequence limits.

CUDA `AttentionPlan::compile(descriptor, policy)` freezes the descriptor,
execution policy and required workspace. The compatibility overload takes
`AttentionConfig`, KV-head count and backend. Use the same plan's
`workspace_bytes()` and `forward()` methods. Blocked GQA reserves compact K/V
buffers instead of full query-head width. Sage plans also remember their
compiling device because its kernel variant changes workspace requirements.
Output must not alias inputs, and execution checks available scratch before
launch. The older free functions remain available.

Pinned exact arithmetic is separate from numerically equivalent blocked/fused
attention. The general CUDA plan rejects the exact contract; existing exact
operator implementations continue to enforce the serialized scale bits and
operation order. Frame banding, Sage and Sol require approximate arithmetic.
These descriptors do not advertise causal support in the general CUDA plan:
causal text attention still uses its dedicated validated implementation.

Vulkan contexts accept `TensorContextOptions::pipeline_sets`. Core operators
are always prepared; codec/reference contexts select their required operator
families, while the default preserves all historical families except reference
encoding. `prepare_pipeline_sets(device, sets)` adds families between batches,
never while a batch is recording. Preparing a family does not enable arithmetic
unsupported by the device. The old `enable_reference_encoder` flag remains a
compatibility alias. Exact, Flash, Sage and text attention are independent sets.

Identical SPIR-V plus entry point, bindings, local dimensions, push constant
size and specialization constants reuse a pipeline on the same device.
Cache entries are weak: they do not keep device/pipeline ownership cycles alive
or retain unused pipelines after their owners are gone. Failed construction
never publishes a cache entry. `Device::pipeline_cache_hits()` exposes reuse
for tests/diagnostics.

The Vulkan implementation now separates device discovery, buffer allocation,
pipeline construction and compute submissions. Tensor operations are separate
translation units for storage, pipelines, weights, GEMM, attention,
normalization, pointwise, conditioning, DiT, reference and audio operations.
Private state/recording headers retain the transactional access rollback and
resource ownership rules. CUDA attention separates blocked kernels, fused
kernels and planning; text execution separates kernels, layer execution and
encoder residency/orchestration. Kernel bodies and numerical operation order
are unchanged by these module extractions.
