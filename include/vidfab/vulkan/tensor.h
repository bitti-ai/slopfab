#pragma once

#include <cstdint>
#include <memory>

#include "vidfab/device_tensor.h"
#include "vidfab/vulkan/compute.h"
#include "vidfab/vulkan/runtime.h"

namespace vidfab::vae {
struct AudioConv1DDesc;
struct AudioConvTranspose1DDesc;
}

namespace vidfab::vulkan {

class TensorContext;
class TensorBatch;
class LinearWeight;
class DenseGemmPlan;
class BlockedAttentionPlan;
class H3AttentionPlan;
class H3AttentionRanges;
class CausalGQAAttentionPlan;
class PreparedAttentionInputs;
class PreparedAttentionView;
class PreparedF16Activation;
class PreparedNVFP4WeightView;
class StreamedNVFP4WeightCache;

struct TensorContextOptions {
  // Two slots let a producer record the next bounded graph chunk while the
  // previous timeline submission is still executing.
  uint32_t max_in_flight = 2;
  // Bounded command/rollback capacity for one device-resident graph chunk.
  // The default preserves the small primitive-test footprint; the 36-block
  // video-VAE graph requests 1024 so it can record without per-block submits.
  uint32_t max_batch_operators = 32;
};

class DeviceTensor {
 public:
  DeviceTensor();
  ~DeviceTensor();
  DeviceTensor(DeviceTensor&&) noexcept;
  DeviceTensor& operator=(DeviceTensor&&) noexcept;
  DeviceTensor(const DeviceTensor&) = delete;
  DeviceTensor& operator=(const DeviceTensor&) = delete;

  DeviceTensorView view() const;
  const TensorLayout& layout() const;
  ScalarType type() const;
  explicit operator bool() const noexcept;

 private:
  struct Impl;
  explicit DeviceTensor(std::unique_ptr<Impl> impl);
  std::shared_ptr<Impl> impl_;
  friend class TensorContext;
  friend class TensorBatch;
  friend class LinearWeight;
  friend class DenseGemmPlan;
  friend class BlockedAttentionPlan;
  friend class H3AttentionPlan;
  friend class H3AttentionRanges;
  friend class CausalGQAAttentionPlan;
  friend class PreparedAttentionInputs;
  friend class PreparedF16Activation;
  friend class StreamedNVFP4WeightCache;
};

struct TensorUpload {
  DeviceTensor* destination = nullptr;
  const void* values = nullptr;
  uint64_t bytes = 0;
};

class TensorWorkspace final : public DeviceWorkspace {
 public:
  explicit TensorWorkspace(const Device& device, uint64_t block_bytes = 4ull << 20);
  ~TensorWorkspace() override;
  TensorWorkspace(TensorWorkspace&&) noexcept;
  TensorWorkspace& operator=(TensorWorkspace&&) noexcept;
  TensorWorkspace(const TensorWorkspace&) = delete;
  TensorWorkspace& operator=(const TensorWorkspace&) = delete;

  DeviceBackend backend() const noexcept override;
  void reserve(uint64_t bytes) override;
  WorkspaceSpan allocate(uint64_t bytes, uint64_t alignment = 256) override;
  void reset() noexcept override;
  uint64_t capacity() const noexcept override;
  uint64_t used() const noexcept override;
  uint64_t generation() const noexcept override;
  bool valid(const WorkspaceSpan& span) const noexcept override;
  uint64_t reserved_bytes() const noexcept;
  uint64_t pooled_used_bytes() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// A bounded device recording scope. Multiple operators record into one Vulkan
// command buffer and `submit` produces one exact timeline token. Tensor
// allocations are retained as they are recorded and then through completion.
// Dropping an unsubmitted batch discards commands and restores tensor access
// tracking.
class TensorBatch {
 public:
  TensorBatch();
  ~TensorBatch();
  TensorBatch(TensorBatch&&) noexcept;
  TensorBatch& operator=(TensorBatch&&) noexcept;
  TensorBatch(const TensorBatch&) = delete;
  TensorBatch& operator=(const TensorBatch&) = delete;

  void copy(DeviceTensor& source, DeviceTensor& destination);
  // Contiguous 2-D row-range transfer. Source/destination must have the same
  // scalar type and column count; ranges are in rows and must not overlap.
  void copy_rows(DeviceTensor& source, DeviceTensor& destination,
                 uint32_t source_row, uint32_t destination_row,
                 uint32_t rows);
  void add(DeviceTensor& a, DeviceTensor& b, DeviceTensor& output);
  void convert(DeviceTensor& source, DeviceTensor& destination);
  void transpose_2d(DeviceTensor& source, DeviceTensor& destination);
  // `indices` is a trusted device tensor: every int32 value must be in
  // [0, source.rows). The shader bounds-checks and writes zero for an invalid
  // value to prevent device memory access, but validation belongs at the
  // producer/callsite because checking device values here would add a host
  // synchronization boundary.
  void gather_rows(DeviceTensor& source, DeviceTensor& indices,
                   DeviceTensor& destination);
  // `indices` is trusted: values must be unique and in [0, destination.rows).
  // The shader bounds-checks invalid values and leaves those rows untouched;
  // uniqueness must be guaranteed by the producer. Untouched destination rows
  // are preserved.
  void scatter_rows(DeviceTensor& source, DeviceTensor& indices,
                    DeviceTensor& destination);
  // The fp32 arithmetic exactness contract below applies to add-bias too.
  void add_bias(DeviceTensor& input, DeviceTensor& bias, DeviceTensor& output);
  // Production Video-VAE pointwise fusions. All tensors are contiguous fp32.
  // Residual is deliberately in-place: x/y are [rows,columns], bias/scale are
  // [columns], and all four allocations are distinct. Arithmetic is a
  // separately rounded bias add followed by fma(biased_y, scale, x).
  void layer_scale_residual_f32(DeviceTensor& x, DeviceTensor& y,
                                DeviceTensor& bias, DeviceTensor& scale);
  // Input [rows,2*inner] stores the gate first and value second; bias has the
  // same doubled width and output is a distinct [rows,inner] allocation.
  void swiglu_bias_f32(DeviceTensor& input, DeviceTensor& bias,
                       DeviceTensor& output);
  // Channel-major [channels,voxels] input/output with mean/std [channels].
  // All four allocations are distinct; the operation is fma(z,std,mean).
  void latent_denorm_f32(DeviceTensor& input, DeviceTensor& mean,
                         DeviceTensor& std_dev, DeviceTensor& output);
  void heads_to_tokens_bf16(DeviceTensor& source, DeviceTensor& destination,
                            uint32_t heads, uint32_t sequence, uint32_t head_dim);
  void depth_to_space(DeviceTensor& source, DeviceTensor& destination,
                      uint32_t time, uint32_t height, uint32_t width,
                      uint32_t channels, uint32_t patch_time, uint32_t patch);
  // In-place BF16 rotary operations with explicit canonical fp32 table tensors.
  // H3 rotates fixed channels [0,96) as 48 half-split pairs and preserves the
  // tail; NeoX rotates the complete even-width head. Shapes are input
  // [rows,heads,head_dim], tables [rows,96] or [rows,head_dim].
  void rope_h3_bf16(DeviceTensor& input, DeviceTensor& cosine,
                     DeviceTensor& sine);
  void rope_neox_bf16(DeviceTensor& input, DeviceTensor& cosine,
                       DeviceTensor& sine);
  // Video-VAE fused split-QKV, bias, head-width-64 RMSNorm and partial-width-48
  // RoPE. QKV is [sequence,heads,192] or its GEMM-native contiguous flattening
  // [sequence,heads*192]; bias is correspondingly [heads,192] or [heads*192]. Tables
  // [sequence,48], and outputs [heads,sequence,64]. Tokens at and beyond
  // num_patches bypass rotation. All tensors are contiguous fp32 and distinct.
  void split_qkv_norm_rope_f32(DeviceTensor& qkv, DeviceTensor& bias,
                               DeviceTensor& cosine, DeviceTensor& sine,
                               DeviceTensor& q, DeviceTensor& k,
                               DeviceTensor& v, uint32_t num_patches,
                               float epsilon);
  // Matches the fp32 video-VAE CUDA reduction tree. Input/output are [rows,
  // dim], weight is [dim], and output may alias input. Epsilon must be finite
  // and positive.
  void rms_norm(DeviceTensor& input, DeviceTensor& weight, DeviceTensor& output,
                float epsilon);
  // As above, with biased variance and affine fp32 weight/bias [dim].
  void layer_norm(DeviceTensor& input, DeviceTensor& weight, DeviceTensor& bias,
                  DeviceTensor& output, float epsilon);
  // Shared transformer BF16 normalization. Shapes are [rows,dim] and [dim].
  // Output may alias input; the exact arithmetic domain is the same gated
  // zero/finite-normal domain as fp32 VAE normalization.
  void rms_norm_bf16(DeviceTensor& input, DeviceTensor& weight,
                     DeviceTensor& output, float epsilon);
  // Same arithmetic over each logical head of token-major [rows,heads*dim].
  // Input/output retain the projection-friendly 2-D layout while the norm
  // reduction is independently applied to rows*heads contiguous D-vectors.
  void rms_norm_heads_bf16(DeviceTensor& input, DeviceTensor& weight,
                           DeviceTensor& output, uint32_t heads,
                           uint32_t head_dim, float epsilon);
  void layer_norm_bf16(DeviceTensor& input, DeviceTensor& weight,
                       DeviceTensor& bias, DeviceTensor& output, float epsilon);
  // scale/shift are fp32 [mod_rows,dim], selectors are int32 [rows]. Invalid
  // device selectors are memory-safe and produce a zero output row.
  void rms_norm_modulate_bf16(DeviceTensor& input, DeviceTensor& weight,
                              DeviceTensor& scale, DeviceTensor& shift,
                              DeviceTensor& selectors, DeviceTensor& output,
                              float epsilon);
  void rms_norm_modulate_bf16_table(DeviceTensor& input, DeviceTensor& weight,
                                    DeviceTensor& tables,
                                    uint32_t mod_rows,
                                    uint32_t scale_table,
                                    uint32_t shift_table,
                                    DeviceTensor& selectors,
                                    DeviceTensor& output, float epsilon);
  void rms_norm_modulate_f32(DeviceTensor& input, DeviceTensor& weight,
                             DeviceTensor& scale, DeviceTensor& shift,
                             DeviceTensor& selectors, DeviceTensor& output,
                             float epsilon);
  // H3 DiT exact-mode pointwise seam. Residual and branch are BF16
  // [rows,dim], gate is fp32 [mod_rows,dim], and selectors is int32 [rows].
  // SwiGLU consumes BF16 [rows,2*inner] with gate first and writes a distinct
  // BF16 [rows,inner]. Invalid selectors leave a residual row unchanged.
  void dit_add_gated_bf16(DeviceTensor& residual, DeviceTensor& branch,
                          DeviceTensor& gate, DeviceTensor& selectors);
  void dit_add_gated_bf16_table(DeviceTensor& residual, DeviceTensor& branch,
                                DeviceTensor& tables, uint32_t mod_rows,
                                uint32_t gate_table,
                                DeviceTensor& selectors);
  void dit_swiglu_bf16(DeviceTensor& fused, DeviceTensor& output);
  // Qwen text-layer exact pointwise seams. Residual add operates in place on
  // contiguous BF16 [rows,dim]. Split SwiGLU consumes two distinct BF16
  // [rows,inner] tensors and writes a third. BF16 subnormals are flushed to
  // signed zero and NaNs are canonicalized before the final RNE conversion.
  void text_add_residual_bf16(DeviceTensor& residual, DeviceTensor& branch);
  void text_swiglu_split_bf16(DeviceTensor& gate, DeviceTensor& up,
                              DeviceTensor& output);
  // Exact Qwen vision GELU-tanh in place over contiguous BF16 [rows,dim].
  // Arithmetic, exceptional values, and the packed-pair tail are pinned to
  // launch_gelu_tanh_exact; the shipped CUDA fast GELU remains unchanged.
  void vision_gelu_tanh_bf16(DeviceTensor& activation);
  // Remaining exact vision layout seams. Position indices address table rows;
  // QKV splits fused [rows,3*dim] into three [rows,dim] tensors; merge-four is
  // the checkpoint's row-major [rows,dim] -> [rows/4,4*dim] reshape-copy.
  // Scatter-add requires unique nonnegative row indices, as produced by the
  // multimodal prompt layout.
  void vision_add_positions_bf16(DeviceTensor& activation,
                                 DeviceTensor& position_table,
                                 DeviceTensor& position_index);
  void vision_split_qkv_bf16(DeviceTensor& fused, DeviceTensor& query,
                             DeviceTensor& key, DeviceTensor& value);
  void vision_merge_four_bf16(DeviceTensor& input, DeviceTensor& output);
  // Raw BF16 row replacement used for image-pad embeddings. Unlike generic
  // scatter_rows (FP32), this preserves all source payload bits exactly.
  void vision_scatter_bf16(DeviceTensor& source,
                           DeviceTensor& destination,
                           DeviceTensor& row_index);
  void vision_scatter_add_bf16(DeviceTensor& source,
                               DeviceTensor& destination,
                               DeviceTensor& row_index);
  // Rank-R checkpoint-native AdaLN expansion. Weight [M*P*C,R], bias
  // [M*P*C], code [T,R]. Output may be canonical [P,T*M,C] or a flat fp32
  // arena with P equal aligned table strides.
  void dit_expand_adaln(DeviceTensor& weight, DeviceTensor& bias,
                        DeviceTensor& code, DeviceTensor& output,
                        uint32_t num_modality, uint32_t num_param,
                        uint32_t channels);
  // Exact H3 rectified-flow Euler update, in place over a contiguous fp32
  // tensor. Arithmetic matches FlowScheduler::step(kEuler).
  void dit_euler_step_f32(DeviceTensor& sample, DeviceTensor& velocity,
                          float sigma_from_timestep, float ratio);
  // Video/keyframe-VAE channel-major GroupNorm+SiLU. Input/output are fp32
  // contiguous [channels,height,width], affine parameters are fp16 [channels],
  // and output may alias input.
  void group_norm_silu_f16_affine(DeviceTensor& input, DeviceTensor& weight,
                                  DeviceTensor& bias, DeviceTensor& output,
                                  uint32_t groups, float epsilon);
  // Exact T=1 H3 keyframe-encoder Conv3D. Weight is checkpoint-native fp16
  // [Cout,Cin,Kt,Kh,Kw], while input/output are contiguous fp32 CHW. The
  // asymmetric stride-2 form uses constant-zero right/bottom padding.
  void keyframe_conv3d_f16(
      DeviceTensor& input, DeviceTensor& weight, DeviceTensor& bias,
      DeviceTensor& output, uint32_t in_channels, uint32_t out_channels,
      uint32_t input_height, uint32_t input_width, uint32_t kernel,
      uint32_t stride, bool reflect_padding, bool asymmetric_padding);
  // Exact audio-VAE fp32 primitives. Convolution tensors use contiguous NCT
  // layouts and the checkpoint-native weight layouts documented by the shared
  // descriptors. A null bias is permitted for bias-free residual convolutions.
  void audio_conv1d(DeviceTensor& input, DeviceTensor& weight,
                    DeviceTensor* bias, DeviceTensor& output,
                    const vae::AudioConv1DDesc& desc);
  void audio_conv_transpose1d(
      DeviceTensor& input, DeviceTensor& weight, DeviceTensor* bias,
      DeviceTensor& output, const vae::AudioConvTranspose1DDesc& desc);
  // Decoder-owned flat arenas may be larger than the current logical tensor.
  // `count` selects their live prefix; zero preserves the shaped primitive API.
  void audio_copy_prefix(DeviceTensor& source, DeviceTensor& destination,
                         uint64_t count);
  void audio_add_inplace(DeviceTensor& input_output, DeviceTensor& branch,
                         uint64_t count = 0);
  void audio_scale_inplace(DeviceTensor& input_output, float scale,
                           uint64_t count = 0);
  void audio_clamp_inplace(DeviceTensor& input_output, float lower, float upper,
                           uint64_t count = 0);
  void audio_interleave(DeviceTensor& planar, DeviceTensor& interleaved,
                        uint32_t batch, uint32_t frames);
  void audio_snake_beta_inplace(DeviceTensor& input_output,
                                DeviceTensor& log_alpha,
                                DeviceTensor& log_beta, uint32_t batch,
                                uint32_t channels, uint32_t length);
  void audio_aa_upsample_snake(
      DeviceTensor& input, DeviceTensor& filter, DeviceTensor& log_alpha,
      DeviceTensor& log_beta, DeviceTensor& output, uint32_t batch,
      uint32_t channels, uint32_t length_in);
  void audio_aa_downsample(DeviceTensor& input, DeviceTensor& filter,
                           DeviceTensor& output, uint32_t batch,
                           uint32_t channels, uint32_t length_in,
                           uint32_t length_out);
  Submission submit();
  uint32_t remaining_operator_capacity() const noexcept;
  // Host-only transactional preflight for graph orchestrators. Failure does
  // not poison or otherwise mutate the recording, so the same batch remains
  // usable for a smaller graph.
  void require_operator_capacity(uint32_t operators) const;
  bool belongs_to(const TensorContext& context) const noexcept;
  explicit operator bool() const noexcept;

 private:
  void rope_bf16(DeviceTensor& input, DeviceTensor& cosine,
                 DeviceTensor& sine, uint32_t mode);
  void materialize_linear_weight(const LinearWeight& weight,
                                 DeviceTensor& dense, bool fp16);
  void transform_linear_activation(const LinearWeight& weight,
                                   DeviceTensor& input, DeviceTensor& output,
                                   bool convrot);
  friend class LinearWeight;
  friend class DenseGemmPlan;
  friend class BlockedAttentionPlan;
  friend class H3AttentionPlan;
  friend class CausalGQAAttentionPlan;
  friend class PreparedAttentionInputs;
  friend class PreparedF16Activation;
  friend class StreamedNVFP4WeightCache;
  struct Impl;
  explicit TensorBatch(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
  friend class TensorContext;
};

// Persistent Vulkan tensor allocation, staging and elementary operators.
// Upload/download are explicit boundaries; copy/add operate device-to-device
// and do not stage through host memory. One TensorContext instance permits one
// active CPU recorder/boundary operation at a time; calls from different host
// threads require external synchronization. Submitted jobs use the configured
// bounded flight slots independently of that recorder lease.
class TensorContext {
 public:
  explicit TensorContext(const Device& device,
                         const TensorContextOptions& options = {});
  ~TensorContext();
  TensorContext(TensorContext&&) noexcept;
  TensorContext& operator=(TensorContext&&) noexcept;
  TensorContext(const TensorContext&) = delete;
  TensorContext& operator=(const TensorContext&) = delete;

  DeviceTensor allocate(const TensorLayout& layout,
                        ScalarType type = ScalarType::kFloat32);
  TensorBatch begin_batch();
  void upload(DeviceTensor& destination, const float* values, uint64_t count);
  // Loader-oriented upload whose host-visible staging allocation is released
  // after the synchronous transfer instead of raising retained staging usage.
  void upload_transient(DeviceTensor& destination, const float* values,
                        uint64_t count);
  void download(DeviceTensor& source, float* values, uint64_t count);
  void upload_bytes(DeviceTensor& destination, const void* values, uint64_t bytes);
  void upload_transient_bytes(DeviceTensor& destination, const void* values,
                              uint64_t bytes);
  // Several complete tensor uploads in one synchronous transfer submission.
  // Destinations must be distinct. Persistent staging grows to the packed
  // high-water once and is then reused without per-call allocation.
  void upload_batch(const TensorUpload* uploads, uint32_t count);
  void download_bytes(DeviceTensor& source, void* values, uint64_t bytes);
  // Exact self-copy and partial aliasing are rejected.
  void copy(DeviceTensor& source, DeviceTensor& destination);
  // Inputs may alias each other; output must be a distinct allocation.
  // Add and add-bias results are CUDA-bit-exact when inputs and the correctly rounded result
  // are zero, normal, or infinity. NaN payload arithmetic is not promised.
  void add(DeviceTensor& a, DeviceTensor& b, DeviceTensor& output);
  // True only when fp32 add/add-bias additionally cover subnormal
  // inputs/results. Call require_* before relying on that wider domain.
  bool full_fp32_arithmetic_exactness() const noexcept;
  void require_full_fp32_arithmetic_exactness() const;
  // Compatibility aliases for the original single-add primitive API.
  bool full_fp32_add_exactness() const noexcept {
    return full_fp32_arithmetic_exactness();
  }
  void require_full_fp32_add_exactness() const {
    require_full_fp32_arithmetic_exactness();
  }
  // Exact normalization currently requires a measured NVIDIA Vulkan tuple and
  // an explicitly enabled shaderInt64 feature.
  // The contract covers zero and finite normal inputs, affine values,
  // intermediates, epsilon and results. Subnormal and NaN arithmetic is
  // deliberately outside the exact domain; require this capability before
  // recording rms_norm/layer_norm.
  bool exact_normalization() const noexcept;
  void require_exact_normalization() const;
  bool exact_fp32_vae_normalization() const noexcept;
  void require_exact_fp32_vae_normalization() const;
  // Exact video-VAE pointwise operations canonicalize every subnormal
  // operand/intermediate/result to signed zero and every NaN to 0x7fc00000.
  // Finite normals, infinities and signed zeros retain their IEEE bits.
  bool exact_vae_pointwise() const noexcept;
  void require_exact_vae_pointwise() const;
  // Exact audio primitives additionally pin deterministic exp/sin/divide and
  // canonicalize subnormal/NaN values at every CUDA/Vulkan arithmetic seam.
  bool exact_audio_vae_primitives() const noexcept;
  void require_exact_audio_vae_primitives() const;
  // Exact blocked attention has its own capability contract even though the
  // currently measured tuple overlaps normalization. It additionally pins the
  // deterministic exp/divide shader and CUDA reference artifacts.
  bool exact_blocked_attention() const noexcept;
  void require_exact_blocked_attention() const;
  bool exact_h3_attention() const noexcept;
  void require_exact_h3_attention() const;
  bool exact_causal_gqa_attention() const noexcept;
  void require_exact_causal_gqa_attention() const;
  // Native block-scaled E2M1 cooperative MMA is deliberately separate from
  // streamed NVFP4->BF16 execution. It remains false until Vulkan exposes and
  // this backend implements an exact FP4 component/scale operand contract.
  bool native_nvfp4_gemm_available() const noexcept;
  void require_native_nvfp4_gemm() const;
  TensorWorkspace& workspace();
  uint64_t reserved_bytes() const;
  uint64_t pooled_used_bytes() const;
  // Persistent upload and readback buffers each have this capacity. They are
  // accounted separately from logical device tensors and retained for reuse.
  uint64_t staging_capacity_bytes() const noexcept;
  uint64_t descriptor_set_allocations() const noexcept;
  // Releases resources retained by already-complete bounded flight slots.
  // Does not wait for outstanding work.
  void collect();
  uint64_t storage_binding_alignment() const noexcept;
  bool owns(const DeviceTensor& tensor) const noexcept;

 private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
  friend class TensorBatch;
  friend class LinearWeight;
  friend class DenseGemmPlan;
  friend class BlockedAttentionPlan;
  friend class H3AttentionPlan;
  friend class H3AttentionRanges;
  friend class CausalGQAAttentionPlan;
  friend class PreparedAttentionInputs;
  friend class PreparedF16Activation;
  friend class StreamedNVFP4WeightCache;
};

// Exact, unmasked, memory-bounded BF16 attention reference. This deliberately
// excludes causal/GQA masks, frame bands, Sage2 and Sol routing; those are
// distinct arithmetic contracts rather than flags on this plan.
struct BlockedAttentionPlanDesc {
  uint32_t sequence = 0;
  uint32_t heads = 0;
  uint32_t head_dim = 0;
  // Required finite-normal positive bit pattern. Callers pin the model's host
  // scale calculation instead of asking a shader to reproduce host sqrt.
  float scale = 0.0f;
};

// One bounded Q/K/V FP16 preparation slot. Preparing records one device-only
// BF16->FP16 pass for all three tensors; its view is scoped to that batch so a
// discarded or superseded recording can never be consumed accidentally.
class PreparedAttentionInputs {
 public:
  PreparedAttentionInputs();
  ~PreparedAttentionInputs();
  PreparedAttentionInputs(PreparedAttentionInputs&&) noexcept;
  PreparedAttentionInputs& operator=(PreparedAttentionInputs&&) noexcept;
  PreparedAttentionInputs(const PreparedAttentionInputs&) = delete;
  PreparedAttentionInputs& operator=(const PreparedAttentionInputs&) = delete;
  static PreparedAttentionInputs create(TensorContext& context,
                                        const BlockedAttentionPlanDesc& desc);
  PreparedAttentionView prepare(TensorBatch& batch, DeviceTensor& query,
                                DeviceTensor& key, DeviceTensor& value);
  uint64_t reserved_bytes() const noexcept;
  explicit operator bool() const noexcept;
 private:
  struct Impl;
  explicit PreparedAttentionInputs(std::shared_ptr<Impl> impl);
  std::shared_ptr<Impl> impl_;
  friend class PreparedAttentionView;
  friend class BlockedAttentionPlan;
};

class PreparedAttentionView {
 public:
  PreparedAttentionView();
  ~PreparedAttentionView();
  PreparedAttentionView(PreparedAttentionView&&) noexcept;
  PreparedAttentionView& operator=(PreparedAttentionView&&) noexcept;
  PreparedAttentionView(const PreparedAttentionView&) = delete;
  PreparedAttentionView& operator=(const PreparedAttentionView&) = delete;
  explicit operator bool() const noexcept;
 private:
  explicit PreparedAttentionView(std::shared_ptr<void> slot,
                                 uintptr_t batch_id,
                                 uint64_t generation) noexcept;
  std::shared_ptr<void> slot_;
  uintptr_t batch_id_ = 0;
  uint64_t generation_ = 0;
  friend class PreparedAttentionInputs;
  friend class BlockedAttentionPlan;
};

class BlockedAttentionPlan {
 public:
  BlockedAttentionPlan();
  ~BlockedAttentionPlan();
  BlockedAttentionPlan(BlockedAttentionPlan&&) noexcept;
  BlockedAttentionPlan& operator=(BlockedAttentionPlan&&) noexcept;
  BlockedAttentionPlan(const BlockedAttentionPlan&) = delete;
  BlockedAttentionPlan& operator=(const BlockedAttentionPlan&) = delete;

  static BlockedAttentionPlan create(TensorContext& context,
                                     const BlockedAttentionPlanDesc& desc);
  const BlockedAttentionPlanDesc& description() const;
  // Original tensors are contiguous token-major [sequence,heads,head_dim]
  // finite BF16 and scale is exact_attention_scale(head_dim). NaN/Inf inputs
  // are outside this exact contract. Output must be distinct from all original
  // and prepared tensors. A row range allows multiple consumers in the same
  // bounded batch without repeating Q/K/V conversion.
  void record(TensorBatch& batch, PreparedAttentionView& inputs,
              DeviceTensor& output,
              uint32_t query_row_offset = 0, uint32_t rows = 0,
              uint32_t output_row_offset = 0) const;
  explicit operator bool() const noexcept;

 private:
  struct Impl;
  explicit BlockedAttentionPlan(std::shared_ptr<Impl> impl);
  std::shared_ptr<Impl> impl_;
};

// Exact H3 full/frame-banded attention is deliberately separate from the
// blocked FP16 plan. Q/K/V remain direct BF16 tensors; QK follows ascending
// 16-channel cooperative tiles with empirically pinned CUDA-WMMA/Vulkan-KHR
// internal semantics, probabilities and V operands are rounded to FP16, and
// selected keys are visited in 64-row blocks. This is the shared CUDA/Vulkan
// exact-mode rebaseline, not byte identity with the shipped fused-MMA kernel.
struct H3AttentionPlanDesc {
  uint32_t sequence = 0;
  uint32_t heads = 0;
  uint32_t head_dim = 0;
  float scale = 0.0f;
};

// Immutable device-resident ranges for one packed H3 sequence. There are four
// int32 values per global 128-query-row tile: two ordered half-open ranges.
// Endpoints are 64-row aligned and may extend to align_up(sequence,64); the
// kernel clamps those padded rows. Construction merges touching/overlapping
// ranges and rejects an empty selected set.
class H3AttentionRanges {
 public:
  H3AttentionRanges();
  ~H3AttentionRanges();
  H3AttentionRanges(H3AttentionRanges&&) noexcept;
  H3AttentionRanges& operator=(H3AttentionRanges&&) noexcept;
  H3AttentionRanges(const H3AttentionRanges&) = delete;
  H3AttentionRanges& operator=(const H3AttentionRanges&) = delete;
  static H3AttentionRanges create(TensorContext& context, uint32_t sequence,
                                  const int32_t* values, uint32_t value_count);
  uint32_t sequence() const;
  uint32_t query_tiles() const;
  uint64_t resident_bytes() const noexcept;
  uint64_t content_hash() const;
  bool belongs_to(const TensorContext& context) const noexcept;
  explicit operator bool() const noexcept;

 private:
  struct Impl;
  explicit H3AttentionRanges(std::shared_ptr<Impl> impl);
  std::shared_ptr<Impl> impl_;
  friend class H3AttentionPlan;
};

class H3AttentionPlan {
 public:
  H3AttentionPlan();
  ~H3AttentionPlan();
  H3AttentionPlan(H3AttentionPlan&&) noexcept;
  H3AttentionPlan& operator=(H3AttentionPlan&&) noexcept;
  H3AttentionPlan(const H3AttentionPlan&) = delete;
  H3AttentionPlan& operator=(const H3AttentionPlan&) = delete;
  static H3AttentionPlan create(TensorContext& context,
                                const H3AttentionPlanDesc& desc);
  const H3AttentionPlanDesc& description() const;
  // Tensors are distinct, nonoverlapping contiguous token-major BF16
  // [sequence,heads,head_dim] or its projection-native [sequence,heads*head_dim]
  // flattening. Null ranges select full attention; otherwise the range table
  // is indexed by the global query row, including for row-chunk records.
  // Exact mode requires finite Q/K/V, every BF16->FP16 V conversion to remain
  // finite, and finite scaled scores, online sums (finite-normal denominator),
  // PV FMA results/accumulators and final numerators. Both backends explicitly
  // canonicalize BF16-subnormal inputs, scaled QK results, online-correction
  // products, cooperative PV tile outputs, corrected accumulator sums, and
  // BF16-subnormal outputs to signed zero. Internal cooperative products obey
  // the pinned hardware tuple rather than a portable scalar ordering.
  void record(TensorBatch& batch, DeviceTensor& query, DeviceTensor& key,
              DeviceTensor& value, DeviceTensor& output,
              const H3AttentionRanges* ranges = nullptr,
              uint32_t query_row_offset = 0, uint32_t rows = 0,
              uint32_t output_row_offset = 0) const;
  explicit operator bool() const noexcept;

 private:
  struct Impl;
  explicit H3AttentionPlan(std::shared_ptr<Impl> impl);
  std::shared_ptr<Impl> impl_;
};

// Exact full-sequence causal grouped-query attention for Qwen text. This is a
// separate arithmetic contract from BlockedAttentionPlan: Q/K/V remain BF16,
// probabilities are rounded to BF16, and K/V have fewer heads than Q.
struct CausalGQAAttentionPlanDesc {
  uint32_t sequence = 0;
  uint32_t query_heads = 64;
  uint32_t kv_heads = 8;
  uint32_t head_dim = 128;
  float scale = 0.0f;
};

class CausalGQAAttentionPlan {
 public:
  CausalGQAAttentionPlan();
  ~CausalGQAAttentionPlan();
  CausalGQAAttentionPlan(CausalGQAAttentionPlan&&) noexcept;
  CausalGQAAttentionPlan& operator=(CausalGQAAttentionPlan&&) noexcept;
  CausalGQAAttentionPlan(const CausalGQAAttentionPlan&) = delete;
  CausalGQAAttentionPlan& operator=(const CausalGQAAttentionPlan&) = delete;

  static CausalGQAAttentionPlan create(
      TensorContext& context, const CausalGQAAttentionPlanDesc& desc);
  const CausalGQAAttentionPlanDesc& description() const;
  // Q is contiguous BF16 [sequence,query_heads,head_dim], K/V are
  // [sequence,kv_heads,head_dim], and output is Q-shaped and distinct. Global
  // query row q reads keys [0,q], including across row-range records. Exact
  // mode requires finite post-conversion inputs, finite scaled scores, and a
  // finite PV accumulator/final numerator at every step. A sufficient bound is
  // `causal_keys * max(abs(V)) <= max_finite_fp32`; this value-domain promise
  // belongs to the caller because record() cannot scan a device tensor.
  // BF16-subnormal inputs and FP32-subnormal products/accumulators are a shared
  // CUDA/Vulkan rebaseline to signed zero. Its sign is retained at the
  // canonicalization boundary; subsequent IEEE arithmetic may combine zeros.
  void record(TensorBatch& batch, DeviceTensor& query, DeviceTensor& key,
              DeviceTensor& value, DeviceTensor& output,
              uint32_t query_row_offset = 0, uint32_t rows = 0,
              uint32_t output_row_offset = 0) const;
  explicit operator bool() const noexcept;

 private:
  struct Impl;
  explicit CausalGQAAttentionPlan(std::shared_ptr<Impl> impl);
  std::shared_ptr<Impl> impl_;
};

}  // namespace vidfab::vulkan
