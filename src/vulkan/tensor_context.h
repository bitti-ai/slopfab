#pragma once
#include "tensor_state.h"

namespace slopfab::vulkan {
struct TensorContext::Impl {
  struct Parameters {
    uint32_t op = 0;
    uint32_t count = 0;
    uint32_t p[6] = {};
  };

  struct NormParameters {
    uint32_t rows = 0;
    uint32_t dim = 0;
    uint32_t epsilon_bits = 0;
    uint32_t mod_rows = 0;
  };

  struct VaeRopeParameters {
    uint32_t sequence = 0;
    uint32_t heads = 0;
    uint32_t head_dim = 64;
    uint32_t rope_dim = 48;
    uint32_t num_patches = 0;
    uint32_t epsilon_bits = 0;
    uint32_t unused[2] = {};
  };

  struct AudioParameters {
    uint32_t op = 0;
    uint32_t batch = 0;
    uint32_t in_channels = 0;
    uint32_t out_channels = 0;
    uint32_t length_in = 0;
    uint32_t length_out = 0;
    uint32_t kernel = 0;
    uint32_t padding_or_stride = 0;
    uint32_t dilation_or_padding = 0;
    uint32_t count = 0;
    uint32_t groups_x = 0;
    uint32_t scalar_bits = 0;
  };

  static_assert(sizeof(AudioParameters) == 48);

  struct KeyframeParameters {
    uint32_t in_channels = 0;
    uint32_t out_channels = 0;
    uint32_t input_height = 0;
    uint32_t input_width = 0;
    uint32_t output_height = 0;
    uint32_t output_width = 0;
    uint32_t kernel = 0;
    uint32_t stride = 0;
    uint32_t reflect_padding = 0;
    uint32_t asymmetric_padding = 0;
    uint32_t count = 0;
    uint32_t groups_x = 0;
  };

  static_assert(sizeof(KeyframeParameters) == 48);

  struct DitParameters {
    uint32_t op = 0;
    uint32_t rows = 0;
    uint32_t dim = 0;
    uint32_t mod_rows = 0;
    uint32_t num_t = 0;
    uint32_t num_modality = 0;
    uint32_t num_param = 0;
    uint32_t rank = 0;
    uint32_t count = 0;
    uint32_t groups_x = 0;
    uint32_t unused[2] = {};
  };

  static_assert(sizeof(DitParameters) == 48);

  struct WeightParameters {
    uint32_t op = 0;
    uint32_t count = 0;
    uint32_t out_features = 0;
    uint32_t in_features = 0;
    uint32_t block_size = 0;
    uint32_t nested_block_size = 0;
    uint32_t scalar_bits = 0;
    uint32_t group = 0;
  };

  struct GemmParameters {
    uint32_t rows = 0;
    uint32_t out_features = 0;
    uint32_t in_features = 0;
    uint32_t input_row_offset = 0;
    uint32_t output_row_offset = 0;
    uint32_t mode = 0;
    uint32_t unused[2] = {};
  };

  struct GemmPrepareParameters {
    uint32_t rows = 0;
    uint32_t in_features = 0;
    uint32_t input_row_offset = 0;
    uint32_t groups_x = 0;
  };

  struct AttentionParameters {
    uint32_t sequence = 0;
    uint32_t heads = 0;
    uint32_t head_dim = 0;
    uint32_t scale_bits = 0;
    uint32_t query_row_offset = 0;
    uint32_t output_row_offset = 0;
    uint32_t rows = 0;
    uint32_t reserved = 0;
  };

  struct CausalGQAAttentionParameters {
    uint32_t sequence = 0;
    uint32_t query_heads = 0;
    uint32_t kv_heads = 0;
    uint32_t head_dim = 0;
    uint32_t scale_bits = 0;
    uint32_t query_row_offset = 0;
    uint32_t output_row_offset = 0;
    uint32_t rows = 0;
  };

  uint32_t max_batch_operators = 0;
  ComputeContext commands;
  BufferPool pool;
  TensorWorkspace scratch;
  ComputePipeline ops_pipeline;
  ComputePipeline rope_pipeline;
  ComputePipeline vae_rope_pipeline;
  ComputePipeline weight_pipeline;
  ComputePipeline gemm_pipeline;
  ComputePipeline gemm_coop_pipeline;
  ComputePipeline gemm_prepare_pipeline;
  ComputePipeline gemm_coop_f16_pipeline;
  ComputePipeline attention_blocked_pipeline;
  ComputePipeline attention_h3_pipeline;
  ComputePipeline attention_h3_banded_pipeline;
  ComputePipeline attention_flash_pipeline, attention_flash_banded_pipeline;
  ComputePipeline attention_vsa_pipeline, attention_vsa_prepare_pipeline;
  std::array<std::array<ComputePipeline, 4>, 4> attention_sage_pipelines,
      attention_sage_banded_pipelines;
  ComputePipeline attention_sage_prepare_pipeline;
  DeviceInfo sage_device_info;
  uint64_t sage_extra_workspace_bytes = 0;
  ComputePipeline attention_prepare_pipeline;
  ComputePipeline attention_causal_gqa_pipeline;
  ComputePipeline rms_norm_pipeline;
  ComputePipeline layer_norm_pipeline;
  ComputePipeline bf16_rms_block_pipeline;
  ComputePipeline bf16_rms_narrow_pipeline;
  ComputePipeline bf16_layer_pipeline;
  ComputePipeline bf16_mod_pipeline;
  ComputePipeline fp32_mod_pipeline;
  ComputePipeline group_norm_pipeline;
  ComputePipeline vae_residual_pipeline;
  ComputePipeline vae_swiglu_pipeline;
  ComputePipeline vae_denorm_pipeline;
  ComputePipeline audio_pipeline;
  ComputePipeline keyframe_pipeline;
  ComputePipeline reference_pipeline;
  ComputePipeline dit_pipeline;
  Buffer upload_buffer;
  Buffer readback_buffer;
  uint64_t staging_capacity = 0;
  std::vector<StorageBinding> ops_bindings;
  std::vector<StorageBinding> vae_pointwise_bindings;
  std::vector<StorageBinding> norm_bindings;
  std::vector<StorageBinding> mod_bindings;
  std::vector<StorageBinding> vae_rope_bindings;
  std::vector<StorageBinding> audio_bindings;
  std::vector<StorageBinding> keyframe_bindings;
  std::vector<StorageBinding> dit_bindings;
  std::vector<StorageBinding> weight_bindings;
  std::vector<StorageBinding> gemm_bindings;
  std::vector<StorageBinding> gemm_prepare_bindings;
  std::vector<StorageBinding> attention_bindings;
  std::vector<StorageBinding> attention_h3_bindings;
  std::vector<StorageBinding> attention_h3_banded_bindings;
  std::vector<StorageBinding> attention_sage_bindings;
  std::vector<StorageBinding> attention_sage_prepare_bindings;
  std::vector<StorageBinding> attention_prepare_bindings;
  std::vector<StorageBinding> attention_causal_gqa_bindings;
  bool full_arithmetic_exact = false;
  bool exact_vae_norm = false;
  bool exact_vae_pointwise = false;
  bool exact_audio = false;
  bool exact_dit_pointwise = false;
  bool exact_attention = false;
  bool exact_h3_attention = false;
  bool flash_attention = false;
  bool sage_attention = false;
  bool exact_causal_gqa_attention = false;
  bool cooperative_gemm = false;
  bool cooperative_f16_gemm = false;
  uint32_t max_dispatch_x = 0;
  uint32_t max_dispatch_y = 0;
  uint64_t max_storage_bytes = 0;
  uint64_t storage_binding_alignment = 1;
  std::atomic<bool> recorder_active{false};

  explicit Impl(const Device& input, const TensorContextOptions& tensor_options);
  void prepare_pipelines(const Device& input, TensorPipelineSet sets);
  uint32_t prepared_sets = 0;
  void* device_identity = nullptr;

  void require_pipeline_set(TensorPipelineSet set) const {
    if ((prepared_sets & static_cast<uint32_t>(set)) == 0)
      throw std::logic_error(
          "vulkan tensor: prepare the required pipeline set before creating a plan");
  }

  uintptr_t context_id = next_context_identity();

  struct RecorderLease {
    Impl* owner = nullptr;
    RecorderLease() = default;

    explicit RecorderLease(Impl* value) : owner(value) {
    }

    RecorderLease(RecorderLease&& other) noexcept : owner(std::exchange(other.owner, nullptr)) {
    }

    RecorderLease& operator=(RecorderLease&& other) noexcept {
      if (this != &other) {
        release();
        owner = std::exchange(other.owner, nullptr);
      }
      return *this;
    }

    RecorderLease(const RecorderLease&) = delete;
    RecorderLease& operator=(const RecorderLease&) = delete;

    ~RecorderLease() {
      release();
    }

    void release() noexcept {
      if (!owner)
        return;
      owner->recorder_active.store(false, std::memory_order_release);
      owner = nullptr;
    }
  };

  RecorderLease acquire_recorder() {
    bool expected = false;
    if (!recorder_active.compare_exchange_strong(expected, true, std::memory_order_acquire,
                                                 std::memory_order_relaxed)) {
      throw std::logic_error("vulkan tensor: another batch or boundary operation is active");
    }
    return RecorderLease(this);
  }

  void ensure_staging(uint64_t bytes) {
    if (bytes <= staging_capacity)
      return;
    Buffer new_upload = pool.allocate(bytes, BufferUsage::kTransferSource, MemoryUsage::kUpload);
    Buffer new_readback =
        pool.allocate(bytes, BufferUsage::kTransferDestination, MemoryUsage::kReadback);
    upload_buffer = std::move(new_upload);
    readback_buffer = std::move(new_readback);
    staging_capacity = bytes;
    pool.trim();
  }

  std::shared_ptr<DeviceTensor::Impl> require(DeviceTensor& tensor) const {
    if (!tensor.impl_ || tensor.impl_->context != context_id) {
      throw std::invalid_argument("vulkan tensor: tensor belongs to another context");
    }
    return tensor.impl_;
  }

  void complete(CommandList&& list) {
    Submission done = commands.submit(std::move(list));
    done.wait();
    commands.collect();
  }

  uint32_t validate_dispatch(uint64_t count) const {
    if (count == 0 || count > std::numeric_limits<uint32_t>::max()) {
      throw std::out_of_range("vulkan tensor: operation exceeds uint32 indexing limits");
    }
    const uint64_t groups = (count + 63ull) / 64ull;
    if (groups > max_dispatch_x) {
      throw std::out_of_range("vulkan tensor: operation exceeds dispatch limits");
    }
    return static_cast<uint32_t>(count);
  }
};

} // namespace slopfab::vulkan
