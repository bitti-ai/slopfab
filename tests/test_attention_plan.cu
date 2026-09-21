#include "detail/nn_kernels_fixture.h"

SLOPFAB_TEST(attention_plan_compact_gqa_workspace_and_execution) {
  using namespace slopfab::cuda;
  AttentionConfig config;
  config.seq_len = 37;
  config.num_heads = 8;
  config.head_dim = 64;
  config.query_block = 11;
  config.key_block = 13;
  const int kv_heads = 2;
  const AttentionConfig original = config;
  const auto plan = AttentionPlan::compile(config, kv_heads, AttentionBackend::kBlocked);
  CHECK(plan.description().key_value_heads == kv_heads);
  CHECK(plan.workspace_bytes() < attention_workspace_bytes(config, AttentionBackend::kBlocked));
  config.seq_len = 9000; // the compiled plan must keep its original shape
  const auto q = bf16_round(make_data(size_t(37) * 8 * 64, 471u, 0.3f));
  const auto k = bf16_round(make_data(size_t(37) * 2 * 64, 472u, 0.3f));
  const auto v = bf16_round(make_data(size_t(37) * 2 * 64, 473u, 0.3f));
  std::vector<float> full_k(q.size()), full_v(q.size());
  for (int row = 0; row < 37; ++row)
    for (int head = 0; head < 8; ++head)
      for (int dim = 0; dim < 64; ++dim) {
        const size_t target = (size_t(row) * 8 + head) * 64 + dim;
        const size_t source = (size_t(row) * 2 + head / 4) * 64 + dim;
        full_k[target] = k[source];
        full_v[target] = v[source];
      }
  BfBuf dq(q), dk(k), dv(v), dfk(full_k), dfv(full_v), output(q.size()), reference(q.size());
  CublasScope handle;
  Workspace workspace;
  workspace.reserve(plan.workspace_bytes());
  plan.forward(handle.h, nullptr, dq.p(), dk.p(), dv.p(), output.p(), workspace);
  CHECK(workspace.used() == 0);
  Workspace reference_workspace;
  reference_workspace.reserve(attention_workspace_bytes(original, AttentionBackend::kBlocked));
  attention_forward(handle.h, nullptr, dq.p(), dfk.p(), dfv.p(), reference.p(), original,
                    AttentionBackend::kBlocked, reference_workspace);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  CHECK_CLOSE(reference.host(), output.host(), 0.0, "compiled compact GQA");
  Workspace empty;
  bool rejected = false;
  try {
    plan.forward(handle.h, nullptr, dq.p(), dk.p(), dv.p(), output.p(), empty);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  CHECK(rejected);
}

SLOPFAB_TEST(attention_plan_rejects_unimplemented_contracts) {
  using namespace slopfab;
  using namespace slopfab::cuda;
  AttentionDescriptor descriptor{37, 37, 8, 2, 64};
  AttentionExecutionPolicy policy;
  descriptor.mask = AttentionMask::kCausal;
  bool rejected = false;
  try {
    (void)AttentionPlan::compile(descriptor, policy);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  CHECK(rejected);
  descriptor.mask = AttentionMask::kFull;
  descriptor.arithmetic = AttentionArithmetic::kExact;
  descriptor.scale = exact_attention_scale(64);
  rejected = false;
  try {
    (void)AttentionPlan::compile(descriptor, policy);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  CHECK(rejected);
  descriptor.arithmetic = AttentionArithmetic::kEquivalent;
  const auto plan = AttentionPlan::compile(descriptor, policy);
  CHECK(plan.workspace_bytes() == 0);
}
