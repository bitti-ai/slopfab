#include "harness.h"
#include "slopfab/attention_descriptor.h"
#include "slopfab/text/backend_capabilities.h"
#include <limits>

namespace {
template <class F> bool rejects(F f) {
  try {
    f();
  } catch (const std::exception&) {
    return true;
  }
  return false;
}
}

SLOPFAB_TEST(attention_descriptor_validates_grouped_query_contract) {
  slopfab::AttentionDescriptor d{17, 31, 12, 3, 64};
  slopfab::validate_attention_descriptor(d);
  d.key_value_heads = 5;
  CHECK(rejects([&] {
    slopfab::validate_attention_descriptor(d);
  }));
  d.key_value_heads = 3;
  d.scale = std::numeric_limits<float>::quiet_NaN();
  CHECK(rejects([&] {
    slopfab::validate_attention_descriptor(d);
  }));
  d.scale = 0;
  d.query_heads = UINT32_MAX;
  CHECK(rejects([&] {
    slopfab::validate_attention_descriptor(d);
  }));
}

SLOPFAB_TEST(attention_descriptor_exact_scale_and_text_capability) {
  using namespace slopfab;
  AttentionDescriptor d{23,
                        23,
                        64,
                        8,
                        128,
                        AttentionLayout::kTokensHeadsChannels,
                        AttentionMask::kCausal,
                        AttentionArithmetic::kExact,
                        exact_attention_scale(128)};
  validate_attention_descriptor(d);
  CHECK(supports_exact_text_attention(d, 4096));
  d.key_value_heads = 4;
  validate_attention_descriptor(d); // mathematically valid, unimplemented exact kernel
  CHECK(!supports_exact_text_attention(d, 4096));
  d.key_value_heads = 8;
  d.key_value_tokens = 22;
  CHECK(!supports_exact_text_attention(d, 4096));
  d.scale = 0;
  CHECK(rejects([&] {
    validate_attention_descriptor(d);
  }));
  CHECK(text::supports_exact_text_layer({23, 5120, 64, 8, 128, 25600, 1e-6f}));
  CHECK(!text::supports_exact_text_layer({23, 4096, 64, 8, 128, 25600, 1e-6f}));
}
