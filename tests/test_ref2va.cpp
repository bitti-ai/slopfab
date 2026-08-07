#include "harness.h"
#include "vidfab/dit/ref2va.h"
using namespace vidfab::dit;
VIDFAB_TEST(ref2va_image_size) {
  int h=0,w=0; resolve_reference_image_size(100,50,&h,&w);
  CHECK(h==2048); CHECK(w==4096);
  CHECK(::vidfab::test::throws([]{int a,b;resolve_reference_image_size(5,1,&a,&b);}));
}
VIDFAB_TEST(ref2va_order_and_clock) {
  ReferenceGeometry image{ReferenceKind::kImage,1,4,4,0};
  ReferenceGeometry clip{ReferenceKind::kVideo,2,4,4,3};
  auto p=build_ref2va_packed_sequence({kTagText,kTagVideo},{image,clip},2,4,4,2);
  CHECK(p.layout.num_condition_video==12); CHECK(p.layout.num_condition_audio==6);
  CHECK(p.layout.condition_audio_is_explicit);
  CHECK(p.layout.total_rows()==32); CHECK(p.indices.video.size()==20); CHECK(p.indices.audio.size()==10);
  CHECK(p.indices.video[0]==2); CHECK(p.indices.audio[0]==6); CHECK(p.indices.video[4]==12);
  CHECK(p.indices.audio[6]==20); CHECK(p.indices.video[12]==24);
  CHECK_NEAR(p.position_ids[2*3],2.0,0); CHECK_NEAR(p.position_ids[6*3],3.0,0);
  CHECK_NEAR(p.position_ids[12*3],3.0,0); CHECK_NEAR(p.position_ids[20*3],11.333333333333334,1e-12);
}

VIDFAB_TEST(ref2va_interleaved_reference_timesteps) {
  ReferenceGeometry image{ReferenceKind::kImage,1,4,4,0};
  ReferenceGeometry sound{ReferenceKind::kAudio,1,0,0,2};
  ReferenceGeometry clip{ReferenceKind::kVideo,1,4,4,1};
  auto p=build_ref2va_packed_sequence({kTagText},{image,sound,clip},1,4,4,2);
  auto rt=build_row_timesteps(p.layout,p.indices,0.75f,0.25f);
  // Indices are condition-first within each modality despite packed interleaving.
  for(int i=0;i<p.layout.num_condition_audio;++i)
    CHECK(rt.indices[static_cast<size_t>(p.indices.audio[i])]==1);
  for(size_t i=static_cast<size_t>(p.layout.num_condition_audio);i<p.indices.audio.size();++i)
    CHECK(rt.indices[static_cast<size_t>(p.indices.audio[i])]==0);
  for(int i:p.indices.video) CHECK(rt.indices[static_cast<size_t>(i)]==1);

  // Image-only references have no audio anchors: every audio row is generated.
  auto image_only=build_ref2va_packed_sequence({kTagText},{image},1,4,4,2);
  auto image_rt=build_row_timesteps(image_only.layout,image_only.indices,0.75f,0.25f);
  for(int i:image_only.indices.audio) CHECK(image_rt.indices[static_cast<size_t>(i)]==0);
}
