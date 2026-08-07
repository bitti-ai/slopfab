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
  CHECK(p.layout.total_rows()==32); CHECK(p.indices.video.size()==20); CHECK(p.indices.audio.size()==10);
  CHECK(p.indices.video[0]==2); CHECK(p.indices.audio[0]==6); CHECK(p.indices.video[4]==12);
  CHECK(p.indices.audio[6]==20); CHECK(p.indices.video[12]==24);
  CHECK_NEAR(p.position_ids[2*3],2.0,0); CHECK_NEAR(p.position_ids[6*3],3.0,0);
  CHECK_NEAR(p.position_ids[12*3],3.0,0); CHECK_NEAR(p.position_ids[20*3],11.333333333333334,1e-12);
}
