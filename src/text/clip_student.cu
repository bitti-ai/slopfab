#include "../fibo/runtime.cuh"
#define CLIP_STUDENT_FUNCTION encode_clip_student_cuda
#define CLIP_RUNTIME moss::Runtime
#define CLIP_FROM_HOST moss::from_host
#define CLIP_TO_HOST moss::to_host
#define CLIP_ROPE fibo::rope
#define CLIP_RESIDUAL moss::residual
#define CLIP_ACTIVATE moss::activate
#include "clip_student.inc"
