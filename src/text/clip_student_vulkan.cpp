#include "../krea/vulkan_runtime.h"
#define CLIP_STUDENT_FUNCTION encode_clip_student_vulkan
#define CLIP_RUNTIME krea_vulkan::Runtime
#define CLIP_FROM_HOST krea_vulkan::from_host
#define CLIP_TO_HOST krea_vulkan::to_host
#define CLIP_ROPE krea_vulkan::ops::rope
#define CLIP_RESIDUAL krea_vulkan::ops::residual
#define CLIP_ACTIVATE krea_vulkan::ops::activate
#include "clip_student.inc"
