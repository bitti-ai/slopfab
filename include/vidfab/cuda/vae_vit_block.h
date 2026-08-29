#pragma once

#include <memory>

#include "vidfab/vae/vit_block.h"

namespace vidfab::cuda {

std::unique_ptr<vae::ExactViTBlockStage> create_exact_vae_vit_block_stage(
    const vae::ViTBlockConfig& config);

}  // namespace vidfab::cuda
