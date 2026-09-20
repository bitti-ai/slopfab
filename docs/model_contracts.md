# Model and adapter contracts

The H3 loader resolves structural capabilities separately from the legacy release enum. Metadata is authoritative when present: renaming an explicitly described checkpoint cannot select Animate behavior or a FastH3 sampling recipe. Missing metadata preserves legacy filename/source inference, reported by `ModelDescriptor::origin`.

`slopfab.model` is a SafeTensors metadata string containing JSON:

```json
{
  "version": 1,
  "family": "h3",
  "modulation": "table",
  "supports_references": true,
  "compressed_attention": false,
  "qkv_layout": "contiguous",
  "legacy_profile": "none",
  "adaln_grid_id": "my-training-grid-v1"
}
```

The first six fields are required. `legacy_profile` and `adaln_grid_id` are optional. Supported modulation implementations are `table` and `timestep_mlp`; supported QKV layouts are `contiguous` and `interleaved`. Unknown keys, versions, families and inconsistent tensor contracts fail before upload. Table modulation requires rank 8 and 1025 rows throughout lookup, adapter conversion and capture. Changing only configuration dimensions is rejected. A legacy `qkv_layout` metadata entry must agree with the descriptor.

`legacy_profile: "animate"` enables compatibility defaults for reference-capable table models without compressed attention. `"fast_h3_v2"` requires compressed-attention gates and table modulation without references. These labels are optional compatibility recipes. Compressed attention itself is selected from `compressed_attention`, independently of a branded release. The full timestep-MLP implementation currently supports the existing reference-conditioned graph only. New families still require implementation and backend validation.

`resolve_model_descriptor` returns structural facts, detected quantization, compatibility enum and metadata origin. `detect_transformer_architecture` remains available for existing callers. Header inspection validates the descriptor's discriminating tensors; full loading still validates every required weight, shape and quantization record.

## Conditioner and tokenizer contracts

`slopfab.conditioner` may describe the existing Qwen conditioner explicitly:

```json
{
  "version": 1,
  "family": "qwen3_vl",
  "tokenizer": "qwen_byte_bpe",
  "output_width": 5120,
  "output_layer": 49,
  "final_normalization": false,
  "vision": true
}
```

All fields are required when the metadata entry is present. `output_layer` is zero-based. Width and selected final layer must match `EncoderConfig`; the implemented output is an unnormalized residual stream. Vision requires the complete 351-tensor Qwen tower. `ConditionerDescriptor::fingerprint()` is canonical and independent of JSON ordering. Missing metadata preserves existing configured semantics. Numerical/backend shape validation remains separate.

Tokenizer loading validates explicitly declared BPE model type, deterministic merge settings, the Qwen split regex, and ByteLevel preprocessing/decoding. Unsupported Unigram/WordPiece models, alternative pre-tokenizers, dropout and incompatible prefix behavior fail instead of silently using Qwen tokenization. Historical minimal vocabulary fixtures may omit these declarations. This validation preserves the existing tokenizer's Unicode category and normalization behavior; it does not claim a new general-purpose tokenizer implementation.

## Adapter targets and assets

Generic factor parsing, strength/alpha validation and composition remain in `src/core/lora.cpp`. H3 target aliases, supported projections and affine AdaLN fitting live in `src/dit/h3_lora.cpp`. The shared NF4 reader now also carries source dtype; transformer loading explicitly requires BF16-source NF4 as before.

Inference is read-only: `LoraAdapters::load` reads an embedded grid or a local companion and never downloads an asset or rewrites an adapter. The historical companion filename `h3_silu_temb_grid.safetensors` remains supported. An adapter can declare a different local asset with `slopfab.lora_grid` JSON:

```json
{
  "version": 1,
  "identity": "my-training-grid-v1",
  "file": "training-grid.safetensors",
  "tensor": "activation_curve",
  "rows": 1025,
  "width": 2688
}
```

Only `version` is required; other fields override the legacy companion defaults or assert compatibility. The file is relative to the adapter directory and may not escape it. Grid shape and dtype are checked. When the base model declares `adaln_grid_id`, the adapter must declare the same identity. Affine fitting still enforces its numerical residual bound. Custom assets never trigger the legacy downloader.

Asset preparation is an explicit operation, separate from inference:

```cpp
#include "slopfab/lora.h"
slopfab::prepare_lora_grid("adapter.safetensors", 2688); // local companion only
slopfab::prepare_lora_grid("adapter.safetensors", 2688, true); // opt into pinned legacy download
```

Preparation embeds the validated grid using atomic replacement and checks the adapter identity before writing. It needs write access and temporary space for a complete copy. The optional downloader remains Windows-only and verifies the pinned asset size and SHA-256. Existing embedded adapters and local companion inference need no write access.

## Transformer modules

The former 3200-line transformer is divided into lifecycle/forward orchestration (`transformer.cpp`), checkpoint planning/loading (`transformer_load.cpp`), sequence/text preparation (`transformer_prepare.cpp`), block arithmetic (`transformer_execution.cpp`), and captures/diagnostics (`transformer_capture.cpp`). Shared state and resource helpers are private headers. Public ABI is unchanged. General CUDA attention uses compiled attention plans for dispatch and scratch sizing; exact, VSA and compact query execution retain their dedicated numerical paths.

## Geometry and archive identity

`LatentGeometry` centralizes the H3 frame rate, spatial compression, patch and channel dimensions, audio rates, temporal codec mapping, canvas bounds and position-grid parameters. `slopfab.geometry` is version-1 JSON; supplied fields override canonical H3 defaults. Unknown fields, fractional dimensions, impossible patch/canvas alignment and invalid position patterns are rejected. `geometry_json`/`fingerprint` provide a canonical complete representation independent of metadata ordering.

Geometry-aware host packing overloads accept a descriptor, including non-default patch sizes, latent channel counts and audio dimensions. For example, a synthetic contract can use three video channels and 1-by-2 patches without another packing implementation. Temporal mapping remains the declared chunk-plus-offset algorithm; a genuinely different algorithm requires another family implementation. H3 floating-point position and rounding operation order is preserved.

The current H3 execution and continuation paths call `require_h3_latent_geometry` and reject incompatible structural descriptors before upload. Trained pixel budget and aspect bounds are canvas policy, so compatible profiles can override them; the shared planner consumes these values and reference caches include geometry when preprocessing uses the target canvas. Host generality does not imply support in unchanged GPU kernels or codecs. New latent archives write their canonical geometry under `slopfab.geometry`, while archives without that entry retain the original H3 geometry. Loading/continuation validates the identity and cannot silently combine different latent contracts.
