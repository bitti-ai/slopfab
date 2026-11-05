# Demucs architecture reference

Slopfab's native HTDemucs implementation follows Meta's MIT-licensed Demucs
4.0.1 architecture and inference algorithms:
https://github.com/facebookresearch/demucs

Reference modules: `htdemucs.py`, `hdemucs.py`, `demucs.py`, `transformer.py`,
`spec.py`, `apply.py` and `separate.py`. The supported released model is
`htdemucs`, signature `955717e8`, with four sources and a 7.8-second segment.

The accompanying LICENSE retains the upstream attribution for this native
adaptation. No Python source or weights are embedded in the executable.
`tools/demucs_reference.py` performs explicit developer-side conversion and
numerical comparisons against the upstream implementation.
