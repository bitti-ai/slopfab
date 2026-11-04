# Whisper reference assets

The embedded tokenizer ranks, language ordering, mel filters and alignment-head
profiles originate from OpenAI Whisper, pinned to commit
`86098128c0b4f24f0e2aa2994de830614b474227`:
https://github.com/openai/whisper/tree/86098128c0b4f24f0e2aa2994de830614b474227

These assets are covered by the accompanying MIT LICENSE. Slopfab's native
inference implementation uses the published model architecture and decoding
algorithm. No Whisper/PyTorch Python code runs in the distributed application.
Model weights are supplied separately by the user.

`tools/import_whisper_assets.py` reproduces the checked-in resources, including
derived model profiles and flattened mel filters. It requires Python and NumPy
only when updating assets. Normal builds consume the checked-in resources and
perform no network requests.
