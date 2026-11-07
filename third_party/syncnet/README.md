# SyncNet architecture reference

The native implementation in `src/syncnet` follows the MIT-licensed SyncNet model,
S3FD network and evaluation pipeline from https://github.com/joonson/syncnet_python.
The upstream copyright/license is preserved in LICENSE. No upstream Python is
shipped or executed. Model checkpoints are downloaded separately.

Reference SyncNetModel.py SHA256:
`41b85aed0006546cb2d6dc9180c5bc3dac737ab80eff643022e397a349c30c22`.
Exact checkpoint hashes, download revisions and supported formats are documented
in `docs/syncnet.md`. Compiled tensor tables are reproduced by
`tools/syncnet_manifest.py`, which rejects any other checkpoint bytes before
using a restricted, inert parser.

MFCC follows the mathematical defaults of python_speech_features (preemphasis,
rectangular windows, power spectrum, triangular mel filters, orthonormal DCT,
cepstral lifter and log energy). The implementation uses its own radix-2 FFT.
