"""CPU checks for the saved-latent long-run experiment's hand-off validation."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).parents[1]/'tools'))
from lcs_drift import write
from lcs_long_run import validate_extension


class LongRunTest(unittest.TestCase):
    def test_rejects_changed_prefix_missing_frames_and_nonfinite_latents(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            meta=dict(width='64',height='32',fps='24',frames='22',video_vae='vae',audio_vae='audio',transformer='dit')
            video=np.arange(14*96,dtype=np.float32).reshape(14,96)
            audio=np.arange(74*32,dtype=np.float32).reshape(74,32)
            write(root/'source',meta,dict(video_rows=video,audio_rows=audio))
            joined_video=np.concatenate([video,np.zeros((10,96),np.float32)])
            joined_audio=np.zeros((2,65,32),np.float32)
            joined_audio[:,:37]=audio.reshape(2,37,32)
            extended=dict(meta,frames='39')
            def save():
                write(root/'next',extended,dict(video_rows=joined_video,audio_rows=joined_audio.reshape(-1,32)))
            save()
            self.assertEqual(validate_extension(root/'source',root/'next',17),39)
            joined_video[0,0]+=1; save()
            with self.assertRaises(ValueError): validate_extension(root/'source',root/'next',17)
            joined_video[0,0]-=1
            joined_audio[1,0,0]+=1; save()
            with self.assertRaises(ValueError): validate_extension(root/'source',root/'next',17)
            joined_audio[1,0,0]-=1
            joined_video[-1,0]=np.nan; save()
            with self.assertRaises(ValueError): validate_extension(root/'source',root/'next',17)
            joined_video[-1,0]=0; extended['frames']='38'; save()
            with self.assertRaises(ValueError): validate_extension(root/'source',root/'next',17)


if __name__=='__main__':
    unittest.main()
