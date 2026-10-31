"""CPU checks for the saved-latent long-run experiment's hand-off validation."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).parents[1]/'tools'))
from lcs_drift import write
from lcs_long_run import validate_extension, verify_first_correction


class LongRunTest(unittest.TestCase):
    def test_first_pair_checks_last_slice_and_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            meta=dict(width='64',height='32',frames='22')
            old=np.zeros((14,96),np.float32)
            plain=np.zeros((24,96),np.float32)
            fixed=plain.copy(); fixed[14::2,0]=.1
            audio=np.zeros((130,32),np.float32)
            write(root/'source',meta,dict(video_rows=old,audio_rows=audio[:74]))
            write(root/'baseline',dict(meta,frames='39'),dict(video_rows=plain,audio_rows=audio))
            def save():
                write(root/'corrected',dict(meta,frames='39'),dict(video_rows=fixed,audio_rows=audio))
            save()
            result=verify_first_correction(root/'source',root/'baseline',root/'corrected')
            self.assertEqual(result['new_latent_slices'],5)
            self.assertTrue(result['all_new_slices_changed'])
            fixed[-2,0]=0; save()
            with self.assertRaises(ValueError): verify_first_correction(root/'source',root/'baseline',root/'corrected')
            fixed[-2,0]=.1; audio[-1,0]=1; save()
            with self.assertRaises(ValueError): verify_first_correction(root/'source',root/'baseline',root/'corrected')

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
