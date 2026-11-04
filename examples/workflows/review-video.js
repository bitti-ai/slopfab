import {frames} from 'slopfab';

// slopfab run examples/workflows/review-video.js -- output.y4m review.png
export default ([input, output = 'review.png']) => {
  if (!input) throw new Error('Expected a video path');
  return frames.extract({input, output, columns: 4, rows: 3, width: 256});
};
