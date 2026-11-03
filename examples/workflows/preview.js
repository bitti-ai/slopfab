import sf from 'slopfab';

export default ([input, output, audio]) => {
  if (!input || !output) throw Error('Usage: slopfab run preview.js -- INPUT.y4m OUTPUT.mp4 [AUDIO.wav]');
  const args = [sf.tool('ffmpeg'), '-nostdin', '-n', '-i', input];
  if (audio) args.push('-i', audio, '-map', '0:v:0', '-map', '1:a:0', '-c:a', 'aac');
  args.push('-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', output);
  sf.exec(args, {timeoutMs: 600000});
  return {path: sf.path.resolve(output), url: sf.path.fileUrl(output)};
};
