import sf, { motions } from "slopfab";

// slopfab motions render lower-third.js --out title.mov -- FONT.ttf "Your title"
export default ([fontPath, title = "A moment in motion"]) => {
  if (!fontPath)
    throw Error(
      "Pass a TTF/OTF font path after --, followed by optional title text.",
    );
  motions.font("MotionTitle", sf.path.resolve(fontPath));
  return {
    width: 1280,
    height: 720,
    fps: 30,
    duration: 5,
    frame({ width, height, time, duration }) {
      const enter = 1 - Math.pow(1 - Math.min(time / 0.6, 1), 3);
      const leave = Math.min((duration - time) / 0.5, 1);
      const e = motions.element;
      return motions.svg(
        width,
        height,
        e(
          "g",
          {
            transform: `translate(${width * 0.05 - (1 - enter) * 80} ${height * 0.75})`,
            opacity: enter * leave,
          },
          [
            e("rect", {
              width: width * 0.72,
              height: height * 0.16,
              rx: 4,
              fill: "#111516",
              "fill-opacity": 0.9,
            }),
            e("rect", { width: 5, height: height * 0.16, fill: "#d4ef85" }),
            motions.text(title, {
              x: 24,
              y: height * 0.1,
              "font-family": "MotionTitle",
              "font-size": height * 0.055,
              fill: "#ffffff",
            }),
          ],
        ),
      );
    },
  };
};
