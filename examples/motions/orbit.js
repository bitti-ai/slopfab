import { motions } from "slopfab";

// A transparent, scalable Motion. Override timing/canvas on the CLI as needed.
export default {
  width: 1280,
  height: 720,
  fps: 30,
  duration: 4,
  frame({ width, height, time, progress }) {
    const e = motions.element;
    const size = Math.min(width, height);
    const radius = size * 0.23;
    const angle = time * Math.PI;
    const x = width / 2 + Math.cos(angle) * radius;
    const y = height / 2 + Math.sin(angle) * radius;
    const opacity = Math.min(1, progress * 8, (1 - progress) * 8);
    return motions.svg(width, height, [
      e(
        "defs",
        {},
        e(
          "linearGradient",
          { id: "accent", x1: "0%", y1: "0%", x2: "100%", y2: "100%" },
          [
            e("stop", { offset: "0%", "stop-color": "#d4ef85" }),
            e("stop", { offset: "100%", "stop-color": "#4dd9cf" }),
          ],
        ),
      ),
      e("g", { opacity }, [
        e("circle", {
          cx: width / 2,
          cy: height / 2,
          r: radius,
          fill: "none",
          stroke: "#d4ef85",
          "stroke-width": 2,
          opacity: 0.5,
        }),
        e("path", {
          d: `M ${width / 2 - radius / 2} ${height / 2} L ${width / 2} ${height / 2 - radius / 2} L ${width / 2 + radius / 2} ${height / 2} L ${width / 2} ${height / 2 + radius / 2} Z`,
          fill: "url(#accent)",
          transform: `rotate(${time * 30} ${width / 2} ${height / 2})`,
        }),
        e("circle", { cx: x, cy: y, r: size * 0.035, fill: "#ffffff" }),
      ]),
    ]);
  },
};
