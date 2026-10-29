"""Render a small GIF (boxes + IDs, heads blurred) from tracker output."""
import argparse
import cv2, imageio.v2 as imageio
from speed import DATA, RES, load_tracks, blur_heads

ap = argparse.ArgumentParser()
ap.add_argument("--seq", default="MOT17-09")
ap.add_argument("--tracker", default="yolo11s_bytetrack_pt")
ap.add_argument("--start", type=int, default=100)
ap.add_argument("--end", type=int, default=250)
ap.add_argument("--stride", type=int, default=3)
ap.add_argument("--width", type=int, default=480)
a = ap.parse_args()

d = load_tracks(RES / "tracks" / a.tracker / f"{a.seq}.txt")
frames = []
for f in range(a.start, a.end, a.stride):
    img = cv2.imread(str(DATA / a.seq / "img1" / f"{f:06d}.jpg"))
    g = d[d.frame == f]
    img = blur_heads(img, g[["x", "y", "w", "h"]].values)
    s = a.width / img.shape[1]
    img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    for r in g.itertuples():
        p1, p2 = (int(r.x * s), int(r.y * s)), (int((r.x + r.w) * s), int((r.y + r.h) * s))
        col = ((r.id * 67) % 255, (r.id * 131) % 255, (r.id * 29) % 255)
        cv2.rectangle(img, p1, p2, col, 1)
        cv2.putText(img, str(r.id), (p1[0], max(p1[1] - 2, 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1, cv2.LINE_AA)
    frames.append(img[:, :, ::-1])
imageio.mimsave("docs/demo.gif", frames, duration=a.stride / 30 * 1000, loop=0)
