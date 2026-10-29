"""Pixel-space speed & trajectory stats from tracker output (default: yolo11s ByteTrack, static-camera sequences).

Foot point = bottom-centre of the box. Speed = displacement of the (moving-average smoothed) foot point per second.
Pixel speeds are NOT metric: perspective makes the same real speed look slower far from the camera.
`pixels_to_ground` shows how to get m/s once 4+ ground-plane reference points are known (not applied to MOT17 here,
because MOT17 ships no ground-plane calibration).
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, cv2

ROOT = Path(__file__).parent
RES = ROOT / "results"
DATA = ROOT / "data/MOT17/train"
FPS = {"MOT17-02": 30, "MOT17-04": 30, "MOT17-09": 30}  # read from seqinfo.ini below; static-camera seqs
MIN_FRAMES = 30  # ignore fragments shorter than 1 s


def load_tracks(txt):
    d = pd.read_csv(txt, header=None, names="frame id x y w h conf a b c".split())
    d["fx"], d["fy"] = d.x + d.w / 2, d.y + d.h
    return d


def smooth(a, k=9):
    pad = np.pad(a, (k // 2, k // 2), mode="edge")
    return np.convolve(pad, np.ones(k) / k, mode="valid")


def track_stats(d, fps):
    rows = []
    for tid, g in d.groupby("id"):
        g = g.sort_values("frame")
        if len(g) < MIN_FRAMES:
            continue
        fx, fy = smooth(g.fx.values), smooth(g.fy.values)
        step = np.hypot(np.diff(fx), np.diff(fy)) / np.maximum(np.diff(g.frame.values), 1)  # px / frame
        rows.append(dict(id=int(tid), frames=len(g), dist_px=float(step.sum()), mean_px_s=float(step.mean() * fps),
                         max_px_s=float(step.max() * fps)))
    return pd.DataFrame(rows)


def pixels_to_ground(pts_px, img_ref, ground_ref):
    """Homography from >=4 image points (px) to their known ground-plane coords (metres) -> metres for pts_px."""
    H, _ = cv2.findHomography(np.float32(img_ref), np.float32(ground_ref))
    p = cv2.perspectiveTransform(np.float32(pts_px).reshape(-1, 1, 2), H)
    return p.reshape(-1, 2)


def fps_of(seq):
    for l in (DATA / seq / "seqinfo.ini").read_text().splitlines():
        if l.startswith("frameRate"):
            return int(l.split("=")[1])


def blur_heads(img, boxes, frac=0.2):
    """Blur the top `frac` of every person box (privacy: faces)."""
    for x, y, w, h in boxes:
        x0, y0, x1, y1 = max(int(x), 0), max(int(y), 0), int(x + w), int(y + h * frac)
        if x1 > x0 and y1 > y0:
            img[y0:y1, x0:x1] = cv2.GaussianBlur(img[y0:y1, x0:x1], (0, 0), max((x1 - x0) / 3, 3))
    return img


def figure(seq, d, stats, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f0 = d.frame.min()
    img = cv2.imread(str(DATA / seq / "img1" / f"{f0:06d}.jpg"))
    img = blur_heads(img, d[d.frame == f0][["x", "y", "w", "h"]].values)
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [2, 1]})
    ax[0].imshow(img[:, :, ::-1])
    keep = set(stats.id)
    cmap = plt.get_cmap("plasma")
    vmax = stats.mean_px_s.max()
    for tid, g in d[d.id.isin(keep)].groupby("id"):
        v = stats.set_index("id").mean_px_s[tid] / vmax
        ax[0].plot(g.fx, g.fy, color=cmap(v), lw=1.2)
    ax[0].set_title(f"{seq}: foot-point trajectories (colour = mean px/s, max {vmax:.0f})"); ax[0].axis("off")
    ax[1].hist(stats.mean_px_s, bins=25, color="#444")
    ax[1].set_xlabel("mean speed per track (px/s)"); ax[1].set_ylabel("tracks"); ax[1].set_title("speed distribution")
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracker", default="yolo11s_bytetrack_pt")
    ap.add_argument("--seqs", nargs="+", default=list(FPS))
    a = ap.parse_args()
    out = {}
    for s in a.seqs:
        d = load_tracks(RES / "tracks" / a.tracker / f"{s}.txt")
        st = track_stats(d, fps_of(s))
        st.to_csv(RES / f"speed_{s}.csv", index=False)
        out[s] = dict(tracks_ge_1s=len(st), median_mean_px_s=round(st.mean_px_s.median(), 1),
                      p90_mean_px_s=round(st.mean_px_s.quantile(0.9), 1), max_px_s=round(st.max_px_s.max(), 1))
        print(s, out[s])
        if s == "MOT17-09":
            (ROOT / "docs").mkdir(exist_ok=True)
            figure(s, d, st, ROOT / "docs/speed_trajectories.png")
    (RES / "speed_summary.json").write_text(json.dumps(dict(tracker=a.tracker, **out), indent=1))


if __name__ == "__main__":
    main()
