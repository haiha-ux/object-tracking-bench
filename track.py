"""Run YOLO11 person detection + ByteTrack / BoT-SORT on MOT17 train sequences.

Writes MOT-format tracks to results/tracks/<det>_<tracker>_<backend>/<seq>.txt and
wall-clock timing (end-to-end: JPEG read + preprocess + detect + NMS + track) to results/timing.json.
"""
import argparse, json, time
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).parent
DATA = ROOT / "data/MOT17/train"
RES = ROOT / "results"
IMGSZ = 1280  # people are small in MOT17 (1920x1080)


def get_weights(name, backend):
    pt = ROOT / "weights" / f"{name}.pt"
    pt.parent.mkdir(exist_ok=True)
    if not pt.exists():
        YOLO(f"{name}.pt")  # downloads into cwd
        Path(f"{name}.pt").replace(pt)
    if backend == "pt":
        return pt
    eng = pt.with_suffix(".engine")
    if not eng.exists():
        YOLO(str(pt)).export(format="engine", half=True, imgsz=IMGSZ, batch=1, device=0)
    return eng


def run_seq(weights, tracker, seq, out_file, device=0):
    model = YOLO(str(weights), task="detect")
    img_dir = DATA / seq / "img1"
    n = len(list(img_dir.glob("*.jpg")))
    # warm-up (CUDA / cudnn / TRT context) on a separate model instance of same weights
    for _ in model.predict(source=str(img_dir), stream=True, imgsz=IMGSZ, device=device, verbose=False, classes=[0], conf=0.1, vid_stride=max(n // 8, 1)):
        pass
    lines = []
    t0 = time.perf_counter()
    gen = model.track(source=str(img_dir), stream=True, imgsz=IMGSZ, device=device, tracker=f"{tracker}.yaml",
                      classes=[0], conf=0.1, persist=True, verbose=False)  # persist=True: image-folder sources would otherwise reset the tracker every frame
    for i, r in enumerate(gen, 1):
        b = r.boxes
        if b is None or b.id is None:
            continue
        for (x1, y1, x2, y2), tid, c in zip(b.xyxy.tolist(), b.id.int().tolist(), b.conf.tolist()):
            lines.append(f"{i},{tid},{x1:.2f},{y1:.2f},{x2-x1:.2f},{y2-y1:.2f},{c:.4f},-1,-1,-1")
    dt = time.perf_counter() - t0
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(lines) + "\n")
    return n, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["yolo11n", "yolo11s"])
    ap.add_argument("--trackers", nargs="+", default=["bytetrack", "botsort"])
    ap.add_argument("--backends", nargs="+", default=["pt"])
    ap.add_argument("--seqs", nargs="+", default=None)
    a = ap.parse_args()
    seqs = a.seqs or sorted(p.name for p in DATA.iterdir() if p.is_dir())
    timing_f = RES / "timing.json"
    timing = json.loads(timing_f.read_text()) if timing_f.exists() else {}
    for m in a.models:
        for be in a.backends:
            w = get_weights(m, be)
            for trk in a.trackers:
                name = f"{m}_{trk}_{be}"
                for s in seqs:
                    n, dt = run_seq(w, trk, s, RES / "tracks" / name / f"{s}.txt")
                    timing.setdefault(name, {})[s] = {"frames": n, "seconds": round(dt, 3), "fps": round(n / dt, 2)}
                    print(name, s, f"{n/dt:.1f} fps", flush=True)
                    RES.mkdir(exist_ok=True)
                    timing_f.write_text(json.dumps(timing, indent=1))


if __name__ == "__main__":
    main()
