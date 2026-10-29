"""Download selected MOT17 *train* sequences (images + GT) via HTTP range requests.

MOT17.zip is 5.9 GB (incl. test); we only fetch the needed members (~0.5 GB).
The three detector variants (DPM/FRCNN/SDP) share identical images and GT, so we take FRCNN.
"""
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from remotezip import RemoteZip

URL = "https://motchallenge.net/data/MOT17.zip"
SEQS = ["02", "04", "05", "09", "10"]  # 3 static + 2 moving cameras
OUT = Path("data")

def main():
    OUT.mkdir(exist_ok=True)
    with RemoteZip(URL) as z:
        sizes = {i.filename: i.file_size for i in z.infolist()}
        names = [n for n in z.namelist()
                 if any(n.startswith(f"MOT17/train/MOT17-{s}-FRCNN/") for s in SEQS)
                 and (("/img1/" in n) or n.endswith("gt/gt.txt") or n.endswith("seqinfo.ini"))
                 and not n.endswith("/")]
        total = sum(z.getinfo(n).file_size for n in names) / 1e6
        print(f"{len(names)} files, {total:.0f} MB")
    for n in names:
        (OUT / n).parent.mkdir(parents=True, exist_ok=True)  # avoid makedirs race across threads
    tl = threading.local()

    def get(n):
        if (OUT / n).exists() and (OUT / n).stat().st_size == sizes[n]:
            return  # already complete (resume-safe)
        if not hasattr(tl, "z"):
            tl.z = RemoteZip(URL)  # one connection per thread
        tl.z.extract(n, OUT)

    with ThreadPoolExecutor(24) as ex:
        for i, _ in enumerate(ex.map(get, names)):
            if i % 500 == 0:
                print(i, file=sys.stderr, flush=True)
    # shorter, detector-agnostic dir names: data/train/MOT17-02
    for s in SEQS:
        src, dst = OUT / "MOT17" / "train" / f"MOT17-{s}-FRCNN", OUT / "MOT17" / "train" / f"MOT17-{s}"
        if src.exists() and not dst.exists():
            src.rename(dst)

if __name__ == "__main__":
    main()
