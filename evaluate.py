"""Score results/tracks/* against MOT17 train GT with TrackEval (HOTA, CLEAR/MOTA, IDF1) and merge FPS.

Output: results/metrics.json, results/metrics.csv. Combined = all sequences pooled (TrackEval 'COMBINED_SEQ').
NOTE: MOT17 *train* GT is public; this is not an official test-server submission.
"""
import json, csv
from pathlib import Path
import trackeval

ROOT = Path(__file__).parent
RES = ROOT / "results"
GT = ROOT / "data/MOT17/train"


def main():
    trackers = sorted(p.name for p in (RES / "tracks").iterdir() if p.is_dir())
    seqs = sorted(p.name for p in GT.iterdir() if p.is_dir())
    seqmap = RES / "seqmap.txt"
    seqmap.write_text("name\n" + "\n".join(seqs) + "\n")
    # TrackEval expects <TRACKERS_FOLDER>/<tracker>/data/<seq>.txt
    tf = RES / "te_trackers"
    for t in trackers:
        d = tf / t / "data"
        d.mkdir(parents=True, exist_ok=True)
        for s in seqs:
            (d / f"{s}.txt").write_text((RES / "tracks" / t / f"{s}.txt").read_text())
    dcfg = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dcfg.update(GT_FOLDER=str(GT), TRACKERS_FOLDER=str(tf), SEQMAP_FILE=str(seqmap), SKIP_SPLIT_FOL=True,
                TRACKERS_TO_EVAL=trackers, BENCHMARK="MOT17", SPLIT_TO_EVAL="train", PRINT_CONFIG=False,
                OUTPUT_FOLDER=str(RES / "te_out"), OUTPUT_SUB_FOLDER="", TRACKER_SUB_FOLDER="data")
    ecfg = trackeval.Evaluator.get_default_eval_config()
    ecfg.update(USE_PARALLEL=False, PRINT_CONFIG=False, PRINT_RESULTS=False, PRINT_ONLY_COMBINED=True,
                OUTPUT_SUMMARY=False, OUTPUT_DETAILED=False, PLOT_CURVES=False)
    metrics = [trackeval.metrics.HOTA({"PRINT_CONFIG": False}), trackeval.metrics.CLEAR({"PRINT_CONFIG": False}),
               trackeval.metrics.Identity({"PRINT_CONFIG": False})]
    out, _ = trackeval.Evaluator(ecfg).evaluate([trackeval.datasets.MotChallenge2DBox(dcfg)], metrics)
    timing = json.loads((RES / "timing.json").read_text())
    rows, per_seq = [], {}
    for t in trackers:
        r = out["MotChallenge2DBox"][t]
        c = r["COMBINED_SEQ"]["pedestrian"]
        fr = sum(v["frames"] for v in timing[t].values())
        sec = sum(v["seconds"] for v in timing[t].values())
        rows.append(dict(tracker=t, HOTA=100 * c["HOTA"]["HOTA"].mean(), DetA=100 * c["HOTA"]["DetA"].mean(),
                         AssA=100 * c["HOTA"]["AssA"].mean(), MOTA=100 * c["CLEAR"]["MOTA"], IDF1=100 * c["Identity"]["IDF1"],
                         IDSW=int(c["CLEAR"]["IDSW"]), FP=int(c["CLEAR"]["CLR_FP"]), FN=int(c["CLEAR"]["CLR_FN"]),
                         FPS=fr / sec))
        per_seq[t] = {s: dict(HOTA=100 * r[s]["pedestrian"]["HOTA"]["HOTA"].mean(), MOTA=100 * r[s]["pedestrian"]["CLEAR"]["MOTA"],
                              IDF1=100 * r[s]["pedestrian"]["Identity"]["IDF1"], IDSW=int(r[s]["pedestrian"]["CLEAR"]["IDSW"]),
                              fps=timing[t][s]["fps"]) for s in seqs}
    rows = [{k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()} for r in rows]
    (RES / "metrics.json").write_text(json.dumps({"combined": rows, "per_sequence": per_seq}, indent=1))
    with open(RES / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
