import numpy as np, pandas as pd
from speed import smooth, track_stats, pixels_to_ground


def test_constant_velocity_speed():
    n = 60  # 2 s at 30 fps, moving 3 px/frame in x => 90 px/s
    d = pd.DataFrame(dict(frame=np.arange(1, n + 1), id=1, x=np.arange(n) * 3.0, y=0.0, w=10.0, h=20.0))
    d["fx"], d["fy"] = d.x + d.w / 2, d.y + d.h
    st = track_stats(d, 30)
    assert abs(st.mean_px_s[0] - 90) < 5  # moving-average edge padding biases slightly low


def test_short_tracks_dropped():
    d = pd.DataFrame(dict(frame=np.arange(1, 10), id=1, fx=np.arange(9.0), fy=0.0))
    assert track_stats(d, 30).empty


def test_smooth_keeps_length():
    assert len(smooth(np.arange(20.0))) == 20


def test_homography_scale():
    # image square 100px -> 2 m ground square: (50,50)px must map to (1,1)m
    p = pixels_to_ground([[50, 50]], [[0, 0], [100, 0], [100, 100], [0, 100]], [[0, 0], [2, 0], [2, 2], [0, 2]])
    assert np.allclose(p, [[1, 1]], atol=1e-4)


def test_tracker_output_format():
    # MOT format has 10 columns: frame,id,x,y,w,h,conf,-1,-1,-1
    import glob
    for f in glob.glob("results/tracks/*/*.txt")[:3]:
        assert len(open(f).readline().split(",")) == 10
