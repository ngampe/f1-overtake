"""Where do real overtakes get lost?  Compares lap-position 'ground truth' passes with
what the grid / detector sees.   python diagnose.py --year 2023 --round 1 \
        --events data_noscreen/events_2023_01.csv
Ground truth (approximate): driver X was behind Y at the end of lap n-1 and ahead at the
end of lap n, neither pitting on lap n-1 or n."""
import argparse
import numpy as np
import pandas as pd
import baseline as B


def gt_passes(laps):
    L = laps.copy()
    L["pit"] = L["PitInTime"].notna() | L["PitOutTime"].notna()
    L = L[["DriverNumber", "LapNumber", "Position", "pit", "LapStartTime", "Time"]]
    key = {(r.DriverNumber, int(r.LapNumber)): r for r in L.itertuples()}
    out = []
    for n in sorted({k[1] for k in key}):
        if n < 2:
            continue
        cur = {d: key[(d, n)] for (d, m) in key if m == n and (d, n - 1) in key}
        for X, rx in cur.items():
            px0 = key[(X, n - 1)]
            for Y, ry in cur.items():
                if X == Y:
                    continue
                py0 = key[(Y, n - 1)]
                if any(pd.isna(v) for v in (rx.Position, ry.Position, px0.Position, py0.Position)):
                    continue
                if px0.pit or py0.pit or rx.pit or ry.pit:
                    continue
                if px0.Position > py0.Position and rx.Position < ry.Position:
                    if pd.isna(rx.LapStartTime) or pd.isna(rx.Time):
                        continue
                    out.append((X, Y, rx.LapStartTime.total_seconds(), rx.Time.total_seconds()))
    return out


def funnel(g, passes, events, cfg=B.DetectCfg(), verbose=8):
    dt = cfg.dt
    hold_n = int(round(cfg.hold_s / dt))
    rows = []
    for X, Y, t0, t1 in passes:
        if X not in g.drv or Y not in g.drv:
            continue
        a, d = g.drv[X], g.drv[Y]
        i0 = int(np.searchsorted(g.t, t0 - 10)); i1 = int(np.searchsorted(g.t, t1 + 10))
        sl = slice(i0, i1)
        both_valid = float(np.mean(a.valid[sl] & d.valid[sl])) if i1 > i0 else 0.0
        flipped = (d.ahead[sl] == X)
        run = np.convolve(flipped.astype(int), np.ones(hold_n, int), "valid") if len(flipped) >= hold_n else np.array([0])
        has_flip = bool((run == hold_n).any())
        first = int(np.argmax(flipped)) + i0 if flipped.any() else None
        mg = np.nan
        if first is not None:
            lo = max(0, first - int(30 / dt))
            seg = slice(lo, first + 1)
            spd = np.maximum(np.nan_to_num(a.speed[seg]) / 3.6, cfg.min_speed_ms)
            gp = np.where(a.ahead[seg] == Y, a.dist[seg] / spd, np.nan)
            mg = float(np.nanmin(gp)) if np.isfinite(gp).any() else np.nan
        det_any = det_pos = False
        if len(events):
            e = events[(events.attacker.astype(str) == X) & (events.defender.astype(str) == Y)
                       & (events.t0_s >= t0 - 10) & (events.t0_s <= t1 + 10)]
            det_any, det_pos = len(e) > 0, bool((e.y == 1).any())
        rows.append(dict(X=X, Y=Y, t0=t0, both_valid=both_valid, has_flip=has_flip,
                         min_gap_before=mg, det_any=det_any, det_pos=det_pos))
    R = pd.DataFrame(rows)
    if R.empty:
        print("no ground-truth passes"); return R
    N = len(R)
    s1 = R.both_valid >= 0.5
    s2 = s1 & R.has_flip
    s3 = s2 & (R.min_gap_before <= cfg.enter_gap_s)
    print(f"ground-truth passes (lap-position swaps, no pits): {N}")
    print(f"  1 both cars have valid grid samples in the lap : {int(s1.sum())}")
    print(f"  2 ... and 'DriverAhead' flips to the attacker (>= {cfg.hold_s}s) : {int(s2.sum())}")
    print(f"  3 ... and attacker was within {cfg.enter_gap_s}s before the flip   : {int(s3.sum())}")
    print(f"  detected as ANY attempt (right pair, right time)     : {int(R.det_any.sum())}")
    print(f"  detected AND labelled success                        : {int(R.det_pos.sum())}")
    if R.min_gap_before.notna().any():
        print("  min gap (s) before flip, quartiles:",
              np.round(R.min_gap_before.dropna().quantile([.25, .5, .75]).to_numpy(), 2))
    print(R.head(verbose).to_string(index=False))
    return R


def main():
    import fastf1
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2023)
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--cache-dir", default="ff1cache")
    ap.add_argument("--events", default="data_noscreen/events_2023_01.csv")
    a = ap.parse_args()
    fastf1.Cache.enable_cache(a.cache_dir)
    s = fastf1.get_session(a.year, a.round, "R")
    s.load(laps=True, telemetry=True, weather=False, messages=False)
    g = B.build_race_grid(s, B.DetectCfg(), a.year, a.round, screen_s=0)
    funnel(g, gt_passes(s.laps), pd.read_csv(a.events))


if __name__ == "__main__":
    main()
