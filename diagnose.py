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


def trace(g, passes, events, cfg=B.DetectCfg()):
    """For each real pass, replay the detector's gates on the attacker's samples."""
    dt = cfg.dt
    cool_n = int(round(cfg.cooldown_s / dt)); close_n = int(round(cfg.close_s / dt))
    win_n = int(round(cfg.outcome_window_s / dt)); hold_n = int(round(cfg.hold_s / dt))
    n = len(g.t); start_n = max(cool_n, close_n)
    cnt = dict(no_cand=0, wrong_def=0, lapped_gate=0, window_invalid=0, passed_all_gates=0)
    rows = []
    for X, Y, t0, t1 in passes:
        if X not in g.drv or Y not in g.drv:
            continue
        A, D = g.drv[X], g.drv[Y]
        i0 = int(np.searchsorted(g.t, t0 - 10)); i1 = int(np.searchsorted(g.t, t1 + 10))
        fl = np.where(D.ahead[i0:i1] == X)[0]
        if len(fl) == 0:
            continue
        f = i0 + int(fl[0])
        ok = A.valid & np.isfinite(A.dist) & (A.ahead != "")
        spd = np.maximum(np.nan_to_num(A.speed, nan=0.0) / 3.6, cfg.min_speed_ms)
        with np.errstate(invalid="ignore", divide="ignore"):
            gap = A.dist / spd
        in_r = ok & (gap <= cfg.enter_gap_s); out_ok = ok & (gap > cfg.enter_gap_s)
        c = np.concatenate([[0], np.cumsum(out_ok)])
        allp = np.zeros(n, bool)
        allp[cool_n:] = (c[cool_n:n] - c[0:n - cool_n]) == cool_n
        cand = np.where(in_r & allp & (np.arange(n) >= start_n))[0]
        cw = cand[(cand >= f - win_n) & (cand <= f)]
        run = 0
        while f - run >= 0 and in_r[f - run - 1 if f - run - 1 >= 0 else 0] and run < n:
            run += 1
            if run > 4000: break
        why = "no_cand"
        if len(cw):
            i = int(cw[-1]); why = "passed_all_gates"
            if A.ahead[i] != Y:
                why = "wrong_def"
            else:
                pa, pd_ = A.pos_prev[i], D.pos_prev[i]
                if np.isfinite(pa) and np.isfinite(pd_) and not (pa > pd_):
                    why = "lapped_gate"
                elif i + win_n + hold_n >= n or not (A.valid[i:i + win_n + hold_n + 1].all()
                                                     and D.valid[i:i + win_n + hold_n + 1].all()):
                    why = "window_invalid"
        cnt[why] += 1
        rows.append(dict(X=X, Y=Y, t_flip=round(float(g.t[f]), 1), in_range_s_before_flip=run * dt,
                         n_cand_in_40s=len(cw), blocked_by=why))
    print("\nWhy real passes are not labelled success:", cnt)
    print(pd.DataFrame(rows).to_string(index=False))


def gap_profiles(g, passes, cfg=B.DetectCfg(), k=6, back=60, step=5):
    """Attacker gap (s) to the car ahead every `step` s in the `back` s before the pass."""
    dt = cfg.dt
    shown = 0
    print("\nAttacker gap_s before the pass ('.'=gap>5s, '-'=other car ahead / no data); "
          f"columns = -{back}s ... 0s, step {step}s")
    for X, Y, t0, t1 in passes:
        if shown >= k or X not in g.drv or Y not in g.drv:
            continue
        A, D = g.drv[X], g.drv[Y]
        i0 = int(np.searchsorted(g.t, t0 - 10)); i1 = int(np.searchsorted(g.t, t1 + 10))
        fl = np.where(D.ahead[i0:i1] == X)[0]
        if len(fl) == 0:
            continue
        f = i0 + int(fl[0]); cells = []
        for s_ in range(-back, 1, step):
            i = f + int(s_ / dt)
            if i < 0:
                cells.append("  ?  "); continue
            if A.ahead[i] != Y or not np.isfinite(A.dist[i]):
                cells.append("  -  "); continue
            gp = A.dist[i] / max(np.nan_to_num(A.speed[i]) / 3.6, 15.0)
            cells.append("  .  " if gp > 5 else f"{gp:5.2f}")
        print(f"{X:>3}>{Y:<3}", "".join(cells)); shown += 1


def sweep(g, passes, events_unused=None):
    import dataclasses, itertools
    usable = []
    for X, Y, t0, t1 in passes:
        if X in g.drv and Y in g.drv:
            i0 = int(np.searchsorted(g.t, t0 - 10)); i1 = int(np.searchsorted(g.t, t1 + 10))
            if (g.drv[Y].ahead[i0:i1] == X).any():
                usable.append((X, Y, t0, t1))
    print(f"\nSweep on {len(usable)} usable real passes "
          "(recall = real passes matched by a success-labelled attempt)")
    print(" cooldown enter_gap window | attempts successes succ_rate | recall")
    best = []
    for cool, eg, win in itertools.product([5.0, 3.0, 2.0, 1.0, 0.0], [1.0, 1.5, 2.0], [40.0, 30.0]):
        cfg = dataclasses.replace(B.DetectCfg(), cooldown_s=cool, enter_gap_s=eg, outcome_window_s=win)
        ev = pd.DataFrame(B.detect_attempts(g, cfg))
        if ev.empty:
            print(f" {cool:8.1f} {eg:9.1f} {win:6.0f} | 0"); continue
        hit = 0
        for X, Y, t0, t1 in usable:
            e = ev[(ev.attacker.astype(str) == X) & (ev.defender.astype(str) == Y)
                   & (ev.t0_s >= t0 - 90) & (ev.t0_s <= t1 + 10) & (ev.y == 1)]
            hit += len(e) > 0
        print(f" {cool:8.1f} {eg:9.1f} {win:6.0f} | {len(ev):8d} {int(ev.y.sum()):9d} "
              f"{ev.y.mean():9.2f} | {hit}/{len(usable)}")


def main():
    import fastf1
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2023)
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--cache-dir", default="ff1cache")
    ap.add_argument("--events", default="data_noscreen/events_2023_01.csv")
    a = ap.parse_args()
    import os, pickle
    fastf1.Cache.enable_cache(a.cache_dir)
    s = fastf1.get_session(a.year, a.round, "R")
    s.load(laps=True, telemetry=True, weather=False, messages=False)
    pk = os.path.join(os.path.dirname(a.events) or ".", f"grid_{a.year}_{a.round:02d}.pkl")
    if os.path.exists(pk):
        g = pickle.load(open(pk, "rb")); print("[cache] grid loaded from", pk)
    else:
        g = B.build_race_grid(s, B.DetectCfg(), a.year, a.round, screen_s=0)
        pickle.dump(g, open(pk, "wb"))
    ev = pd.read_csv(a.events)
    gp = gt_passes(s.laps)
    funnel(g, gp, ev)
    trace(g, gp, ev)
    gap_profiles(g, gp)
    sweep(g, gp)


if __name__ == "__main__":
    main()
