#!/usr/bin/env python3
"""
Pre-T0 gap, closing speed, tire compound/age of BOTH
          attacker and defender, DRS availability, track category, race context.
Baselines:
            (1) gap-distance heuristic
            (2) logistic regression on the full feature set
            (3) XGBoost tuned with grid search and Optuna and validation PR-AUC
Circuit-level train/validation/test split 20% of circuits held out
          as test, random split as a reference for ROC-AUC, PR-AUC, F1, calibration error
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# Config
# =============================================================================


@dataclass
class DetectCfg:
    dt: float = 0.5               # grid step (s)
    enter_gap_s: float = 1.0      # "attempt range" (DRS-style 1 s)
    cooldown_s: float = 0.0       # 0 = an attempt starts whenever the attacker is in range and no earlier
                                  # attempt window is still open. (5 s hid ~95% of real passes in Bahrain 2023:
                                  # attackers follow within ~1 s for 10-60 s before passing.)
    close_s: float = 3.0          # window for closing speed
    outcome_window_s: float = 40.0
    hold_s: float = 5.0           # a pass must hold this long
    min_speed_ms: float = 15.0    # floor when converting gap distance -> gap time


FEATURES = [
    "gap_s", "gap_m", "closing_speed_ms",
    "att_speed_kmh", "speed_diff_kmh",
    "drs_eligible", "drs_on",
    "att_compound", "def_compound", "compound_diff",
    "att_tyre_age", "def_tyre_age", "tyre_age_diff",
    "lap_number", "laps_remaining", "race_progress",
    "track_high_speed",
]
# NOTE: continuous per-circuit constants (e.g. average lap speed) are deliberately
# NOT features: a tree could use them to fingerprint the circuit and defeat the
# leave-one-circuit-out test. Only the binary track category is used (Sec. 2.1).

COMPOUND_RANK = {
    "HYPERSOFT": 0, "ULTRASOFT": 0, "SUPERSOFT": 0, "SOFT": 0,
    "MEDIUM": 1, "HARD": 2, "SUPERHARD": 2, "INTERMEDIATE": 3, "WET": 4,
}


# =============================================================================
# Event detection -- pure numpy, testable without any F1 data
# =============================================================================


@dataclass
class DriverArrays:
    ahead: np.ndarray       # object array of driver-number strings ('' = none)
    dist: np.ndarray        # metres to car ahead
    speed: np.ndarray       # km/h
    drs: np.ndarray         # raw DRS channel value
    lapno: np.ndarray
    compound: np.ndarray    # compound rank (nan = unknown)
    tyre_age: np.ndarray
    pos_prev: np.ndarray    # race position at the end of the previous lap
    valid: np.ndarray       # bool: usable sample (green flag, not pit, lap > 1, has telemetry)


@dataclass
class RaceGrid:
    t: np.ndarray                       # session time (s) of each grid sample
    drv: dict                           # driver number (str) -> DriverArrays
    total_laps: int
    meta: dict = field(default_factory=dict)


def _has_pass_that_holds(passed: np.ndarray, win_n: int, hold_n: int) -> bool:
    """True if `passed` has a run of >= hold_n Trues starting at an index <= win_n."""
    cs = np.concatenate([[0], np.cumsum(passed.astype(int))])
    run = cs[hold_n:] - cs[:-hold_n]            # run[s] = sum(passed[s:s+hold_n])
    return bool(np.any(run[: win_n + 1] == hold_n))


def detect_attempts(g: RaceGrid, cfg: DetectCfg = DetectCfg()) -> list[dict]:
    dt = cfg.dt
    cool_n = int(round(cfg.cooldown_s / dt))
    close_n = int(round(cfg.close_s / dt))
    win_n = int(round(cfg.outcome_window_s / dt))
    hold_n = int(round(cfg.hold_s / dt))
    n = len(g.t)
    start_n = max(cool_n, close_n)
    out: list[dict] = []

    for a, A in g.drv.items():
        ok = A.valid & np.isfinite(A.dist) & (A.ahead != "")
        spd = np.maximum(np.nan_to_num(A.speed, nan=0.0) / 3.6, cfg.min_speed_ms)
        with np.errstate(invalid="ignore", divide="ignore"):
            gap_s = A.dist / spd
            in_range = ok & (gap_s <= cfg.enter_gap_s)
            out_ok = ok & (gap_s > cfg.enter_gap_s)

        # all_prev[i] == True  <=>  out_ok[i-cool_n : i] are all True
        c = np.concatenate([[0], np.cumsum(out_ok)])
        all_prev = np.zeros(n, dtype=bool)
        if n > cool_n:
            all_prev[cool_n:] = (c[cool_n:n] - c[0: n - cool_n]) == cool_n
        cand = np.where(in_range & all_prev & (np.arange(n) >= start_n))[0]

        next_allowed = 0
        for i in cand:
            if i < next_allowed:
                continue
            d = A.ahead[i]
            if d not in g.drv:
                continue
            D = g.drv[d]
            # skip lapped defenders: attacker must be classified BEHIND the defender
            pa, pd_ = A.pos_prev[i], D.pos_prev[i]
            if np.isfinite(pa) and np.isfinite(pd_) and not (pa > pd_):
                continue
            j_end = i + win_n + hold_n
            if j_end >= n:
                continue
            if not (A.valid[i: j_end + 1].all() and D.valid[i: j_end + 1].all()):
                continue                         # pit / yellow / SC in window -> no label
            next_allowed = i + win_n             # one attempt per attacker per window

            passed = D.ahead[i: j_end + 1] == a
            y = int(_has_pass_that_holds(passed, win_n, hold_n))

            # ---- features: ONLY samples at or before T0 (index i) ----
            k = i - close_n
            drs = A.drs[i]
            ra = A.lapno[i]
            out.append(dict(
                circuit=g.meta.get("circuit", "?"), year=g.meta.get("year", 0),
                round=g.meta.get("round", 0), attacker=a, defender=d,
                t0_s=float(g.t[i]),
                gap_s=float(gap_s[i]), gap_m=float(A.dist[i]),
                closing_speed_ms=float((A.dist[k] - A.dist[i]) / (close_n * dt)),
                att_speed_kmh=float(A.speed[i]),
                speed_diff_kmh=float(A.speed[i] - D.speed[i]),
                drs_eligible=np.nan if np.isnan(drs) else float(drs >= 8),
                drs_on=np.nan if np.isnan(drs) else float(drs >= 10),
                att_compound=float(A.compound[i]), def_compound=float(D.compound[i]),
                compound_diff=float(A.compound[i] - D.compound[i]),
                att_tyre_age=float(A.tyre_age[i]), def_tyre_age=float(D.tyre_age[i]),
                tyre_age_diff=float(A.tyre_age[i] - D.tyre_age[i]),
                lap_number=float(ra), laps_remaining=float(g.total_laps - ra),
                race_progress=float(ra / g.total_laps),
                track_avg_speed=float(g.meta.get("track_avg_speed", np.nan)),
                y=y,
            ))
    return out


# =============================================================================
# FastF1 loading
# =============================================================================


def _lap_is_valid(lap) -> bool:
    ts = lap.get("TrackStatus")
    green = (not pd.isna(ts)) and set(str(ts)) <= {"1"}
    return bool(int(lap["LapNumber"]) > 1 and green
                and pd.isna(lap["PitInTime"]) and pd.isna(lap["PitOutTime"]))


def _screen_laps(laps: pd.DataFrame, screen_s: float):
    """Cheap pre-filter to avoid computing driver-ahead telemetry for every lap.

    A pair of cars that is within `screen_s` at the end of lap n keeps BOTH cars for
    laps n and n+1: lap n holds the approach, lap n+1 holds the rest of the outcome
    window (a successful pass often ends with the attacker pulling away, so filtering
    on the attacker alone would silently drop successful passes and bias the labels
    toward failures). Laps with missing timing are kept to be safe.
    Returns None when screening is disabled (screen_s <= 0)."""
    if screen_s <= 0:
        return None
    L = laps[["DriverNumber", "LapNumber", "Time", "Position"]].copy()
    L["t"] = L["Time"].dt.total_seconds()
    keep = set()
    for n, grp in L.groupby("LapNumber"):
        n = int(n)
        missing = grp[grp["t"].isna() | grp["Position"].isna()]
        for d in missing["DriverNumber"]:
            keep.update({(d, n), (d, n + 1)})
        grp = grp.dropna(subset=["t", "Position"]).sort_values("Position")
        ts, nums = grp["t"].to_numpy(), grp["DriverNumber"].to_numpy()
        for k in range(1, len(grp)):
            if ts[k] - ts[k - 1] <= screen_s:        # follower k is close to leader k-1
                for d in (nums[k], nums[k - 1]):
                    keep.update({(d, n), (d, n + 1)})
    return keep


def build_race_grid(session, cfg: DetectCfg, year: int, rnd: int, screen_s: float = 3.0,
                    max_laps_per_driver: int | None = None, log=print) -> RaceGrid:
    laps = session.laps
    drivers = [d for d in session.drivers if (laps["DriverNumber"] == d).any()]
    t_min = laps["LapStartTime"].dropna().min().total_seconds()
    t_max = laps["Time"].dropna().max().total_seconds()
    gt = np.arange(t_min, t_max, cfg.dt)
    n = len(gt)
    total_laps = int(getattr(session, "total_laps", None) or laps["LapNumber"].max())

    pos_prev_map = {(r.DriverNumber, int(r.LapNumber)): r.Position
                    for r in laps.itertuples() if not pd.isna(r.Position)}
    keep = _screen_laps(laps, screen_s)

    arrays = {}
    for drv in drivers:
        arrays[drv] = DriverArrays(
            ahead=np.full(n, "", dtype=object), dist=np.full(n, np.nan), speed=np.full(n, np.nan),
            drs=np.full(n, np.nan), lapno=np.full(n, np.nan), compound=np.full(n, np.nan),
            tyre_age=np.full(n, np.nan), pos_prev=np.full(n, np.nan), valid=np.zeros(n, dtype=bool))

    n_done = n_fail = 0
    t_start = time.time()
    for drv in drivers:
        A = arrays[drv]
        count = 0
        for _, lap in laps[laps["DriverNumber"] == drv].iterlaps():
            ln = int(lap["LapNumber"])
            if not _lap_is_valid(lap) or pd.isna(lap["LapStartTime"]) or pd.isna(lap["Time"]):
                continue
            if keep is not None and (drv, ln) not in keep:
                continue
            if max_laps_per_driver is not None and count >= max_laps_per_driver:
                break
            try:
                cd = lap.get_car_data(pad=1, pad_side="both")
                if len(cd) < 6:
                    continue
                # per-lap driver-ahead, as recommended by the FastF1 docs (integration error)
                da = cd.iloc[1:-1].add_driver_ahead()
                ts = da["SessionTime"].dt.total_seconds().to_numpy()
                ahead = da["DriverAhead"]
                ahead = np.where(pd.isna(ahead), "", ahead.astype(str)).astype(object)
                vals = dict(dist=da["DistanceToDriverAhead"].to_numpy(float),
                            speed=da["Speed"].to_numpy(float), drs=da["DRS"].to_numpy(float))
            except Exception as e:  # noqa: BLE001 -- keep going, count failures
                n_fail += 1
                if n_fail <= 3:
                    log(f"    [warn] telemetry failed for driver {drv} lap {ln}: {type(e).__name__}: {e}")
                continue
            m = (gt >= ts[0]) & (gt <= ts[-1])
            gi = np.where(m)[0]
            if len(gi) == 0:
                continue
            si = np.searchsorted(ts, gt[gi], side="right") - 1     # zero-order hold
            A.ahead[gi] = ahead[si]
            for key, v in vals.items():
                getattr(A, key)[gi] = v[si]
            A.lapno[gi] = ln
            A.compound[gi] = COMPOUND_RANK.get(str(lap.get("Compound")).upper(), np.nan)
            A.tyre_age[gi] = lap.get("TyreLife", np.nan)
            A.pos_prev[gi] = pos_prev_map.get((drv, ln - 1), np.nan)
            A.valid[gi] = True
            count += 1
            n_done += 1
        log(f"    driver {drv}: {count} laps processed ({time.time() - t_start:.0f}s elapsed)")

    meta = dict(circuit=str(session.event["Location"]), event=str(session.event["EventName"]),
                year=int(year), round=int(rnd))
    try:
        meta["track_avg_speed"] = float(session.laps.pick_fastest().get_car_data()["Speed"].mean())
    except Exception:  # noqa: BLE001
        meta["track_avg_speed"] = np.nan
    log(f"    laps processed: {n_done}, telemetry failures: {n_fail}")
    return RaceGrid(t=gt, drv=arrays, total_laps=total_laps, meta=meta)


def cmd_build(args):
    import fastf1
    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)
    fastf1.Cache.enable_cache(args.cache_dir)
    cfg = DetectCfg()
    failed: list[str] = []
    for year in args.years:
        if args.rounds == ["all"]:
            sched = fastf1.get_event_schedule(year, include_testing=False)
            rounds = [int(r) for r in sched["RoundNumber"] if int(r) > 0]
        else:
            rounds = [int(r) for r in args.rounds]
        for rnd in rounds:
            path = os.path.join(args.data_dir, f"events_{year}_{rnd:02d}.csv")
            if os.path.exists(path) and not args.force:
                print(f"[skip] {path} exists (use --force to rebuild)")
                continue
            print(f"[build] {year} round {rnd}: loading session ...")
            t0 = time.time()
            try:
                s = fastf1.get_session(year, rnd, "R")
                s.load(laps=True, telemetry=True, weather=False, messages=False)
                grid = build_race_grid(s, cfg, year, rnd, screen_s=args.screen_s,
                                       max_laps_per_driver=args.max_laps_per_driver)
                ev = pd.DataFrame(detect_attempts(grid, cfg))
            except Exception as e:  # noqa: BLE001 -- e.g. cancelled race: log it and keep going
                msg = f"{year} round {rnd}: {type(e).__name__}: {str(e)[:200]}"
                print(f"[FAILED] {msg}")
                failed.append(msg)
                continue
            ev.to_csv(path, index=False)
            npos = int(ev["y"].sum()) if len(ev) else 0
            print(f"[build] {grid.meta['event']} ({grid.meta['circuit']}): {len(ev)} attempts, "
                  f"{npos} successful -> {path}  [{time.time() - t0:.0f}s]")
    if failed:
        print("\n[build] rounds that FAILED (re-run to retry; existing files are skipped):")
        for m in failed:
            print("   ", m)


# =============================================================================
# Data, splits, metrics
# =============================================================================


def load_events(data_dir: str) -> pd.DataFrame:
    files = sorted(f for f in os.listdir(data_dir) if f.startswith("events_") and f.endswith(".csv"))
    if not files:
        sys.exit(f"No events_*.csv in {data_dir}. Run `python baseline.py build ...` first.")
    df = pd.concat([pd.read_csv(os.path.join(data_dir, f)) for f in files], ignore_index=True)
    return add_track_category(df)


def add_track_category(df: pd.DataFrame) -> pd.DataFrame:
    """Binary track category (high-speed vs technical) = median split of circuit average
    lap speed over the circuits in the dataset. Swap in a hand-tagged table here if preferred."""
    if "track_high_speed" in df.columns:
        return df
    cs = df.groupby("circuit")["track_avg_speed"].mean()
    thr = cs.median()
    df = df.copy()
    df["track_high_speed"] = (df["circuit"].map(cs) >= thr).astype(float) if cs.notna().any() else np.nan
    return df


def circuit_split(df: pd.DataFrame, test_frac=0.2, val_frac=0.2, seed=0):
    circuits = sorted(df["circuit"].unique())
    if len(circuits) < 3:
        sys.exit(f"Need >= 3 distinct circuits for a circuit-level split, found {len(circuits)}. "
                 f"Build more races (e.g. --rounds 1 2 3 4 5 6 7 8).")
    n_test = max(1, round(test_frac * len(circuits)))
    n_val = max(1, round(val_frac * (len(circuits) - n_test)))
    for attempt in range(200):                       # reshuffle until splits contain both classes
        rng = np.random.default_rng(seed + attempt)
        perm = list(rng.permutation(circuits))
        te, va, tr = perm[:n_test], perm[n_test:n_test + n_val], perm[n_test + n_val:]
        parts = [df[df.circuit.isin(p)] for p in (tr, va, te)]
        if all(len(p) > 0 and p["y"].nunique() == 2 for p in parts):
            return parts[0], parts[1], parts[2], dict(train=tr, val=va, test=te)
    sys.exit("Could not find a circuit split with both classes in every part; build more data.")


def random_split(df: pd.DataFrame, seed=0, val_frac=0.15, test_frac=0.15):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n_te, n_va = int(test_frac * len(df)), int(val_frac * len(df))
    te, va, tr = idx[:n_te], idx[n_te:n_te + n_va], idx[n_te + n_va:]
    return df.iloc[tr], df.iloc[va], df.iloc[te]


def ece_score(y, p, bins=10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    ids = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = ids == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def best_f1_threshold(y, s) -> float:
    from sklearn.metrics import precision_recall_curve
    p, r, th = precision_recall_curve(y, s)
    f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
    return float(th[int(np.argmax(f1))]) if len(th) else 0.5


def evaluate(y, s, thr, is_prob=True) -> dict:
    from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score
    y = np.asarray(y)
    r = dict(n=len(y), pos_rate=float(y.mean()),
             roc_auc=float(roc_auc_score(y, s)), pr_auc=float(average_precision_score(y, s)),
             f1=float(f1_score(y, (s >= thr).astype(int), zero_division=0)))
    r["ece"] = ece_score(y, s) if is_prob else np.nan
    r["brier"] = float(brier_score_loss(y, s)) if is_prob else np.nan
    return r


def bootstrap_ci(y, s, n_boot=200, seed=0):
    """Event-level bootstrap. Ignores within-race correlation, so treat as optimistic."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    rng = np.random.default_rng(seed)
    y, s = np.asarray(y), np.asarray(s)
    ro, pr = [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        ro.append(roc_auc_score(y[i], s[i]))
        pr.append(average_precision_score(y[i], s[i]))
    if len(ro) < 20:
        return (np.nan, np.nan), (np.nan, np.nan)
    return tuple(np.percentile(ro, [2.5, 97.5])), tuple(np.percentile(pr, [2.5, 97.5]))


# =============================================================================
# Models
# =============================================================================


def make_logreg():
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(max_iter=2000))


def make_xgb(params: dict, seed=0):
    from xgboost import XGBClassifier
    return XGBClassifier(objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                         random_state=seed, n_jobs=1, **params)


#!/usr/bin/env python3
"""Overtake-attempt success prediction from FastF1 telemetry.

Features (all measured at or before attempt start T0): gap, closing speed, speed difference,
DRS state, tyre compound and age of attacker and defender, race context, track category.

Baselines: gap-distance heuristic, logistic regression, XGBoost tuned by grid search,
XGBoost tuned by Optuna (both tuned on validation PR-AUC).

Evaluation: circuit-level train/val/test split (unseen circuits), plus a random event-level
split of the same sizes as a reference. Metrics: ROC-AUC, PR-AUC, F1, ECE, Brier.
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class DetectCfg:
    dt: float = 0.5
    enter_gap_s: float = 1.0
    close_s: float = 3.0
    outcome_window_s: float = 40.0
    hold_s: float = 5.0
    min_speed_ms: float = 15.0


# track_avg_speed is saved but deliberately not a feature: it would identify the circuit.
FEATURES = [
    "gap_s", "gap_m", "closing_speed_ms", "att_speed_kmh", "speed_diff_kmh",
    "drs_eligible", "drs_on",
    "att_compound", "def_compound", "compound_diff",
    "att_tyre_age", "def_tyre_age", "tyre_age_diff",
    "lap_number", "laps_remaining", "race_progress", "track_high_speed",
]

COMPOUND_RANK = {
    "HYPERSOFT": 0, "ULTRASOFT": 0, "SUPERSOFT": 0, "SOFT": 0,
    "MEDIUM": 1, "HARD": 2, "SUPERHARD": 2, "INTERMEDIATE": 3, "WET": 4,
}


@dataclass
class DriverArrays:
    ahead: np.ndarray
    dist: np.ndarray
    speed: np.ndarray
    drs: np.ndarray
    lapno: np.ndarray
    compound: np.ndarray
    tyre_age: np.ndarray
    pos_prev: np.ndarray
    valid: np.ndarray


@dataclass
class RaceGrid:
    t: np.ndarray
    drv: dict
    total_laps: int
    meta: dict = field(default_factory=dict)


def _has_pass_that_holds(passed, win_n, hold_n):
    cs = np.concatenate([[0], np.cumsum(passed.astype(int))])
    run = cs[hold_n:] - cs[:-hold_n]
    return bool(np.any(run[: win_n + 1] == hold_n))


def detect_attempts(g: RaceGrid, cfg: DetectCfg = DetectCfg()) -> list[dict]:
    dt = cfg.dt
    close_n, win_n, hold_n = (int(round(x / dt)) for x in (cfg.close_s, cfg.outcome_window_s, cfg.hold_s))
    n = len(g.t)
    out = []
    for a, A in g.drv.items():
        spd = np.maximum(np.nan_to_num(A.speed, nan=0.0) / 3.6, cfg.min_speed_ms)
        with np.errstate(invalid="ignore", divide="ignore"):
            gap_s = A.dist / spd
            cand = A.valid & np.isfinite(A.dist) & (A.ahead != "") & (gap_s <= cfg.enter_gap_s)
        cand[:close_n] = False

        next_allowed = 0
        for i in np.flatnonzero(cand):
            if i < next_allowed:
                continue
            d = A.ahead[i]
            if d not in g.drv:
                continue
            D = g.drv[d]
            pa, pd_ = A.pos_prev[i], D.pos_prev[i]
            if np.isfinite(pa) and np.isfinite(pd_) and not pa > pd_:
                continue
            j_end = i + win_n + hold_n
            if j_end >= n or not (A.valid[i: j_end + 1].all() and D.valid[i: j_end + 1].all()):
                continue
            next_allowed = i + win_n

            y = int(_has_pass_that_holds(D.ahead[i: j_end + 1] == a, win_n, hold_n))
            k, drs, ra = i - close_n, A.drs[i], A.lapno[i]
            out.append(dict(
                circuit=g.meta.get("circuit", "?"), year=g.meta.get("year", 0),
                round=g.meta.get("round", 0), attacker=a, defender=d, t0_s=float(g.t[i]),
                gap_s=float(gap_s[i]), gap_m=float(A.dist[i]),
                closing_speed_ms=float((A.dist[k] - A.dist[i]) / (close_n * dt)),
                att_speed_kmh=float(A.speed[i]), speed_diff_kmh=float(A.speed[i] - D.speed[i]),
                drs_eligible=np.nan if np.isnan(drs) else float(drs >= 8),
                drs_on=np.nan if np.isnan(drs) else float(drs >= 10),
                att_compound=float(A.compound[i]), def_compound=float(D.compound[i]),
                compound_diff=float(A.compound[i] - D.compound[i]),
                att_tyre_age=float(A.tyre_age[i]), def_tyre_age=float(D.tyre_age[i]),
                tyre_age_diff=float(A.tyre_age[i] - D.tyre_age[i]),
                lap_number=float(ra), laps_remaining=float(g.total_laps - ra),
                race_progress=float(ra / g.total_laps),
                track_avg_speed=float(g.meta.get("track_avg_speed", np.nan)),
                y=y,
            ))
    return out


def _lap_is_valid(lap) -> bool:
    ts = lap.get("TrackStatus")
    green = not pd.isna(ts) and set(str(ts)) <= {"1"}
    return bool(int(lap["LapNumber"]) > 1 and green
                and pd.isna(lap["PitInTime"]) and pd.isna(lap["PitOutTime"]))


def _screen_laps(laps: pd.DataFrame, screen_s: float):
    """(driver, lap) pairs worth loading: both cars of any pair within screen_s at a lap
    boundary, for that lap and the next. None disables screening."""
    if screen_s <= 0:
        return None
    L = laps[["DriverNumber", "LapNumber", "Time", "Position"]].copy()
    L["t"] = L["Time"].dt.total_seconds()
    keep = set()
    for n, grp in L.groupby("LapNumber"):
        n = int(n)
        for d in grp.loc[grp["t"].isna() | grp["Position"].isna(), "DriverNumber"]:
            keep.update({(d, n), (d, n + 1)})
        grp = grp.dropna(subset=["t", "Position"]).sort_values("Position")
        ts, nums = grp["t"].to_numpy(), grp["DriverNumber"].to_numpy()
        for k in range(1, len(grp)):
            if ts[k] - ts[k - 1] <= screen_s:
                for d in (nums[k], nums[k - 1]):
                    keep.update({(d, n), (d, n + 1)})
    return keep


def build_race_grid(session, cfg: DetectCfg, year: int, rnd: int, screen_s: float = 3.0,
                    max_laps_per_driver: int | None = None) -> RaceGrid:
    laps = session.laps
    drivers = [d for d in session.drivers if (laps["DriverNumber"] == d).any()]
    gt = np.arange(laps["LapStartTime"].dropna().min().total_seconds(),
                   laps["Time"].dropna().max().total_seconds(), cfg.dt)
    n = len(gt)
    total_laps = int(getattr(session, "total_laps", None) or laps["LapNumber"].max())
    pos_prev_map = {(r.DriverNumber, int(r.LapNumber)): r.Position
                    for r in laps.itertuples() if not pd.isna(r.Position)}
    keep = _screen_laps(laps, screen_s)

    def empty():
        return DriverArrays(ahead=np.full(n, "", dtype=object), valid=np.zeros(n, dtype=bool),
                            **{k: np.full(n, np.nan) for k in
                               ("dist", "speed", "drs", "lapno", "compound", "tyre_age", "pos_prev")})
    arrays = {drv: empty() for drv in drivers}

    n_done = n_fail = 0
    t_start = time.time()
    for drv in drivers:
        A, count = arrays[drv], 0
        for _, lap in laps[laps["DriverNumber"] == drv].iterlaps():
            ln = int(lap["LapNumber"])
            if not _lap_is_valid(lap) or pd.isna(lap["LapStartTime"]) or pd.isna(lap["Time"]):
                continue
            if keep is not None and (drv, ln) not in keep:
                continue
            if max_laps_per_driver is not None and count >= max_laps_per_driver:
                break
            try:
                cd = lap.get_car_data(pad=1, pad_side="both")
                if len(cd) < 6:
                    continue
                da = cd.iloc[1:-1].add_driver_ahead()
                ts = da["SessionTime"].dt.total_seconds().to_numpy()
                ahead = da["DriverAhead"]
                ahead = np.where(pd.isna(ahead), "", ahead.astype(str)).astype(object)
                vals = dict(dist=da["DistanceToDriverAhead"].to_numpy(float),
                            speed=da["Speed"].to_numpy(float), drs=da["DRS"].to_numpy(float))
            except Exception as e:  # noqa: BLE001
                n_fail += 1
                if n_fail <= 3:
                    print(f"    [warn] telemetry failed for driver {drv} lap {ln}: {type(e).__name__}: {e}")
                continue
            gi = np.flatnonzero((gt >= ts[0]) & (gt <= ts[-1]))
            if len(gi) == 0:
                continue
            si = np.searchsorted(ts, gt[gi], side="right") - 1
            A.ahead[gi] = ahead[si]
            for key, v in vals.items():
                getattr(A, key)[gi] = v[si]
            A.lapno[gi] = ln
            A.compound[gi] = COMPOUND_RANK.get(str(lap.get("Compound")).upper(), np.nan)
            A.tyre_age[gi] = lap.get("TyreLife", np.nan)
            A.pos_prev[gi] = pos_prev_map.get((drv, ln - 1), np.nan)
            A.valid[gi] = True
            count += 1
            n_done += 1
        print(f"    driver {drv}: {count} laps ({time.time() - t_start:.0f}s)")

    meta = dict(circuit=str(session.event["Location"]), event=str(session.event["EventName"]),
                year=int(year), round=int(rnd))
    try:
        meta["track_avg_speed"] = float(session.laps.pick_fastest().get_car_data()["Speed"].mean())
    except Exception:  # noqa: BLE001
        meta["track_avg_speed"] = np.nan
    print(f"    laps processed: {n_done}, telemetry failures: {n_fail}")
    return RaceGrid(t=gt, drv=arrays, total_laps=total_laps, meta=meta)


def cmd_build(args):
    import fastf1
    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)
    fastf1.Cache.enable_cache(args.cache_dir)
    cfg = DetectCfg()
    failed = []
    for year in args.years:
        if args.rounds == ["all"]:
            sched = fastf1.get_event_schedule(year, include_testing=False)
            rounds = [int(r) for r in sched["RoundNumber"] if int(r) > 0]
        else:
            rounds = [int(r) for r in args.rounds]
        for rnd in rounds:
            path = os.path.join(args.data_dir, f"events_{year}_{rnd:02d}.csv")
            if os.path.exists(path) and not args.force:
                print(f"[skip] {path} exists")
                continue
            print(f"[build] {year} round {rnd}")
            t0 = time.time()
            try:
                s = fastf1.get_session(year, rnd, "R")
                s.load(laps=True, telemetry=True, weather=False, messages=False)
                grid = build_race_grid(s, cfg, year, rnd, screen_s=args.screen_s,
                                       max_laps_per_driver=args.max_laps_per_driver)
                ev = pd.DataFrame(detect_attempts(grid, cfg))
            except Exception as e:  # noqa: BLE001
                msg = f"{year} round {rnd}: {type(e).__name__}: {str(e)[:200]}"
                print(f"[FAILED] {msg}")
                failed.append(msg)
                continue
            ev.to_csv(path, index=False)
            npos = int(ev["y"].sum()) if len(ev) else 0
            print(f"[build] {grid.meta['event']}: {len(ev)} attempts, {npos} successful "
                  f"-> {path} [{time.time() - t0:.0f}s]")
    if failed:
        print("\nFailed rounds:")
        for m in failed:
            print("   ", m)


def load_events(data_dir: str) -> pd.DataFrame:
    files = sorted(f for f in os.listdir(data_dir) if f.startswith("events_") and f.endswith(".csv"))
    if not files:
        sys.exit(f"No events_*.csv in {data_dir}. Run `build` first.")
    return pd.concat([pd.read_csv(os.path.join(data_dir, f)) for f in files], ignore_index=True)


def add_track_category(df: pd.DataFrame, train_circuits) -> pd.DataFrame:
    """High-speed vs technical: circuit average speed >= median over TRAINING circuits only."""
    if "track_high_speed" in df.columns:
        return df
    cs = df.groupby("circuit")["track_avg_speed"].mean()
    df = df.copy()
    if cs.notna().any():
        thr = cs[cs.index.isin(train_circuits)].median()
        df["track_high_speed"] = (df["circuit"].map(cs) >= thr).astype(float)
    else:
        df["track_high_speed"] = np.nan
    return df


def circuit_split(df: pd.DataFrame, test_frac=0.2, val_frac=0.2, seed=0) -> dict:
    circuits = sorted(df["circuit"].unique())
    if len(circuits) < 3:
        sys.exit(f"Need >= 3 circuits for a circuit-level split, found {len(circuits)}.")
    n_test = max(1, round(test_frac * len(circuits)))
    n_val = max(1, round(val_frac * (len(circuits) - n_test)))
    for attempt in range(200):
        perm = list(np.random.default_rng(seed + attempt).permutation(circuits))
        split = dict(train=perm[n_test + n_val:], val=perm[n_test:n_test + n_val], test=perm[:n_test])
        if all(df[df.circuit.isin(c)]["y"].nunique() == 2 for c in split.values()):
            return split
    sys.exit("No circuit split has both classes in every part; build more data.")


def random_split(df: pd.DataFrame, n_val: int, n_test: int, seed=0):
    idx = np.random.default_rng(seed).permutation(len(df))
    te, va, tr = idx[:n_test], idx[n_test:n_test + n_val], idx[n_test + n_val:]
    return df.iloc[tr], df.iloc[va], df.iloc[te]


def ece_score(y, p, bins=10) -> float:
    ids = np.clip(np.digitize(p, np.linspace(0, 1, bins + 1)[1:-1]), 0, bins - 1)
    return float(sum(m.mean() * abs(y[m].mean() - p[m].mean())
                     for m in (ids == b for b in range(bins)) if m.any()))


def best_f1_threshold(y, s) -> float:
    from sklearn.metrics import precision_recall_curve
    p, r, th = precision_recall_curve(y, s)
    f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
    return float(th[int(np.argmax(f1))]) if len(th) else 0.5


def evaluate(y, s, thr, is_prob=True) -> dict:
    from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score
    y = np.asarray(y)
    return dict(n=len(y), pos_rate=float(y.mean()),
                roc_auc=float(roc_auc_score(y, s)), pr_auc=float(average_precision_score(y, s)),
                f1=float(f1_score(y, (s >= thr).astype(int), zero_division=0)),
                ece=ece_score(y, s) if is_prob else np.nan,
                brier=float(brier_score_loss(y, s)) if is_prob else np.nan)


def bootstrap_ci(y, s, n_boot=200, seed=0):
    """Event-level bootstrap; ignores within-race correlation, so optimistic."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    rng = np.random.default_rng(seed)
    y, s = np.asarray(y), np.asarray(s)
    ro, pr = [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        ro.append(roc_auc_score(y[i], s[i]))
        pr.append(average_precision_score(y[i], s[i]))
    if len(ro) < 20:
        return (np.nan, np.nan), (np.nan, np.nan)
    return tuple(np.percentile(ro, [2.5, 97.5])), tuple(np.percentile(pr, [2.5, 97.5]))


def make_logreg():
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(max_iter=2000))


def make_xgb(params: dict, seed=0):
    from xgboost import XGBClassifier
    return XGBClassifier(objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                         random_state=seed, n_jobs=1, **params)


def val_pr_auc(params, Xtr, ytr, Xva, yva, seed=0) -> float:
    from sklearn.metrics import average_precision_score
    m = make_xgb(params, seed).fit(Xtr, ytr)
    return float(average_precision_score(yva, m.predict_proba(Xva)[:, 1]))


GRID = dict(learning_rate=[0.05, 0.1, 0.2], max_depth=[3, 5, 7],
            n_estimators=[100, 300], subsample=[0.8, 1.0])


def grid_search(Xtr, ytr, Xva, yva, seed=0):
    history, best, best_p = [], -np.inf, None
    for i, combo in enumerate(itertools.product(*GRID.values()), 1):
        params = dict(zip(GRID, combo))
        score = val_pr_auc(params, Xtr, ytr, Xva, yva, seed)
        if score > best:
            best, best_p = score, params
        history.append(dict(trial=i, value=score, best_so_far=best, params=params))
    return best_p, history


def optuna_search(Xtr, ytr, Xva, yva, n_trials=30, seed=0):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    history, best = [], [-np.inf]

    def objective(trial):
        params = dict(
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            max_depth=trial.suggest_int("max_depth", 2, 8),
            n_estimators=trial.suggest_int("n_estimators", 50, 500),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
        )
        score = val_pr_auc(params, Xtr, ytr, Xva, yva, seed)
        best[0] = max(best[0], score)
        history.append(dict(trial=trial.number + 1, value=score, best_so_far=best[0], params=params))
        return score

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, history


def fit_eval_all(tr, va, te, feats, params=None, n_trials=30, n_boot=200, seed=0):
    """Fit and score the four baselines. Given `params`, tuning is skipped and they are reused."""
    Xtr, ytr = tr[feats].to_numpy(float), tr["y"].to_numpy(int)
    Xva, yva = va[feats].to_numpy(float), va["y"].to_numpy(int)
    Xte, yte = te[feats].to_numpy(float), te["y"].to_numpy(int)
    rows, preds, chosen, histories = [], {}, {}, {}

    def add(name, s_va, s_te, is_prob=True):
        preds[name] = np.asarray(s_te, dtype=float)
        r = evaluate(yte, s_te, best_f1_threshold(yva, s_va), is_prob)
        (rlo, rhi), (plo, phi) = bootstrap_ci(yte, s_te, n_boot, seed) if n_boot else ((np.nan,) * 2,) * 2
        r.update(model=name, roc_ci=f"[{rlo:.2f},{rhi:.2f}]", pr_ci=f"[{plo:.2f},{phi:.2f}]")
        rows.append(r)

    add("Gap heuristic", -va["gap_s"].to_numpy(float), -te["gap_s"].to_numpy(float), is_prob=False)
    lr = make_logreg().fit(Xtr, ytr)
    add("Logistic regression", lr.predict_proba(Xva)[:, 1], lr.predict_proba(Xte)[:, 1])

    searches = dict(grid=lambda: grid_search(Xtr, ytr, Xva, yva, seed),
                    optuna=lambda: optuna_search(Xtr, ytr, Xva, yva, n_trials, seed))
    for key, name in (("grid", "XGBoost (grid)"), ("optuna", "XGBoost (Optuna)")):
        if params is None:
            print(f"  tuning {name} ...")
            chosen[key], histories[key] = searches[key]()
        else:
            chosen[key] = params[key]
        m = make_xgb(chosen[key], seed).fit(Xtr, ytr)
        add(name, m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1])

    table = pd.DataFrame(rows)[["model", "n", "pos_rate", "roc_auc", "roc_ci", "pr_auc", "pr_ci",
                                "f1", "ece", "brier"]]
    return table, chosen, histories, preds


def make_synthetic(n_circuits=10, per_circuit=120, seed=0) -> pd.DataFrame:
    """Fake data for pipeline smoke tests only."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_circuits):
        high = float(c % 2)
        circ_effect = rng.normal(0, 0.6)
        for _ in range(per_circuit):
            gap_s = rng.uniform(0.1, 1.0)
            closing = rng.normal(2.0, 1.5)
            drs_on = float(rng.random() < 0.45)
            tyre_diff = rng.normal(0, 6)
            lap = rng.integers(3, 55)
            logit = (-0.4 + 2.5 * (0.6 - gap_s) + 0.35 * closing + 1.0 * drs_on
                     + 0.05 * tyre_diff + 0.8 * high * drs_on + circ_effect + rng.normal(0, 0.8))
            y = int(rng.random() < 1 / (1 + np.exp(-logit)))
            speed = rng.normal(260, 25)
            rows.append(dict(
                circuit=f"SYN-{c:02d}", year=2023, round=c + 1, attacker="1", defender="2", t0_s=0.0,
                gap_s=gap_s, gap_m=gap_s * speed / 3.6, closing_speed_ms=closing,
                att_speed_kmh=speed, speed_diff_kmh=rng.normal(5, 8),
                drs_eligible=drs_on, drs_on=drs_on,
                att_compound=float(rng.integers(0, 3)), def_compound=float(rng.integers(0, 3)),
                compound_diff=float(rng.integers(-2, 3)),
                att_tyre_age=float(rng.integers(1, 30)), def_tyre_age=float(rng.integers(1, 30)),
                tyre_age_diff=tyre_diff, lap_number=float(lap), laps_remaining=float(57 - lap),
                race_progress=lap / 57, track_avg_speed=np.nan, track_high_speed=high, y=y))
    return pd.DataFrame(rows)


def cmd_run(args):
    if args.synthetic:
        print("SYNTHETIC DATA: pipeline smoke test only, not real results.\n")
        df, source = make_synthetic(seed=args.seed), "synthetic"
    else:
        df, source = load_events(args.data_dir), f"FastF1 ({args.data_dir}/)"

    df = df.dropna(subset=["y"]).reset_index(drop=True)
    split = circuit_split(df, args.test_frac, args.val_frac, args.seed)
    df = add_track_category(df, split["train"])
    feats = [f for f in FEATURES if f in df.columns and df[f].notna().any()]
    tr, va, te = (df[df.circuit.isin(split[k])] for k in ("train", "val", "test"))

    races = df[["year", "round"]].drop_duplicates().shape[0]
    print(f"Data: {source}, {races} races, {df['circuit'].nunique()} circuits, "
          f"{len(df)} attempts, pass rate {df['y'].mean():.1%}")
    for name, part in (("train", tr), ("val", va), ("test", te)):
        print(f"  {name:5s}: {len(part):5d} events, pass rate {part['y'].mean():.1%}, "
              f"circuits = {', '.join(map(str, split[name]))}")

    fmt = dict(index=False, float_format=lambda v: f"{v:.3f}")
    print("\nCircuit-level split (unseen circuits)")
    table, chosen, hist, preds = fit_eval_all(tr, va, te, feats, n_trials=args.optuna_trials,
                                              n_boot=args.n_boot, seed=args.seed)
    print(table.to_string(**fmt))
    for k, v in chosen.items():
        print(f"  {k}: {json.dumps({a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items()})}")

    print("\nRandom event split, same sizes (reference)")
    rtr, rva, rte = random_split(df, len(va), len(te), args.seed)
    rtable, _, _, _ = fit_eval_all(rtr, rva, rte, feats, params=chosen, n_boot=0, seed=args.seed)
    print(rtable.drop(columns=["roc_ci", "pr_ci"]).to_string(**fmt))

    print("\nPR-AUC gap (random - circuit)")
    for m in table["model"]:
        gap = rtable.loc[rtable.model == m, "pr_auc"].iloc[0] - table.loc[table.model == m, "pr_auc"].iloc[0]
        print(f"  {m:22s} {gap:+.3f}")

    os.makedirs(args.out, exist_ok=True)
    pdf = pd.DataFrame({"circuit": te["circuit"].to_numpy(), "y": te["y"].to_numpy(int), **preds})
    pdf.to_csv(os.path.join(args.out, "preds_circuit_split.csv"), index=False)
    table.to_csv(os.path.join(args.out, "metrics_circuit_split.csv"), index=False)
    rtable.to_csv(os.path.join(args.out, "metrics_random_split.csv"), index=False)
    with open(os.path.join(args.out, "search_histories.json"), "w") as f:
        json.dump(dict(chosen=chosen, histories=hist), f, indent=2, default=float)
    with open(os.path.join(args.out, "run_config.json"), "w") as f:
        json.dump(dict(source=source, seed=args.seed, features=feats, splits=split,
                       detect_cfg=DetectCfg().__dict__, optuna_trials=args.optuna_trials), f, indent=2)
    print(f"\nSaved to {args.out}/")


def _toy_grid(gap_fn, defender_ahead_fn=None, att_ahead_fn=None, def_valid_fn=None,
              pos_att=2.0, pos_def=1.0, T=300.0, cfg=DetectCfg()):
    """Defender '1' and attacker '2' at a constant 250 km/h; gap_fn(t) gives the gap in seconds."""
    t = np.arange(0, T, cfg.dt)
    n, spd = len(t), 250.0

    def base():
        return DriverArrays(ahead=np.full(n, "", dtype=object), dist=np.full(n, np.nan),
                            speed=np.full(n, spd), drs=np.zeros(n), lapno=np.full(n, 5.0),
                            compound=np.full(n, 1.0), tyre_age=np.full(n, 10.0),
                            pos_prev=np.full(n, np.nan), valid=np.ones(n, dtype=bool))
    A, D = base(), base()
    A.pos_prev[:], D.pos_prev[:] = pos_att, pos_def
    A.ahead[:] = "1"
    A.dist[:] = np.array([gap_fn(x) for x in t], dtype=float) * spd / 3.6
    for i, x in enumerate(t):
        if att_ahead_fn is not None:
            A.ahead[i] = att_ahead_fn(x)
            if A.ahead[i] == "":
                A.dist[i] = np.nan
        if defender_ahead_fn is not None:
            D.ahead[i] = defender_ahead_fn(x)
    if def_valid_fn is not None:
        D.valid = np.array([def_valid_fn(x) for x in t])
    return RaceGrid(t=t, drv={"1": D, "2": A}, total_laps=57, meta=dict(circuit="TOY", year=2023, round=1))


def _approach(t, t0=40.0, t1=60.0, g0=3.0, g1=0.9):
    if t <= t0:
        return g0
    if t >= t1:
        return g1
    return g0 + (g1 - g0) * (t - t0) / (t1 - t0)


def cmd_selftest(_args):
    cfg = DetectCfg()
    results = []

    def check(name, cond):
        results.append(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    def clean_pass():
        return _toy_grid(lambda t: _approach(t) if t < 72 else 0.3,
                         defender_ahead_fn=lambda t: "2" if t >= 72 else "",
                         att_ahead_fn=lambda t: "1" if t < 72 else "")

    def drop_back():
        return _toy_grid(lambda t: _approach(t) if t < 62 else min(0.9 + 0.1 * (t - 62), 2.5))

    ev = detect_attempts(clean_pass(), cfg)
    check("clean pass -> 1 attempt, success", len(ev) == 1 and ev[0]["y"] == 1)

    ev = detect_attempts(drop_back(), cfg)
    check("approach then drop back -> 1 attempt, failure", len(ev) == 1 and ev[0]["y"] == 0)

    g = _toy_grid(lambda t: _approach(t) if t < 72 else 0.5,
                  defender_ahead_fn=lambda t: "2" if 72 <= t < 75 else "",
                  att_ahead_fn=lambda t: "" if 72 <= t < 75 else "1")
    ev = detect_attempts(g, cfg)
    check("pass re-passed within 3 s -> failure", len(ev) >= 1 and ev[0]["t0_s"] < 65 and ev[0]["y"] == 0)

    g = _toy_grid(_approach, def_valid_fn=lambda t: t < 75)
    check("defender pits in window -> discarded", len(detect_attempts(g, cfg)) == 0)

    g = _toy_grid(_approach, pos_att=1.0, pos_def=5.0)
    check("lapped defender -> excluded", len(detect_attempts(g, cfg)) == 0)

    ev = detect_attempts(_toy_grid(lambda t: 0.8), cfg)
    t0s = [e["t0_s"] for e in ev]
    check("DRS train -> non-overlapping windows, all failures",
          len(ev) >= 2 and all(b - a >= cfg.outcome_window_s for a, b in zip(t0s, t0s[1:]))
          and all(e["y"] == 0 for e in ev))

    e1, e0 = detect_attempts(clean_pass(), cfg)[0], detect_attempts(drop_back(), cfg)[0]
    diff = [k for k in e1 if k != "y" and not (pd.isna(e1[k]) and pd.isna(e0[k])) and e1[k] != e0[k]]
    check("features identical when only the outcome differs (no leakage)", e1["y"] != e0["y"] and not diff)
    if diff:
        print(f"         differing fields: {diff}")

    print(f"\n{sum(results)}/{len(results)} passed")
    sys.exit(0 if all(results) else 1)


SURFACE, INK, INK2, GRID_C, AXIS_C = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#c9c8c3"
MODEL_STYLE = {
    "Gap heuristic":       dict(color="#8a8984", ls="--", marker=None),
    "Logistic regression": dict(color="#2a78d6", ls="-", marker="o"),
    "XGBoost (grid)":      dict(color="#eb6834", ls="-", marker="s"),
    "XGBoost (Optuna)":    dict(color="#1baf7a", ls="-", marker="^"),
}
SPLIT_COLORS = {"circuit": "#2a78d6", "random": "#eb6834"}


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for _, r in df.iterrows()]
    return "\n".join(lines) + "\n"


def _style_ax(ax, title, grid_axis="y"):
    ax.set_facecolor(SURFACE)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(AXIS_C)
    ax.tick_params(colors=INK2, labelsize=8, length=3)
    ax.grid(axis=grid_axis, color=GRID_C, linewidth=0.8)
    ax.set_axisbelow(True)
    for lab in (ax.xaxis.label, ax.yaxis.label):
        lab.set_color(INK2)
        lab.set_size(9)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, fontweight="bold", pad=8)


def _legend(ax, y, ncol=2, fontsize=8):
    ax.legend(frameon=False, fontsize=fontsize, loc="upper center", bbox_to_anchor=(0.5, y),
              ncol=ncol, labelcolor=INK2)


def _save(fig, path):
    import matplotlib.pyplot as plt
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def _fmt(v: pd.Series, sign=""):
    if v.isna().all():
        return "n/a"
    return f"{v.mean():{sign}.3f} ± {v.std(ddof=1):.3f}" if len(v) > 1 else f"{v.mean():{sign}.3f}"


def cmd_report(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

    os.makedirs(args.out, exist_ok=True)
    dirs = sorted({d for p in args.results for d in glob.glob(p)
                   if os.path.isfile(os.path.join(d, "metrics_circuit_split.csv"))})
    if not dirs:
        sys.exit("No results directories with metrics_circuit_split.csv. Run `run` first.")
    print(f"[report] {len(dirs)} result dir(s): {', '.join(dirs)}")
    plt.rcParams.update({"font.family": "sans-serif", "figure.facecolor": SURFACE})
    out_path = lambda name: os.path.join(args.out, name)  # noqa: E731

    # Dataset table and per-circuit success rate
    df = load_events(args.data_dir).dropna(subset=["y"])
    df["race"] = df["year"].astype(int) * 100 + df["round"].astype(int)
    rate = df["y"].mean()
    g = df.groupby("circuit").agg(races=("race", "nunique"), attempts=("y", "size"), successes=("y", "sum"))
    g["success_rate"] = g["successes"] / g["attempts"]
    g = g.sort_values("success_rate", ascending=False)
    tot = pd.DataFrame([dict(races=int(df["race"].nunique()), attempts=len(df),
                             successes=int(df["y"].sum()), success_rate=float(rate))], index=["ALL"])
    dtab = pd.concat([g, tot])
    dtab.index.name = "circuit"
    dtab.to_csv(out_path("table_dataset.csv"))
    out = dtab.reset_index()
    out["success_rate"] = (out["success_rate"] * 100).round(1).astype(str) + "%"
    for c in ("races", "attempts", "successes"):
        out[c] = out[c].astype(int)
    with open(out_path("table_dataset.md"), "w") as f:
        f.write(_md_table(out))
    print(f"  wrote table_dataset.csv/.md ({int(tot.races.iloc[0])} races, {len(g)} circuits, "
          f"{len(df)} attempts, {int(df['y'].sum())} successes, {rate * 100:.1f}%)")

    fig, ax = plt.subplots(figsize=(6.4, 0.26 * len(g) + 1.1))
    gs = g.sort_values("success_rate")
    ax.barh(gs.index, gs["success_rate"] * 100, height=0.6, color=SPLIT_COLORS["circuit"])
    ax.axvline(rate * 100, color=INK2, lw=1.2, ls="--", zorder=1)
    ax.set_ylim(-0.7, len(gs) + 0.2)
    ax.text(rate * 100 + 0.3, len(gs) - 0.3, f"overall {rate * 100:.1f}%", color=INK2, fontsize=8, va="bottom")
    for i, r in enumerate(gs.itertuples()):
        ax.text(r.success_rate * 100 + 0.25, i, f"{r.success_rate * 100:.1f}%  ({int(r.successes)}/{int(r.attempts)})",
                va="center", fontsize=7.5, color=INK2, bbox=dict(fc=SURFACE, ec="none", pad=0.6), zorder=3)
    ax.set_xlabel("Overtake-attempt success rate (%)")
    ax.set_xlim(0, gs["success_rate"].max() * 100 * 1.35)
    ax.tick_params(axis="y", labelsize=8)
    _style_ax(ax, "Attempt success rate by circuit", "x")
    ax.grid(axis="y", visible=False)
    _save(fig, out_path("fig_class_balance.png"))

    # Metrics aggregated over result dirs (one per circuit split seed)
    rows_c, rows_r = [], []
    for d in dirs:
        seed = json.load(open(os.path.join(d, "run_config.json")))["seed"]
        rows_c.append(pd.read_csv(os.path.join(d, "metrics_circuit_split.csv")).assign(seed=seed))
        rp = os.path.join(d, "metrics_random_split.csv")
        if os.path.isfile(rp):
            rows_r.append(pd.read_csv(rp).assign(seed=seed))
    C = pd.concat(rows_c)
    R = pd.concat(rows_r) if rows_r else None
    order = [m for m in MODEL_STYLE if m in set(C["model"])]
    num, txt = [], []
    for m in order:
        cm = C[C.model == m]
        rn, rt = dict(model=m, n_splits=len(cm)), dict(model=m)
        for k in ("roc_auc", "pr_auc", "f1", "ece", "brier"):
            v = cm[k].astype(float)
            rn[k + "_mean"], rn[k + "_sd"] = v.mean(), v.std(ddof=1) if len(v) > 1 else np.nan
            rt[k] = _fmt(v)
        if R is not None:
            rm = R[R.model == m]
            gap = rm.set_index("seed")["pr_auc"] - cm.set_index("seed")["pr_auc"]
            # Subtracting logistic regression's gap isolates the extra gap attributable to memorization.
            lr_c = C[C.model == "Logistic regression"].set_index("seed")["pr_auc"]
            lr_r = R[R.model == "Logistic regression"].set_index("seed")["pr_auc"]
            ex = gap - (lr_r - lr_c)
            rn.update(pr_auc_random_mean=rm["pr_auc"].mean(), pr_auc_random_sd=rm["pr_auc"].std(ddof=1),
                      pr_gap_mean=gap.mean(), pr_gap_sd=gap.std(ddof=1),
                      pr_gap_vs_lr_mean=ex.mean(), pr_gap_vs_lr_sd=ex.std(ddof=1))
            rt["PR-AUC random split"] = f"{rm['pr_auc'].mean():.3f}"
            rt["PR-AUC gap (random - circuit)"] = _fmt(gap, "+")
            rt["Extra gap vs logistic regression"] = _fmt(ex, "+")
        num.append(rn)
        txt.append(rt)
    pd.DataFrame(num).to_csv(out_path("table_results.csv"), index=False)
    T = pd.DataFrame(txt).rename(columns=dict(roc_auc="ROC-AUC", pr_auc="PR-AUC", f1="F1", ece="ECE", brier="Brier"))
    with open(out_path("table_results.md"), "w") as f:
        f.write(f"Circuit-level held-out test, mean ± sd over {len(dirs)} circuit splits.\n\n" + _md_table(T))
    print(f"  wrote table_results.csv/.md\n\n{T.to_string(index=False)}\n")

    # Circuit vs random split PR-AUC
    if R is not None:
        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        x, w, tops = np.arange(len(order)), 0.36, []
        for j, (key, lab, df_) in enumerate((("circuit", "Circuit-level split (unseen circuits)", C),
                                              ("random", "Random event split (reference)", R))):
            mu = [df_[df_.model == m]["pr_auc"].mean() for m in order]
            sd = [df_[df_.model == m]["pr_auc"].std(ddof=1) if (df_.model == m).sum() > 1 else 0 for m in order]
            xs = x + (j - 0.5) * (w + 0.02)
            ax.bar(xs, mu, width=w, color=SPLIT_COLORS[key], label=lab)
            ax.errorbar(xs, mu, yerr=sd, fmt="none", ecolor=INK2, elinewidth=1, capsize=2)
            for xi, v, e in zip(xs, mu, sd):
                ax.text(xi, v + (e if np.isfinite(e) else 0) + 0.012, f"{v:.2f}", ha="center", fontsize=8, color=INK2)
            tops.append(max(v + (e if np.isfinite(e) else 0) for v, e in zip(mu, sd)))
        ax.set_xticks(x)
        ax.set_xticklabels(order, fontsize=8)
        ax.axhline(float(C["pos_rate"].mean()), color=INK2, lw=1, ls=":", label="Chance (positive rate)")
        ax.set_ylabel("PR-AUC")
        _style_ax(ax, "PR-AUC: unseen circuits vs random event split")
        ax.set_ylim(0, max(tops) * 1.18)
        _legend(ax, -0.12, ncol=1)
        _save(fig, out_path("fig_split_comparison.png"))

    # ROC, PR and calibration from saved test predictions
    pdir = args.curve_dir or next((d for d in dirs if os.path.isfile(os.path.join(d, "preds_circuit_split.csv"))), None)
    if pdir is None:
        print("  [skip] no preds_circuit_split.csv found for ROC/PR/calibration")
    else:
        P = pd.read_csv(os.path.join(pdir, "preds_circuit_split.csv"))
        y = P["y"].to_numpy(int)
        seed = json.load(open(os.path.join(pdir, "run_config.json")))["seed"]
        fig, axs = plt.subplots(1, 2, figsize=(9.0, 3.8))
        for m in order:
            st, sc = MODEL_STYLE[m], P[m].to_numpy(float)
            pr, rc, _ = precision_recall_curve(y, sc)
            fpr, tpr, _ = roc_curve(y, sc)
            axs[0].plot(rc, pr, color=st["color"], ls=st["ls"], lw=2, label=f"{m} (AP {average_precision_score(y, sc):.2f})")
            axs[1].plot(fpr, tpr, color=st["color"], ls=st["ls"], lw=2, label=f"{m} (AUC {roc_auc_score(y, sc):.2f})")
        axs[0].axhline(y.mean(), color=INK2, lw=1, ls=":")
        axs[0].text(0.02, y.mean() + 0.012, "chance", ha="left", fontsize=7.5, color=INK2)
        axs[1].plot([0, 1], [0, 1], color=INK2, lw=1, ls=":")
        axs[0].set_xlabel("Recall"); axs[0].set_ylabel("Precision")
        axs[1].set_xlabel("False-positive rate"); axs[1].set_ylabel("True-positive rate")
        for a, t in zip(axs, (f"Precision-recall, unseen circuits (split seed {seed})", "ROC, unseen circuits")):
            _style_ax(a, t, "both")
            a.set_xlim(0, 1); a.set_ylim(0, 1.02)
            _legend(a, -0.17, fontsize=7.5)
        fig.tight_layout()
        _save(fig, out_path("fig_roc_pr.png"))

        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        hi = 0.0
        for m in (m for m in order if m != "Gap heuristic"):
            st = MODEL_STYLE[m]
            fo, mp = calibration_curve(y, P[m].to_numpy(float), n_bins=8, strategy="quantile")
            hi = max(hi, fo.max(), mp.max())
            ax.plot(mp, fo, color=st["color"], lw=2, marker=st["marker"], ms=7, mec=SURFACE, mew=1.5, label=m)
        hi = min(1.0, hi * 1.1 + 0.02)
        ax.plot([0, hi], [0, hi], color=INK2, lw=1, ls=":", label="Perfect calibration")
        ax.set_xlim(0, hi); ax.set_ylim(0, hi)
        ax.set_xlabel("Mean predicted probability (8 quantile bins)"); ax.set_ylabel("Observed success rate")
        _style_ax(ax, "Calibration on unseen circuits", "both")
        _legend(ax, -0.16)
        _save(fig, out_path("fig_calibration.png"))

    # Tuning progress, relative to the best validation PR-AUC either method found per split
    H = {"grid": [], "optuna": []}
    for d in dirs:
        hp = os.path.join(d, "search_histories.json")
        if os.path.isfile(hp):
            hist = json.load(open(hp))["histories"]
            if "grid" in hist and "optuna" in hist:
                g_, o_ = ([t["best_so_far"] for t in hist[k]] for k in ("grid", "optuna"))
                ref = max(g_[-1], o_[-1])
                H["grid"].append([v - ref for v in g_])
                H["optuna"].append([v - ref for v in o_])
    if all(H.values()):
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        for k, lab, m in (("grid", "Grid search (fixed order)", "XGBoost (grid)"),
                          ("optuna", "Optuna (TPE)", "XGBoost (Optuna)")):
            L = min(len(h) for h in H[k])
            A = np.array([h[:L] for h in H[k]])
            xs, col = np.arange(1, L + 1), MODEL_STYLE[m]["color"]
            if len(A) > 1:
                ax.fill_between(xs, A.min(0), A.max(0), color=col, alpha=0.15, lw=0)
            ax.plot(xs, A.mean(0), color=col, lw=2, label=f"{lab}  [{L} configs]")
        ax.axhline(0, color=INK2, lw=1, ls=":")
        ax.set_xlabel("Configurations evaluated")
        ax.set_ylabel("Best validation PR-AUC so far,\nrelative to best found (0 = best)")
        _style_ax(ax, f"Tuning progress (mean and range over {len(dirs)} splits)", "both")
        _legend(ax, -0.18)
        _save(fig, out_path("fig_search_history.png"))
    print(f"\n[report] done -> {args.out}/")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("selftest", help="unit tests for event detection").set_defaults(fn=cmd_selftest)

    b = sub.add_parser("build", help="download races and detect overtake attempts")
    b.add_argument("--years", type=int, nargs="+", default=[2023])
    b.add_argument("--rounds", nargs="+", default=["1"], help="round numbers, or 'all'")
    b.add_argument("--cache-dir", default="ff1cache")
    b.add_argument("--data-dir", default="data")
    b.add_argument("--screen-s", type=float, default=0.0,
                   help="only load laps where a car is within this many seconds of the car ahead; 0 disables")
    b.add_argument("--max-laps-per-driver", type=int, default=None)
    b.add_argument("--force", action="store_true")
    b.set_defaults(fn=cmd_build)

    r = sub.add_parser("run", help="train and evaluate baselines")
    r.add_argument("--data-dir", default="data")
    r.add_argument("--out", default="results")
    r.add_argument("--synthetic", action="store_true", help="fake data, smoke test only")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--test-frac", type=float, default=0.2)
    r.add_argument("--val-frac", type=float, default=0.2)
    r.add_argument("--optuna-trials", type=int, default=30)
    r.add_argument("--n-boot", type=int, default=200)
    r.set_defaults(fn=cmd_run)

    rp = sub.add_parser("report", help="tables and figures from saved `run` outputs")
    rp.add_argument("--results", nargs="+", default=["results", "results_seed*"])
    rp.add_argument("--data-dir", default="data")
    rp.add_argument("--out", default="report")
    rp.add_argument("--curve-dir", default=None)
    rp.set_defaults(fn=cmd_report)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
