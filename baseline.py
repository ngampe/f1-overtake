#!/usr/bin/env python3
"""
baseline.py -- Baseline experiment for
"Telemetry-Based Detection and Prediction of Formula 1 Overtaking Events"
(IDAI.780.01 capstone, Checkpoint 1 proposal, Sec. 2.4 and Sec. 3).

What this implements (and where it comes from in the proposal)
--------------------------------------------------------------
Sec. 2.1  Pre-T0 features only: gap, closing speed, tire compound/age of BOTH
          attacker and defender, DRS availability, track category, race context.
Sec. 2.4  Baselines:
            (1) gap-distance heuristic
            (2) logistic regression on the full feature set
            (3) gradient-boosted trees (XGBoost) tuned with grid search AND
                Bayesian optimization (Optuna), validation PR-AUC as objective.
          The ReAct tuning agent is NOT here -- it is the Ckpt 3 contribution.
          This script saves the grid/Optuna search histories so the agent can
          later be compared on the same fold and budget.
Sec. 3    Circuit-level train/validation/test split (~20% of circuits held out
          as test, untouched during tuning), random split as a secondary
          reference, metrics: ROC-AUC, PR-AUC, F1, calibration error.

Usage
-----
  python baseline.py selftest                       # unit tests for event detection (no data needed)
  python baseline.py build --years 2023 --rounds 1 2 3 4 5 6 7 8   # download + detect events -> data/
  python baseline.py run                            # train baselines + evaluate -> results/
  python baseline.py run --synthetic                # pipeline smoke test on FAKE data (never report these numbers)

Operational definition of an overtake attempt (the part to tune on real races)
------------------------------------------------------------------------------
  Grid:      every driver's telemetry is resampled onto a common 0.5 s time grid.
  Valid:     green-flag laps only, not lap 1, not an in/out lap.
  T0:        first sample where the car behind is within ENTER_GAP_S (time gap =
             distance-to-car-ahead / own speed) after having been OUT of range
             for COOLDOWN_S continuously (fresh entry, not a long DRS train).
  Excluded:  defender is a lapped car (attacker is classified ahead of it).
  Success:   the attacker becomes the defender's DriverAhead within
             OUTCOME_WINDOW_S and stays there for HOLD_S.
  Failure:   no such sustained pass (includes "passed, then immediately re-passed").
  Discarded: either car has an invalid sample (pit, yellow/SC/VSC, ...) anywhere
             in the outcome window -- those attempts get no label at all.
  Leakage:   every feature is computed from samples at or before T0.
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
    cooldown_s: float = 5.0       # must be out of range this long before a new attempt
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
# FastF1 loading (thin glue; the part that needs real data to validate)
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
            s = fastf1.get_session(year, rnd, "R")
            s.load(laps=True, telemetry=True, weather=False, messages=False)
            grid = build_race_grid(s, cfg, year, rnd, screen_s=args.screen_s,
                                   max_laps_per_driver=args.max_laps_per_driver)
            ev = pd.DataFrame(detect_attempts(grid, cfg))
            ev.to_csv(path, index=False)
            npos = int(ev["y"].sum()) if len(ev) else 0
            print(f"[build] {grid.meta['event']} ({grid.meta['circuit']}): {len(ev)} attempts, "
                  f"{npos} successful -> {path}  [{time.time() - t0:.0f}s]")


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


def val_pr_auc(params, Xtr, ytr, Xva, yva, seed=0) -> float:
    from sklearn.metrics import average_precision_score
    m = make_xgb(params, seed).fit(Xtr, ytr)
    return float(average_precision_score(yva, m.predict_proba(Xva)[:, 1]))


GRID = dict(learning_rate=[0.05, 0.1, 0.2], max_depth=[3, 5, 7],
            n_estimators=[100, 300], subsample=[0.8, 1.0])


def grid_search(Xtr, ytr, Xva, yva, seed=0):
    keys = list(GRID)
    history, best, best_p = [], -np.inf, None
    for i, combo in enumerate(itertools.product(*GRID.values()), 1):
        params = dict(zip(keys, combo))
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
            reg_lambda=trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),   # L2
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),      # L1
        )
        score = val_pr_auc(params, Xtr, ytr, Xva, yva, seed)
        best[0] = max(best[0], score)
        history.append(dict(trial=trial.number + 1, value=score, best_so_far=best[0], params=params))
        return score

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, history


def fit_eval_all(tr, va, te, feats, grid_params=None, optuna_params=None, n_trials=30,
                 n_boot=200, seed=0, log=print):
    """Fit and evaluate the four baselines. If params are given, skip tuning (used for the
    random-split reference so that only the *split* differs between the two tables)."""
    Xtr, ytr = tr[feats].to_numpy(float), tr["y"].to_numpy(int)
    Xva, yva = va[feats].to_numpy(float), va["y"].to_numpy(int)
    Xte, yte = te[feats].to_numpy(float), te["y"].to_numpy(int)
    rows, histories, chosen = [], {}, {}

    def add(name, s_va, s_te, is_prob=True):
        thr = best_f1_threshold(yva, s_va)             # threshold picked on validation, never on test
        r = evaluate(yte, s_te, thr, is_prob)
        (rlo, rhi), (plo, phi) = bootstrap_ci(yte, s_te, n_boot, seed) if n_boot else ((np.nan,) * 2, (np.nan,) * 2)
        r.update(model=name, roc_ci=f"[{rlo:.2f},{rhi:.2f}]", pr_ci=f"[{plo:.2f},{phi:.2f}]")
        rows.append(r)

    # (1) gap-distance heuristic: smaller gap => higher predicted pass chance
    log("  [1/4] gap-distance heuristic")
    add("Gap heuristic", -va["gap_s"].to_numpy(float), -te["gap_s"].to_numpy(float), is_prob=False)

    # (2) logistic regression
    log("  [2/4] logistic regression")
    lr = make_logreg().fit(Xtr, ytr)
    add("Logistic regression", lr.predict_proba(Xva)[:, 1], lr.predict_proba(Xte)[:, 1])

    # (3) XGBoost + grid search
    if grid_params is None:
        log(f"  [3/4] XGBoost + grid search ({np.prod([len(v) for v in GRID.values()])} configs)")
        grid_params, histories["grid"] = grid_search(Xtr, ytr, Xva, yva, seed)
    else:
        log("  [3/4] XGBoost + grid search (reusing tuned params)")
    m = make_xgb(grid_params, seed).fit(Xtr, ytr)
    add("XGBoost (grid)", m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1])
    chosen["grid"] = grid_params

    # (4) XGBoost + Optuna (Bayesian optimization)
    if optuna_params is None:
        log(f"  [4/4] XGBoost + Optuna ({n_trials} trials)")
        optuna_params, histories["optuna"] = optuna_search(Xtr, ytr, Xva, yva, n_trials, seed)
    else:
        log("  [4/4] XGBoost + Optuna (reusing tuned params)")
    m = make_xgb(optuna_params, seed).fit(Xtr, ytr)
    add("XGBoost (Optuna)", m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1])
    chosen["optuna"] = optuna_params

    table = pd.DataFrame(rows)[["model", "n", "pos_rate", "roc_auc", "roc_ci", "pr_auc", "pr_ci",
                                "f1", "ece", "brier"]]
    return table, chosen, histories


# =============================================================================
# Synthetic data (pipeline smoke test ONLY)
# =============================================================================


def make_synthetic(n_circuits=10, per_circuit=120, seed=0) -> pd.DataFrame:
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


# =============================================================================
# `run` command
# =============================================================================


def summarize_dataset(df, splits_info, tr, va, te, source):
    print("=" * 78)
    print("DATASET")
    print("=" * 78)
    print(f"  source              : {source}")
    seasons = sorted(df['year'].unique().tolist())
    print(f"  seasons / races     : {seasons} / {df[['year', 'round']].drop_duplicates().shape[0]} races")
    print(f"  circuits            : {df['circuit'].nunique()}")
    print(f"  attempt events      : {len(df)}   (successful: {int(df['y'].sum())}, "
          f"failed: {int((1 - df['y']).sum())}, pass rate: {df['y'].mean():.1%})")
    print(f"  features ({len(FEATURES)})       : {', '.join(FEATURES)}")
    print("  circuit-level split :")
    for name, part, key in (("train", tr, "train"), ("val  ", va, "val"), ("test ", te, "test")):
        print(f"    {name}: {len(part):5d} events, pass rate {part['y'].mean():.1%}, "
              f"circuits = {', '.join(map(str, splits_info[key]))}")


def cmd_run(args):
    if args.synthetic:
        print("#" * 78)
        print("#  SYNTHETIC DATA -- pipeline smoke test only. NOT real F1 data.         #")
        print("#  Do not report or screenshot these numbers as results.                 #")
        print("#" * 78)
        df, source = make_synthetic(seed=args.seed), "SYNTHETIC (smoke test)"
    else:
        df, source = load_events(args.data_dir), f"FastF1 (events from {args.data_dir}/)"

    df = df.dropna(subset=["y"]).reset_index(drop=True)
    feats = [f for f in FEATURES if f in df.columns and df[f].notna().any()]
    tr, va, te, info = circuit_split(df, args.test_frac, args.val_frac, args.seed)
    summarize_dataset(df, info, tr, va, te, source)

    os.makedirs(args.out, exist_ok=True)
    print("\n" + "=" * 78)
    print("PRIMARY: circuit-level held-out evaluation (generalization to unseen circuits)")
    print("=" * 78)
    t0 = time.time()
    table, chosen, hist = fit_eval_all(tr, va, te, feats, n_trials=args.optuna_trials,
                                       n_boot=args.n_boot, seed=args.seed)
    print(f"\n  (tuning + fitting took {time.time() - t0:.0f}s)\n")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n  Chosen hyperparameters:")
    for k, v in chosen.items():
        print(f"    {k:7s}: {json.dumps({a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items()})}")

    print("\n" + "=" * 78)
    print("SECONDARY: random event-level split (reference only -- shows memorization effect)")
    print("=" * 78)
    rtr, rva, rte = random_split(df, args.seed)
    rtable, _, _ = fit_eval_all(rtr, rva, rte, feats, grid_params=chosen["grid"],
                                optuna_params=chosen["optuna"], n_boot=0, seed=args.seed)
    print()
    print(rtable.drop(columns=["roc_ci", "pr_ci"]).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\n  PR-AUC gap (random - circuit): large positive gap => model leans on circuit/driver identity")
    for m in table["model"]:
        a = float(rtable.loc[rtable.model == m, "pr_auc"].iloc[0])
        b = float(table.loc[table.model == m, "pr_auc"].iloc[0])
        print(f"    {m:22s} {a - b:+.3f}")

    table.to_csv(os.path.join(args.out, "metrics_circuit_split.csv"), index=False)
    rtable.to_csv(os.path.join(args.out, "metrics_random_split.csv"), index=False)
    with open(os.path.join(args.out, "search_histories.json"), "w") as f:
        json.dump(dict(chosen=chosen, histories=hist), f, indent=2, default=float)
    with open(os.path.join(args.out, "run_config.json"), "w") as f:
        json.dump(dict(source=source, seed=args.seed, features=feats, splits=info,
                       detect_cfg=DetectCfg().__dict__, optuna_trials=args.optuna_trials), f, indent=2)
    print(f"\nSaved metrics, search histories (for the Ckpt 3 ReAct comparison), and config to {args.out}/")


# =============================================================================
# `selftest`: unit tests for the event-detection logic
# =============================================================================


def _toy_grid(gap_fn, defender_ahead_fn=None, att_ahead_fn=None, def_valid_fn=None,
              pos_att=2.0, pos_def=1.0, T=300.0, cfg=DetectCfg()):
    """Two cars: defender '1' and attacker '2' at constant 250 km/h. gap_fn(t) -> gap seconds."""
    t = np.arange(0, T, cfg.dt)
    n = len(t)
    spd = 250.0
    gap = np.array([gap_fn(x) for x in t], dtype=float)

    def base():
        return DriverArrays(ahead=np.full(n, "", dtype=object), dist=np.full(n, np.nan),
                            speed=np.full(n, spd), drs=np.zeros(n), lapno=np.full(n, 5.0),
                            compound=np.full(n, 1.0), tyre_age=np.full(n, 10.0),
                            pos_prev=np.full(n, np.nan), valid=np.ones(n, dtype=bool))
    A, D = base(), base()
    A.pos_prev[:], D.pos_prev[:] = pos_att, pos_def
    A.ahead[:] = "1"
    A.dist[:] = gap * spd / 3.6
    if att_ahead_fn is not None:
        for i, x in enumerate(t):
            A.ahead[i] = att_ahead_fn(x)
            if A.ahead[i] == "":
                A.dist[i] = np.nan
    if defender_ahead_fn is not None:
        for i, x in enumerate(t):
            D.ahead[i] = defender_ahead_fn(x)
    if def_valid_fn is not None:
        D.valid = np.array([def_valid_fn(x) for x in t])
    return RaceGrid(t=t, drv={"1": D, "2": A}, total_laps=57,
                    meta=dict(circuit="TOY", year=2023, round=1))


def _approach(t, t0=40.0, t1=60.0, g0=3.0, g1=0.9):
    """Gap closes linearly from g0 (at t0) to g1 (at t1); constant before/after."""
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

    print("Event-detection self-tests (synthetic grids with known ground truth)")

    # a) clean pass
    g = _toy_grid(lambda t: _approach(t) if t < 72 else 0.3,
                  defender_ahead_fn=lambda t: "2" if t >= 72 else "",
                  att_ahead_fn=lambda t: "1" if t < 72 else "")
    ev = detect_attempts(g, cfg)
    check("clean pass -> exactly 1 attempt, labelled success", len(ev) == 1 and ev[0]["y"] == 1)

    # b) approach then fall back
    g = _toy_grid(lambda t: _approach(t) if t < 62 else min(0.9 + 0.1 * (t - 62), 2.5))
    ev = detect_attempts(g, cfg)
    check("approach then drop back -> 1 attempt, labelled failure", len(ev) == 1 and ev[0]["y"] == 0)

    # c) pass that does not hold (re-passed after 3 s < HOLD_S)
    g = _toy_grid(lambda t: _approach(t) if t < 72 else 0.5,
                  defender_ahead_fn=lambda t: "2" if 72 <= t < 75 else "",
                  att_ahead_fn=lambda t: "" if 72 <= t < 75 else "1")
    ev = detect_attempts(g, cfg)
    check("pass re-passed within 3 s -> labelled failure (position must HOLD)",
          len(ev) == 1 and ev[0]["y"] == 0)

    # d) defender pits during the outcome window -> no label at all
    g = _toy_grid(lambda t: _approach(t), def_valid_fn=lambda t: t < 75)
    check("defender pits in window -> attempt discarded", len(detect_attempts(g, cfg)) == 0)

    # e) lapped defender (attacker classified AHEAD of it)
    g = _toy_grid(lambda t: _approach(t), pos_att=1.0, pos_def=5.0)
    check("lapped defender -> excluded", len(detect_attempts(g, cfg)) == 0)

    # f) already glued to the car ahead for the whole window: no fresh entry
    g = _toy_grid(lambda t: 0.8)
    check("DRS-train (never out of range) -> no new attempt", len(detect_attempts(g, cfg)) == 0)

    # g) LEAKAGE TEST: two worlds identical up to ~T0+3s, opposite outcomes.
    def world(success):
        if success:
            return _toy_grid(lambda t: _approach(t) if t < 72 else 0.3,
                             defender_ahead_fn=lambda t: "2" if t >= 72 else "",
                             att_ahead_fn=lambda t: "1" if t < 72 else "")
        return _toy_grid(lambda t: _approach(t) if t < 62 else min(0.9 + 0.1 * (t - 62), 2.5))
    e1, e0 = detect_attempts(world(True), cfg)[0], detect_attempts(world(False), cfg)[0]
    diff = [k for k in e1 if k not in ("y",) and not (pd.isna(e1[k]) and pd.isna(e0[k])) and e1[k] != e0[k]]
    check("features identical when only the post-T0 outcome differs (no leakage)",
          e1["y"] != e0["y"] and not diff)
    if diff:
        print(f"         differing fields: {diff}")

    print(f"\n{sum(results)}/{len(results)} tests passed")
    sys.exit(0 if all(results) else 1)


# =============================================================================
# CLI
# =============================================================================


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("selftest", help="unit tests for event detection (no data needed)").set_defaults(fn=cmd_selftest)

    b = sub.add_parser("build", help="download FastF1 races and detect overtake attempts")
    b.add_argument("--years", type=int, nargs="+", default=[2023])
    b.add_argument("--rounds", nargs="+", default=["1"], help="round numbers, or 'all'")
    b.add_argument("--cache-dir", default="ff1cache")
    b.add_argument("--data-dir", default="data")
    b.add_argument("--screen-s", type=float, default=3.0,
                   help="only compute telemetry for laps where the car is within this many seconds of the car "
                        "ahead at a lap boundary (speed-up); 0 disables")
    b.add_argument("--max-laps-per-driver", type=int, default=None, help="cap for quick tests")
    b.add_argument("--force", action="store_true")
    b.set_defaults(fn=cmd_build)

    r = sub.add_parser("run", help="train baselines and evaluate")
    r.add_argument("--data-dir", default="data")
    r.add_argument("--out", default="results")
    r.add_argument("--synthetic", action="store_true", help="FAKE data, pipeline smoke test only")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--test-frac", type=float, default=0.2)
    r.add_argument("--val-frac", type=float, default=0.2)
    r.add_argument("--optuna-trials", type=int, default=30)
    r.add_argument("--n-boot", type=int, default=200)
    r.set_defaults(fn=cmd_run)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
