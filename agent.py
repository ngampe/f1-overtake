#!/usr/bin/env python3
"""ReAct hyperparameter-tuning agent for XGBoost (Ckpt 3).

Objective:  theta* = argmax_{theta in Theta}  PR-AUC_val( XGB_theta trained on train ),
            subject to a budget of B evaluations (same B as Optuna).

The LLM (Gemini, function calling) alternates Thought -> Action -> Observation:
  tools: evaluate_config(params) -> val metrics ; get_history() ; finish(best_trial)
Guardrails live OUTSIDE the LLM: params are clipped to the search space, the budget
is enforced by the evaluator, and the test set is never visible to the agent.

Core is data-agnostic: it needs (X_tr, y_tr, X_va, y_va) arrays.
Usage:
  python agent.py selftest                 # synthetic data + scripted mock policy
  GEMINI_API_KEY=... python agent.py selftest --live   # synthetic data, real Gemini
"""
import argparse, json, os, sys, time
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import xgboost as xgb

# ----- search space (identical to the Optuna baseline's 8 hyperparameters) -----
SPACE = {  # identical to baseline.optuna_search so the comparison is fair
    "max_depth":        (2, 8, "int"),
    "learning_rate":    (0.01, 0.3, "log"),
    "n_estimators":     (50, 500, "int"),
    "min_child_weight": (1, 10, "int"),
    "subsample":        (0.5, 1.0, "float"),
    "colsample_bytree": (0.5, 1.0, "float"),
    "reg_lambda":       (0.01, 10.0, "log"),
    "reg_alpha":        (0.001, 1.0, "log"),
}
DEFAULT = dict(max_depth=5, learning_rate=0.1, n_estimators=200, min_child_weight=1,
               subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, reg_alpha=0.01)


def clip_params(p):
    out = {}
    for k, (lo, hi, kind) in SPACE.items():
        v = p.get(k, DEFAULT[k])
        v = min(max(float(v), lo), hi)
        out[k] = int(round(v)) if kind == "int" else float(v)
    return out


class BudgetedEvaluator:
    """Same objective as Optuna: baseline.val_pr_auc(params). Enforces budget; logs trials."""
    def __init__(self, Xtr, ytr, Xva, yva, budget=30, seed=0, score_fn=None):
        self.Xtr, self.ytr, self.Xva, self.yva = Xtr, ytr, Xva, yva
        self.budget, self.seed, self.history = budget, seed, []
        if score_fn is None:
            import baseline
            score_fn = baseline.val_pr_auc
        self.score_fn = score_fn

    @property
    def remaining(self):
        return self.budget - len(self.history)

    def evaluate(self, params):
        if self.remaining <= 0:
            return {"error": "budget exhausted; call finish"}
        p = clip_params(params)
        v = float(self.score_fn(p, self.Xtr, self.ytr, self.Xva, self.yva, self.seed))
        best = max([v] + [h["value"] for h in self.history])
        rec = {"trial": len(self.history) + 1, "value": v, "best_so_far": best, "params": p}
        self.history.append(rec)
        return {**rec, "val_pr_auc": v, "remaining": self.remaining}

    def best(self):
        return max(self.history, key=lambda h: h["value"]) if self.history else None

    def curve(self):
        return [h["best_so_far"] for h in self.history]


SYSTEM = """You tune an XGBoost classifier that predicts whether a Formula 1 overtake attempt succeeds
(~7% positive). Objective: maximise validation PR-AUC using at most {B} evaluations.
Search space: {space}.
Work in a ReAct loop: state a short Thought about what the history shows, then call evaluate_config
with a full parameter set. Start with a few diverse configs, then exploit promising regions
(overfitting shows as high depth / many trees / low min_child_weight with weak val score).
Never repeat a config. When the budget is spent or gains plateau, call finish."""

TOOLS = [
    {"name": "evaluate_config",
     "description": "Train XGBoost with these hyperparameters; returns validation PR-AUC/ROC-AUC.",
     "parameters": {"type": "object", "properties": {k: {"type": "number"} for k in SPACE},
                    "required": list(SPACE)}},
    {"name": "get_history", "description": "Return all trials so far.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "finish", "description": "Stop and report the chosen trial index.",
     "parameters": {"type": "object", "properties": {"best_trial": {"type": "integer"},
                                                      "rationale": {"type": "string"}},
                    "required": ["best_trial"]}},
]


def dispatch(ev, name, args):
    if name == "evaluate_config":
        return ev.evaluate(args)
    if name == "get_history":
        return {"history": ev.history, "remaining": ev.remaining}
    return {"ok": True}


# ----- LLM policies: each yields (thought, tool, args) given history --------------
class MockPolicy:
    """Scripted stand-in for the LLM (tests only): random-ish local search around best."""
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def next(self, ev):
        if ev.remaining <= 0:
            return "Budget spent.", "finish", {"best_trial": ev.best()["trial"]}
        if not ev.history:
            return "Start from sane default.", "evaluate_config", dict(DEFAULT)
        b = dict(ev.best()["params"])
        k = self.rng.choice(list(SPACE))
        lo, hi, kind = SPACE[k]
        b[k] = b[k] * float(np.exp(self.rng.normal(0, .4))) if kind == "log" else \
            b[k] + (hi - lo) * float(self.rng.normal(0, .15))
        return f"Perturb {k} around best.", "evaluate_config", b


def run_mock(ev, seed=0):
    pol, log = MockPolicy(seed), []
    for _ in range(ev.budget + 2):
        th, tool, args = pol.next(ev)
        if tool == "finish":
            break
        log.append({"thought": th, "tool": tool, "args": args, "obs": dispatch(ev, tool, args)})
    return log


def _generate(client, model, contents, cfg, tries=8):
    """Retry transient Gemini errors (429 rate limit, 500/503 overloaded) with backoff."""
    from google.genai import errors
    for k in range(tries):
        try:
            return client.models.generate_content(model=model, contents=contents, config=cfg)
        except (errors.ServerError, errors.ClientError) as e:
            code = getattr(e, "code", None)
            if code not in (429, 500, 503) or k == tries - 1 or 'PerDay' in str(e):
                if 'PerDay' in str(e):
                    print('  [gemini] DAILY free-tier quota exhausted: enable billing or switch GEMINI_MODEL')
                raise
            wait = min(60, 2 ** k * 2)
            print(f"  [gemini {code}] retrying in {wait}s ({k + 1}/{tries - 1})")
            time.sleep(wait)
        except Exception as e:  # dropped connection / timeout
            if k == tries - 1 or not any(w in type(e).__name__ for w in ('Protocol', 'Connect', 'Timeout', 'Network', 'Read')):
                raise
            wait = min(60, 2 ** k * 2)
            print(f"  [network] {type(e).__name__}; retrying in {wait}s ({k + 1}/{tries - 1})")
            time.sleep(wait)


def run_gemini(ev, model=None, max_turns=None, sleep=0.0):
    """Manual function-calling loop with google-genai (automatic calling disabled)."""
    from google import genai
    from google.genai import types
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        sys.exit("Set GEMINI_API_KEY in the environment (never hard-code it).")
    model = model or os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
    client = genai.Client(api_key=key)
    cfg = types.GenerateContentConfig(
        system_instruction=SYSTEM.format(B=ev.budget, space=json.dumps(SPACE)),
        tools=[types.Tool(function_declarations=TOOLS)],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        temperature=0.7)
    contents = [types.Content(role="user", parts=[types.Part(
        text=f"Begin. Budget: {ev.budget} evaluations.")])]
    log = []
    for _ in range(max_turns or ev.budget * 2 + 4):
        resp = _generate(client, model, contents, cfg)
        cand = resp.candidates[0].content
        contents.append(cand)
        calls = [p.function_call for p in cand.parts if p.function_call]
        thought = " ".join(p.text for p in cand.parts if getattr(p, "text", None))
        if not calls:
            contents.append(types.Content(role="user", parts=[types.Part(
                text="Call a tool: evaluate_config or finish.")]))
            continue
        replies, done = [], False
        for fc in calls:
            args = dict(fc.args or {})
            obs = dispatch(ev, fc.name, args)
            log.append({"thought": thought, "tool": fc.name, "args": args, "obs": obs})
            done |= fc.name == "finish"
            replies.append(types.Part.from_function_response(name=fc.name, response={"result": obs}))
        contents.append(types.Content(role="user", parts=replies))
        if done or ev.remaining <= 0:
            break
        time.sleep(sleep)
    return log


# ----- selftest -----------------------------------------------------------------
def synth(n=6000, d=17, pos=0.07, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    z = 1.2 * X[:, 0] - 0.8 * X[:, 1] + 0.6 * X[:, 2] * X[:, 3] + rng.normal(0, 1.5, n)
    y = (z > np.quantile(z, 1 - pos)).astype(int)
    return X, y


def selftest(live=False):
    X, y = synth()
    Xtr, ytr, Xva, yva = X[:4000], y[:4000], X[4000:], y[4000:]
    def sc(p, a, b, c, d, seed=0):
        m = xgb.XGBClassifier(**p, tree_method="hist", n_jobs=1, random_state=seed).fit(a, b)
        return average_precision_score(d, m.predict_proba(c)[:, 1])
    ev = BudgetedEvaluator(Xtr, ytr, Xva, yva, budget=8, score_fn=sc)
    # guardrails
    p = clip_params({"max_depth": 99, "learning_rate": -1, "n_estimators": 1e9})
    assert p["max_depth"] == 8 and p["learning_rate"] == 0.01 and p["n_estimators"] == 500
    log = run_gemini(ev) if live else run_mock(ev)
    assert len(ev.history) <= ev.budget
    assert ev.evaluate(DEFAULT).get("error") or ev.remaining > 0   # budget enforced
    c = ev.curve()
    assert all(b >= a for a, b in zip(c, c[1:]))
    print("trials:", len(ev.history), "best val PR-AUC: %.3f" % ev.best()["value"])
    print("curve:", [round(x, 3) for x in c])
    print("selftest OK", "(live Gemini)" if live else "(mock policy)")


# ----- compare: agent vs grid vs Optuna on identical splits --------------------------
def _prep(args, seed):
    """Rebuild the exact split used by `baseline.py run --seed`."""
    import baseline as B
    df = B.make_synthetic(seed=seed) if args.synthetic else B.load_events(args.data_dir)
    df = df.dropna(subset=["y"]).reset_index(drop=True)
    split = B.circuit_split(df, 0.2, 0.2, seed)
    df = B.add_track_category(df, split["train"])
    feats = [f for f in B.FEATURES if f in df.columns and df[f].notna().any()]
    tr, va, te = (df[df.circuit.isin(split[k])] for k in ("train", "val", "test"))
    return B, feats, split, tr, va, te


def cmd_compare(args):
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "ckpt"), exist_ok=True)
    for seed in args.seeds:
        path = os.path.join(args.out, f"compare_seed{seed}.json")
        if os.path.exists(path) and not args.force:
            print(f"[skip] {path}"); continue
        B, feats, split, tr, va, te = _prep(args, seed)
        Xtr, ytr = tr[feats].to_numpy(float), tr["y"].to_numpy(int)
        Xva, yva = va[feats].to_numpy(float), va["y"].to_numpy(int)
        Xte, yte = te[feats].to_numpy(float), te["y"].to_numpy(int)
        out = {"seed": seed, "budget": args.budget, "features": feats, "splits": split, "metrics": {}}
        _, out["grid"] = B.grid_search(Xtr, ytr, Xva, yva, seed)
        _, out["optuna"] = B.optuna_search(Xtr, ytr, Xva, yva, args.budget, seed)
        out["agent"] = []

        def score(name, s_va, s_te, prob=True):          # all metrics from baseline.evaluate
            out["metrics"].setdefault(name, []).append(B.evaluate(yte, s_te, B.best_f1_threshold(yva, s_va), prob))

        score("Gap heuristic", -va["gap_s"].to_numpy(float), -te["gap_s"].to_numpy(float), False)
        lr = B.make_logreg().fit(Xtr, ytr)
        score("Logistic regression", lr.predict_proba(Xva)[:, 1], lr.predict_proba(Xte)[:, 1])
        for k, name in (("grid", "XGBoost (grid)"), ("optuna", "XGBoost (Optuna)")):
            bp = max(out[k], key=lambda h: h["value"])["params"]
            m = B.make_xgb(bp, seed).fit(Xtr, ytr)
            m.save_model(os.path.join(args.out, "ckpt", f"seed{seed}_{k}.json"))
            score(name, m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1])
            out[k + "_test_pr_auc"] = out["metrics"][name][-1]["pr_auc"]
        logf = open(os.path.join(args.out, f"agent_trace_seed{seed}.log"), "w")
        for r in range(args.agent_runs):
            ev = BudgetedEvaluator(Xtr, ytr, Xva, yva, budget=args.budget, seed=seed)
            t0 = time.time()
            log = run_mock(ev, seed * 100 + r) if args.policy == "mock" else \
                run_gemini(ev, args.model, sleep=args.sleep)
            bst = ev.best()
            m = B.make_xgb(bst["params"], seed).fit(Xtr, ytr)    # chosen by VAL only
            m.save_model(os.path.join(args.out, "ckpt", f"seed{seed}_agent_run{r}.json"))
            score("ReAct agent", m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1])
            for i, st in enumerate(log, 1):              # human-readable ReAct trace (saved training log)
                o = st["obs"]
                logf.write(f"[run {r} step {i}] THOUGHT: {st['thought'][:300]}\n  ACTION: {st['tool']}"
                           f"({json.dumps(st['args'], default=float)})\n  OBS: val_pr_auc="
                           f"{o.get('value', float('nan')):.4f} best={o.get('best_so_far', float('nan')):.4f}\n")
            out["agent"].append({"history": ev.history, "log": log, "secs": time.time() - t0,
                                 "test_pr_auc": out["metrics"]["ReAct agent"][-1]["pr_auc"]})
            print(f"  seed {seed} agent run {r}: val {bst['value']:.3f} test {out['agent'][-1]['test_pr_auc']:.3f}")
        logf.close()
        json.dump(out, open(path, "w"), indent=1, default=float)
        print(f"[saved] {path}")


def cmd_demo(args):
    """Inference demo: load saved checkpoint, score a batch of unseen-circuit test events."""
    import pandas as pd
    B, feats, split, tr, va, te = _prep(args, args.seed)
    m = xgb.XGBClassifier(); m.load_model(os.path.join(args.out, "ckpt", f"seed{args.seed}_{args.method}.json"))
    rng = np.random.default_rng(args.seed)
    pos, neg = np.flatnonzero(te["y"] == 1), np.flatnonzero(te["y"] == 0)
    idx = np.concatenate([rng.choice(pos, min(args.n // 2, len(pos)), replace=False),
                          rng.choice(neg, args.n - min(args.n // 2, len(pos)), replace=False)])
    b = te.iloc[idx]; p = m.predict_proba(b[feats].to_numpy(float))[:, 1]
    thr = B.best_f1_threshold(va["y"].to_numpy(int), m.predict_proba(va[feats].to_numpy(float))[:, 1])
    d = pd.DataFrame({"circuit": b["circuit"].to_numpy(), "attacker": b["attacker"].to_numpy(),
                      "defender": b["defender"].to_numpy(), "gap_s": b["gap_s"].round(2).to_numpy(),
                      "drs_on": b["drs_on"].to_numpy(), "P(success)": p.round(3),
                      "pred": (p >= thr).astype(int), "actual": b["y"].to_numpy(int)})
    print(f"Checkpoint: seed{args.seed}_{args.method}.json | test circuits (unseen): {', '.join(split['test'])}")
    print(f"Decision threshold (chosen on validation): {thr:.3f}\n")
    print(d.sort_values("P(success)", ascending=False).to_string(index=False))
    print(f"\nbatch accuracy {(d.pred == d.actual).mean():.0%}  (batch is class-balanced on purpose; real rate is ~7%)")


def _curve(hist, n):
    c = [h["best_so_far"] for h in hist][:n]
    return c + [c[-1]] * (n - len(c))


def cmd_report(args):
    import glob, pandas as pd
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    files = sorted(glob.glob(os.path.join(args.out, "compare_seed*.json")))
    if not files: sys.exit("No compare_seed*.json; run `compare` first.")
    D = [json.load(open(f)) for f in files]; n = D[0]["budget"]
    rows, curves = [], {"Grid": [], "Optuna": [], "ReAct agent": []}
    for d in D:
        a_val = np.mean([max(h["value"] for h in r["history"]) for r in d["agent"]])
        a_test = np.mean([r["test_pr_auc"] for r in d["agent"]])
        ref = max(max(h["value"] for h in d["grid"]), max(h["value"] for h in d["optuna"]),
                  *[max(h["value"] for h in r["history"]) for r in d["agent"]])
        cg, co = _curve(d["grid"], n), _curve(d["optuna"], n)
        ca = np.mean([_curve(r["history"], n) for r in d["agent"]], axis=0)
        for k, c in (("Grid", cg), ("Optuna", co), ("ReAct agent", ca)): curves[k].append(np.array(c) - ref)
        within = lambda c: next((i + 1 for i, v in enumerate(c) if ref - v <= 0.01), np.nan)
        rows.append(dict(seed=d["seed"], grid_val=cg[-1], optuna_val=co[-1], agent_val=a_val,
                         grid_test=d["grid_test_pr_auc"], optuna_test=d["optuna_test_pr_auc"], agent_test=a_test,
                         grid_regret_auc=-np.mean(np.array(cg) - ref), optuna_regret_auc=-np.mean(np.array(co) - ref),
                         agent_regret_auc=-np.mean(ca - ref), grid_to001=within(cg), optuna_to001=within(co), agent_to001=within(ca)))
    T = pd.DataFrame(rows); T.to_csv(os.path.join(args.out, "agent_compare.csv"), index=False)
    M = {}  # method -> metric -> list over (seed, run)
    for d in D:
        for name, lst in d["metrics"].items():
            for r in lst:
                for k in ("pr_auc", "roc_auc", "f1", "ece"):
                    M.setdefault(name, {}).setdefault(k, []).append(r[k])
    fm = lambda v: "--" if not len(v) or np.isnan(v).all() else f"{np.nanmean(v):.3f}"
    sd = lambda v: "" if len(v) < 2 or np.isnan(v).all() else f" $\\pm$ {np.nanstd(v, ddof=1):.3f}"
    lines = ["\\begin{tabular}{lcccc}\\toprule", "Method & PR-AUC & ROC-AUC & F1 & ECE \\\\\\midrule"]
    md = ["| method | PR-AUC | ROC-AUC | F1 | ECE |", "|---|---|---|---|---|"]
    for name in ("Gap heuristic", "Logistic regression", "XGBoost (grid)", "XGBoost (Optuna)", "ReAct agent"):
        if name in M:
            lines.append(f"{name} & {fm(M[name]['pr_auc'])}{sd(M[name]['pr_auc'])} & {fm(M[name]['roc_auc'])} & {fm(M[name]['f1'])} & {fm(M[name]['ece'])} \\\\")
            md.append(f"| {name} | " + " | ".join(fm(M[name][k]) + sd(M[name][k]).replace("$\\pm$", "±") for k in ("pr_auc", "roc_auc", "f1", "ece")) + " |")
    lines.append("\\bottomrule\\end{tabular}")
    open(os.path.join(args.out, "table_agent.tex"), "w").write("\n".join(lines) + "\n")
    open(os.path.join(args.out, "table_agent.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))
    print("\nSearch efficiency (mean over splits): regret-AUC grid %.3f optuna %.3f agent %.3f; evals-to-0.01: grid %s optuna %s agent %s" % (
        T.grid_regret_auc.mean(), T.optuna_regret_auc.mean(), T.agent_regret_auc.mean(),
        T.grid_to001.mean(), T.optuna_to001.mean(), T.agent_to001.mean()))
    d_ = T["agent_test"] - T["optuna_test"]
    print(T.round(3).to_string(index=False))
    print(f"\npaired agent - Optuna test PR-AUC: {d_.mean():+.3f} +/- {d_.std(ddof=1) if len(d_)>1 else float('nan'):.3f} over {len(T)} splits")
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    for k, col, ls in (("Grid", "#eb6834", "-"), ("Optuna", "#1baf7a", "-"), ("ReAct agent", "#7a4fd1", "-")):
        ax.plot(range(1, n + 1), np.mean(curves[k], axis=0), color=col, ls=ls, lw=1.8, label=k)
    ax.set_xlabel("Evaluations"); ax.set_ylabel("Val PR-AUC vs best (0)")
    ax.legend(frameon=False, fontsize=7); ax.grid(color="#e6e5e1")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig.savefig(os.path.join(args.out, "fig_agent_curve.pdf"), bbox_inches="tight")
    print("wrote table_agent.tex, fig_agent_curve.pdf")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("selftest"); t.add_argument("--live", action="store_true")
    c = sub.add_parser("compare")
    c.add_argument("--data-dir", default="data"); c.add_argument("--out", default="results_agent")
    c.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    c.add_argument("--budget", type=int, default=30); c.add_argument("--agent-runs", type=int, default=3)
    c.add_argument("--policy", choices=["gemini", "mock"], default="gemini")
    c.add_argument("--model", default=None); c.add_argument("--sleep", type=float, default=0.0)
    c.add_argument("--synthetic", action="store_true"); c.add_argument("--force", action="store_true")
    r = sub.add_parser("report"); r.add_argument("--out", default="results_agent")
    dm = sub.add_parser("demo"); dm.add_argument("--data-dir", default="data"); dm.add_argument("--out", default="results_agent")
    dm.add_argument("--seed", type=int, default=0); dm.add_argument("--n", type=int, default=10)
    dm.add_argument("--method", default="agent_run0", help="agent_run0 | optuna | grid"); dm.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    if a.cmd == "selftest": selftest(a.live)
    elif a.cmd == "compare": cmd_compare(a)
    elif a.cmd == "demo": cmd_demo(a)
    else: cmd_report(a)
