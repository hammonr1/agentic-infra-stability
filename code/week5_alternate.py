#!/usr/bin/env python3
"""WEEK 5 --- Close the gap. Alternate the design and the witness.

Week 4 solved an LP at ONE witness, anchored at the fallback. Every answer it
gave was valid, and every answer was pessimistic: the certificate reported a
contraction level of 0.94 while the design actually achieved rho(K) ~ 0.87. That
slack is authority we refused for no reason.

This week you close it. The loop is three lines:

      solve the LP at the current witness v_k      ->  design y_k
      recompute the witness at that design          ->  v_{k+1}
      repeat

  Part A   one system, iteration by iteration
  Part B   why each step is valid (and what is NOT claimed)
  Part C   the medium experiment: 30 systems, before and after
  Part D   where it stops, and why

    python week5_alternate.py                 # one system, verbose
    python week5_alternate.py --systems 30    # the medium experiment
    python week5_alternate.py --exercise
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from week4_design import (ALPHA_BAR, assemble, fallback_witness, make_system,
                          rho, solve_joint, week3_gamma_star)

TOL_V = 1e-8          # witness movement
TOL_C = 1e-6          # relative objective change
TOL_NUM = 1e-6        # above the LP solver's own tolerance (~1e-7): below this,
                      # a "violation" is arithmetic, not a broken claim
MAX_IT = 20           # predeclared cap


# ===========================================================================
def refine_witness(K, v, steps=60):
    """Power iteration: v <- K v, renormalised.

    For a nonnegative primitive K this converges to the Perron vector, at which
    max_i (Kv)_i / v_i equals rho(K) exactly -- the tightest possible witness
    for that design. We cap the steps because we do not need the limit, only a
    better witness than the one we had.
    """
    out = v.copy()
    for _ in range(steps):
        w = K @ out
        if not np.all(w > 0):          # degenerate: keep what we had
            return out
        out = w / w.max()
    return out


def row_ratio(K, v):
    return float(np.max(K @ v / v))


def alternate(sys_, alpha_bar=ALPHA_BAR, max_it=MAX_IT, verbose=False):
    """The Week 5 loop. Returns the full trace."""
    v = fallback_witness(sys_, alpha=alpha_bar)
    trace, best = [], None
    for k in range(max_it):
        sol = solve_joint(sys_, v, alpha_bar=alpha_bar)
        if sol is None:
            break
        K = assemble(sys_, sol["m"], sol["z"], sol["g"])
        rec = dict(it=k, obj=sol["obj"], mean_gamma=float(sol["g"].mean()),
                   links=int((sol["z"] > 0.5).sum()),
                   mean_m=float(sol["m"].mean()),
                   ratio_at_solve=row_ratio(K, v), rho=rho(K))
        v_new = refine_witness(K, v)
        rec["ratio_after_refine"] = row_ratio(K, v_new)
        rec["unused"] = alpha_bar - rec["rho"]   # certified margin left on the table
        rec["certified"] = rec["ratio_after_refine"] <= alpha_bar + TOL_NUM
        trace.append(rec)
        if verbose:
            print(f"  it {k:>2}  obj {rec['obj']:>9.4f}  "
                  f"gamma {rec['mean_gamma']:.4f}  links {rec['links']:>3}  "
                  f"certified@ {rec['ratio_at_solve']:.4f}  "
                  f"rho {rec['rho']:.4f}  unused {rec['unused']:.4f}")
        if best is None or rec["obj"] <= best["obj"]:
            best = dict(rec, m=sol["m"], z=sol["z"], g=sol["g"])
        dv = float(np.max(np.abs(v_new - v)))
        dc = (abs(trace[-2]["obj"] - rec["obj"]) / max(abs(rec["obj"]), 1e-12)
              if len(trace) > 1 else 1.0)
        v = v_new
        if dv <= TOL_V:
            rec["stop"] = "witness converged"; break
        if dc <= TOL_C:
            rec["stop"] = "objective converged"; break
    else:
        if trace:
            trace[-1]["stop"] = "iteration cap"
    return trace, best, v


# ===========================================================================
def part_a_b(seed, n_p, n_a):
    sys_ = make_system(seed, n_p, n_a)
    print(f"\n{'='*76}\nPART A --- one system, iteration by iteration "
          f"(seed {seed}, {n_p} sectors, {n_a} agents)\n{'='*76}")
    trace, best, v = alternate(sys_, verbose=True)
    first, last = trace[0], trace[-1]
    print(f"\n  stopped after {len(trace)} iterations: "
          f"{last.get('stop', 'cap')}")
    print(f"  objective      {first['obj']:>9.4f}  ->  {last['obj']:>9.4f}"
          f"   ({100*(last['obj']-first['obj'])/abs(first['obj']):+.1f}%)")
    print(f"  mean authority {first['mean_gamma']:>9.4f}  ->  "
          f"{last['mean_gamma']:>9.4f}"
          f"   ({last['mean_gamma']/max(first['mean_gamma'],1e-12):.2f}x)")
    print(f"  unused margin  {first['unused']:>9.4f}  ->  {last['unused']:>9.4f}"
          f"   ({100*(1-last['unused']/max(first['unused'],1e-12)):.0f}% recovered)")
    mono = all(trace[i+1]["obj"] <= trace[i]["obj"] + TOL_NUM
               for i in range(len(trace)-1))
    print(f"  objective monotone non-increasing: {mono}")
    print(f"  every iterate certified          : "
          f"{all(t['certified'] for t in trace)}")

    print(f"\n{'='*76}\nPART B --- why each step is valid\n{'='*76}")
    print("""  1. The LP is solved exactly at the current witness, so the design it
     returns satisfies  K(y_k) v_k <= alpha_bar v_k  by construction.
  2. Refining the witness cannot break that. For nonnegative K and v > 0,
     power iteration gives  max_i (K v')_i / v'_i  <=  max_i (K v)_i / v_i.
     So y_k is still certified at the NEW witness, at a level that is at
     least as tight. (This is Collatz-Wielandt from Week 3, used forwards.)
  3. Therefore y_k is feasible for the NEXT LP, whose optimum can only be at
     least as good. Hence the objective never increases.

  What is NOT claimed. The iterates may cycle; a decreasing bounded sequence
  of objective values does not imply the designs converge. We claim feasible
  descent with a certificate at every step, and we stop on a predeclared
  rule -- not on a convergence theorem.

  A note on arithmetic. Steps 1-3 are exact in real arithmetic. In floating
  point the LP solver works to about 1e-7, so you will see certification
  excesses around 1e-9 and objective "increases" around 3e-7. We test against
  1e-6, above the solver tolerance. Report that threshold; do not report
  "monotone" against 1e-15 and then quietly widen it when it fails.""")
    return sys_, trace


def part_d(sys_):
    print(f"\n{'='*76}\nPART D --- where it stops, and why\n{'='*76}")
    v = fallback_witness(sys_)
    K_anchor = assemble(sys_, np.ones(sys_["n_p"]),
                        np.zeros(sys_["n_e"]), np.zeros(sys_["n_a"]))
    print(f"  anchor (fallback) rho              = {rho(K_anchor):.4f}")
    trace, best, v_end = alternate(sys_)
    last = trace[-1]
    print(f"  final design rho(K)                = {last['rho']:.4f}")
    print(f"  final certified level              = {last['ratio_after_refine']:.4f}")
    print(f"  unused margin remaining            = {last['unused']:.6f}")
    print("""
  Read that last number. At iteration 0 the certificate said 0.94 while the
  design achieved rho = 0.87 -- seven points of contraction margin bought and
  never spent. After the loop, the design sits exactly on its own certificate.
  There is nothing left to recover WITHOUT changing the target alpha_bar.

  That is the right place to stop. The loop does not make the system safer; it
  stops us from being needlessly cautious about a system that was already
  within budget. Those are different things, and conflating them is how people
  talk themselves into relaxing alpha_bar.""")


# ===========================================================================
def medium(n_sys, n_p, n_a, seed0, out):
    print(f"{'='*76}\nPART C --- MEDIUM EXPERIMENT: {n_sys} systems, "
          f"{n_p} sectors, {n_a} agents\n{'='*76}")
    recs = []
    for k in range(n_sys):
        sd = seed0 + k
        sys_ = make_system(sd, n_p, n_a)
        try:
            trace, best, _ = alternate(sys_)
        except RuntimeError:
            continue
        if not trace:
            continue
        f, l = trace[0], trace[-1]
        v0 = fallback_witness(sys_)
        g3 = week3_gamma_star(sys_, v0, f.get("m_arr", np.zeros(sys_["n_p"]))
                              if False else np.zeros(sys_["n_p"]),
                              np.zeros(sys_["n_e"]))
        recs.append(dict(seed=sd, iters=len(trace), stop=l.get("stop", "cap"),
                         obj0=f["obj"], obj1=l["obj"],
                         gamma0=f["mean_gamma"], gamma1=l["mean_gamma"],
                         links0=f["links"], links1=l["links"],
                         slack0=f["unused"], slack1=l["unused"],
                         rho1=l["rho"], cert=all(t["certified"] for t in trace),
                         mono=all(trace[i+1]["obj"] <= trace[i]["obj"] + TOL_NUM
                                  for i in range(len(trace)-1))))
        print(f"  [{k+1:>2}/{n_sys}] seed {sd}  iters {len(trace):>2}  "
              f"gamma {f['mean_gamma']:.4f} -> {l['mean_gamma']:.4f}  "
              f"unused {f['unused']:.4f} -> {l['unused']:.4f}")

    import statistics as st
    g_ratio = [r["gamma1"] / max(r["gamma0"], 1e-12) for r in recs]
    s0 = [r["slack0"] for r in recs]; s1 = [r["slack1"] for r in recs]
    print(f"\n  systems solved                 : {len(recs)}/{n_sys}")
    print(f"  iterations to stop             : median {st.median([r['iters'] for r in recs]):.0f} "
          f"(max {max(r['iters'] for r in recs)})")
    print(f"  authority gain, after vs before: median {st.median(g_ratio):.2f}x  "
          f"IQR [{np.percentile(g_ratio,25):.2f}, {np.percentile(g_ratio,75):.2f}]")
    print(f"  unused certified margin        : median {st.median(s0):.4f} "
          f"-> {st.median(s1):.4f}  "
          f"({100*(1-st.median(s1)/max(st.median(s0),1e-12)):.0f}% recovered)")
    print(f"  every iterate certified        : {all(r['cert'] for r in recs)}")
    print(f"  objective monotone             : {all(r['mono'] for r in recs)}")
    from collections import Counter
    print(f"  stopping reason                : "
          f"{dict(Counter(r['stop'] for r in recs))}")
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(recs, indent=2))
        print(f"  written to {out}")


EXERCISES = """
EXERCISES (bring to the Week 6 meeting)

  1. In Part A, the objective falls fastest on the first iteration and then
     flattens. Explain which of the three design families (m, z, gamma) is
     responsible for most of the first-iteration gain. Support it with numbers
     from the trace.

  2. Replace power iteration with a SINGLE step (steps=1 in refine_witness).
     How much of the gain survives? This tells you how much of the benefit is
     "any better witness" versus "the best witness".

  3. Start the loop from a deliberately bad witness -- all ones instead of the
     resolvent anchor. Does it reach the same place? Report the iteration count
     and the final slack for five systems. What does this say about whether the
     loop has a unique fixed point?

  4. Instrument the loop to record the DESIGN, not just the objective. Do the
     iterates settle, or do they move while the objective plateaus? Plot
     ||y_{k+1} - y_k|| against k for one system and state what you see.

  5. The stopping rule is predeclared (witness tolerance 1e-8, relative cost
     1e-6, cap 20). Re-run with the cap at 3. How much do you lose? Argue for
     a cap you would actually deploy, with a number.

  6. Compare your Week 4 prediction (exercise 6) against what actually
     happened. If you were wrong, say exactly which step of the reasoning
     failed. This matters more than being right.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", type=int, default=1)
    ap.add_argument("--n-p", type=int, default=24)
    ap.add_argument("--n-a", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20260829)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--exercise", action="store_true")
    args = ap.parse_args()

    if args.systems == 1:
        sys_, trace = part_a_b(args.seed, args.n_p, args.n_a)
        part_d(sys_)
    else:
        medium(args.systems, args.n_p, args.n_a, args.seed, args.output)

    if args.exercise:
        print(EXERCISES)
    else:
        print("\n(run with --exercise to see this week's exercises)")


if __name__ == "__main__":
    main()
