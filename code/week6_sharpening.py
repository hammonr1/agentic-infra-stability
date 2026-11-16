#!/usr/bin/env python3
"""WEEK 6 --- The threat model, and what "sharpening" means.

Weeks 4-5 designed against the NOMINAL comparison matrix. Nothing so far asks
what happens when an agent is not merely wrong but actively compromised. This
week adds a declared attack set, writes the certificate that survives it, and
then sharpens that certificate.

"Sharpening" has a precise meaning here: replacing a valid constraint by a
tighter valid constraint. Both certify. One refuses designs the other admits.
The gap is pure conservatism and it is measurable.

  Part A  the declared attack set and the per-row source loads b[i,s]
  Part B  the GLOBAL surrogate  R_f  (valid, and where it comes from)
  Part C  the ROWWISE counterpart  R_f^(i)  (valid, and never worse)
  Part D  measure the gap: conservatism ratio, and when global goes vacuous
  Part E  both are convex -- verify the top-f epigraph identity numerically

    python week6_sharpening.py                  # one system, verbose
    python week6_sharpening.py --systems 40     # the experiment
    python week6_sharpening.py --verify         # Part E identity checks only
    python week6_sharpening.py --exercise
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from week4_design import (ALPHA_BAR, KAPPA_M, assemble, fallback_witness,
                          make_system, rho, solve_joint)

A_BAR = 1.0        # declared attack amplitude: a compromised source may amplify
                   # its own message gains by up to this factor
F_COMP = 2         # declared group size: at most f sources compromised at once


# ===========================================================================
# Part A --- the attack set, and the per-row source loads
# ===========================================================================
def row_loads(sys_, z, v):
    r"""b[i, s] = witness-scaled load that source s places on row i.

    THE ATTACK SET.  A compromised group B, |B| <= f, amplifies the message
    gains of its own outgoing links by a factor a_s in [0, a_bar]:

        A[q, s]  ->  (1 + a_s) * A[q, s]      for s in B, every target q.

    Nothing else changes: the attacker does not invent new links, does not
    touch the physical blocks, and does not alter sensing. That is what
    "declared" means -- the set is written down, and anything outside it is
    outside the guarantee.

    THE INDUCED ROW CONSUMPTION.  Row a of  K v <= alpha v  gains exactly

        sum_{s in B} a_s * sum_{q: tgt=a, src=s} w_e z_e v^A_s / v^A_a
                          \_______________________________________/
                                        =: b[a, s]

    so b[a, s] is the quantity we must budget. Only AGENT rows are affected --
    the attack lives in the message block -- but we return the full matrix so
    the indexing matches the rows of the LP.
    """
    n_p, n_a, n_e = sys_["n_p"], sys_["n_a"], sys_["n_e"]
    vA = v[n_p:]
    b = np.zeros((n_p + n_a, n_a))
    for e in range(n_e):
        t, s = sys_["edge_t"][e], sys_["edge_s"][e]
        b[n_p + t, s] += sys_["edge_w"][e] * z[e] * vA[s] / vA[t]
    return b


def nominal_ratios(sys_, m, z, g, v):
    """row-wise  (K v)_i / v_i  for the nominal design."""
    K = assemble(sys_, m, z, g)
    return (K @ v) / v


def topf(x, f):
    """Sum of the f largest entries. The object both certificates are built on."""
    return float(np.sort(np.asarray(x))[::-1][:f].sum())


# ===========================================================================
# Parts B & C --- the two certificates
# ===========================================================================
def global_Rf(b, f=F_COMP):
    """GLOBAL surrogate.  ell_s = max_i b[i,s]   then   R_f = topf(ell, f).

    Where it comes from: you want one number that bounds the attack's effect on
    EVERY row, so you take each source's worst row, then the worst f sources.
    Valid, and simple. But the max over i is taken BEFORE the sum over s, so
    R_f can combine worst cases from rows that cannot both be worst at once.
    """
    ell = b.max(axis=0)
    return topf(ell, f), ell


def rowwise_Rf(b, f=F_COMP):
    """ROWWISE counterpart.  For each row i:  R_f^(i) = topf(b[i, :], f).

    Same construction, done per row. Since b[i,s] <= max_j b[j,s] = ell_s for
    every i, we get R_f^(i) <= R_f for every row -- the rowwise constraint is
    NEVER more conservative, and usually much less.
    """
    return np.array([topf(b[i, :], f) for i in range(b.shape[0])])


def attacked_rate(sys_, m, z, g, v, f=F_COMP, a_bar=A_BAR, mode="rowwise"):
    """Worst-case  (K^atk v)_i / v_i  over the declared attack set, per row."""
    nom = nominal_ratios(sys_, m, z, g, v)
    b = row_loads(sys_, z, v)
    if mode == "rowwise":
        return nom + a_bar * rowwise_Rf(b, f)
    Rf, _ = global_Rf(b, f)
    return nom + a_bar * Rf          # the same number added to every row


def true_worst_rate(sys_, m, z, g, v, f=F_COMP, a_bar=A_BAR):
    """Brute force: enumerate every group of f sources, amplify fully, measure.

    This is the ground truth the two certificates are approximating. Only
    feasible because n_a is small -- that is exactly why we need certificates.
    """
    from itertools import combinations
    n_p, n_a, n_e = sys_["n_p"], sys_["n_a"], sys_["n_e"]
    worst = -np.inf
    for B in combinations(range(n_a), f):
        z_atk = z.copy()
        for e in range(n_e):
            if sys_["edge_s"][e] in B:
                z_atk[e] = z[e] * (1.0 + a_bar)
        K = assemble(sys_, m, z_atk, g)
        worst = max(worst, float(np.max((K @ v) / v)))
    return worst


# ===========================================================================
# the LP, with either certificate as a hard constraint
# ===========================================================================
def solve_with_threat(sys_, v, mode, f=F_COMP, a_bar=A_BAR,
                      alpha_bar=ALPHA_BAR, cost=None):
    """Same LP as Week 4, plus the attack budget -- via the Part E epigraph.

    The naive approach is to guess which f sources are worst, charge those, and
    iterate. It does not work: the LP simply shifts load onto sources outside
    the guessed set and the constraint is dodged. (Try it -- exercise Q7.)

    The epigraph of Part E removes the guess. For a row i,

        topf(b[i,:], f)  =  min over tau_i, r_i >= 0 of  f*tau_i + sum_s r_{i,s}
                            subject to  r_{i,s} >= b_{i,s}(z) - tau_i .

    Since b_{i,s}(z) is linear in z, every constraint here is linear, and the
    LP chooses the worst set itself. Variables:

        x = [ m (n_p) | z (n_e) | gamma (n_a) | tau | r | ell ]

    rowwise : tau has n_a entries (one per agent row), r has n_a*n_a.
    global  : tau is one scalar and r has n_a entries, over the per-source
              maxima ell_s >= b_{i,s} for every row i -- which needs ell as
              n_a more variables and one more block of constraints.
    """
    from week4_design import build_lp
    n_p, n_a, n_e = sys_["n_p"], sys_["n_a"], sys_["n_e"]
    if mode == "none":
        return solve_joint(sys_, v, alpha_bar, cost)
    vA = v[n_p:]

    c0, A0, b0 = build_lp(sys_, v, alpha_bar, cost)
    n_base = n_p + n_e + n_a
    n_rows0 = A0.shape[0]

    # --- coefficient of z_e in b[a, s], for the edges that realise it --------
    def bcoef(a, s):
        """row vector over z giving b[a,s](z)."""
        out = np.zeros(n_e)
        for e in range(n_e):
            if sys_["edge_t"][e] == a and sys_["edge_s"][e] == s:
                out[e] += sys_["edge_w"][e] * vA[s] / vA[a]
        return out

    if mode == "rowwise":
        n_tau, n_r, n_ell = n_a, n_a * n_a, 0
    else:
        n_tau, n_r, n_ell = 1, n_a, n_a
    nx = n_base + n_tau + n_r + n_ell
    t0 = n_base
    r0 = t0 + n_tau
    e0 = r0 + n_r

    c = np.concatenate([c0, np.zeros(nx - n_base)])
    A = np.zeros((n_rows0, nx)); A[:, :n_base] = A0
    b = b0.copy()
    extra_A, extra_b = [], []

    if mode == "rowwise":
        for a in range(n_a):
            i = n_p + a
            # charge a_bar * vA[a] * (f*tau_a + sum_s r_{a,s}) to agent row i
            A[i, t0 + a] += a_bar * vA[a] * f
            A[i, r0 + a * n_a:r0 + (a + 1) * n_a] += a_bar * vA[a]
            for s in range(n_a):
                row = np.zeros(nx)
                row[n_p:n_p + n_e] = bcoef(a, s)      # b[a,s](z)
                row[t0 + a] = -1.0                     # - tau_a
                row[r0 + a * n_a + s] = -1.0           # - r_{a,s}
                extra_A.append(row); extra_b.append(0.0)
    else:
        for i in range(n_p + n_a):                     # charged to EVERY row
            A[i, t0] += a_bar * v[i] * f
            A[i, r0:r0 + n_a] += a_bar * v[i]
        for s in range(n_a):
            row = np.zeros(nx)
            row[e0 + s] = 1.0; row[t0] = -1.0; row[r0 + s] = -1.0
            extra_A.append(row); extra_b.append(0.0)   # ell_s - tau - r_s <= 0
            for a in range(n_a):                       # ell_s >= b[a,s](z)
                row2 = np.zeros(nx)
                row2[n_p:n_p + n_e] = bcoef(a, s)
                row2[e0 + s] = -1.0
                extra_A.append(row2); extra_b.append(0.0)

    A = np.vstack([A] + [np.array(extra_A)]) if extra_A else A
    b = np.concatenate([b, np.array(extra_b)]) if extra_b else b

    bounds = [(0.0, 1.0)] * n_base + [(0.0, None)] * (nx - n_base)
    res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")
    if not res.success:
        return None
    x = res.x
    return dict(m=x[:n_p], z=x[n_p:n_p + n_e], g=x[n_p + n_e:n_base],
                obj=float(c0 @ x[:n_base]))


# ===========================================================================
# Part E --- convexity: the top-f epigraph identity
# ===========================================================================
def verify_epigraph(trials=4000, seed=20260829):
    """topf(x, f) = min over tau, r >= 0 of  f*tau + sum r_s  s.t. r_s >= x_s - tau.

    This is why BOTH certificates are convex: the sum of the f largest entries
    is the optimal value of a linear program in (tau, r), so adding it to a
    constraint keeps the feasible set convex. Without this the rowwise
    constraint would be a combinatorial max over subsets.
    """
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(trials):
        n = int(rng.integers(3, 12))
        f = int(rng.integers(1, n + 1))
        x = rng.random(n) * 3.0
        direct = topf(x, f)
        taus = np.concatenate(([0.0], np.sort(x)))      # optimum is at an x value
        epi = float(np.min(f * taus + np.maximum(x[None, :] - taus[:, None], 0).sum(1)))
        worst = max(worst, abs(epi - direct))
    return worst


def verify_ordering(sys_list, f=F_COMP):
    """R_f^(i) <= R_f for every row, every system. The sharpening is one-sided."""
    bad = 0
    gaps = []
    for sys_ in sys_list:
        v = fallback_witness(sys_)
        z = np.random.default_rng(0).random(sys_["n_e"])
        b = row_loads(sys_, z, v)
        Rg, _ = global_Rf(b, f)
        Rr = rowwise_Rf(b, f)
        if np.any(Rr > Rg + 1e-12):
            bad += 1
        act = Rr > 1e-12
        if act.any():
            gaps.append(float(Rg / np.median(Rr[act])))
    return bad, gaps


# ===========================================================================
def report_one(seed, n_p, n_a, f, a_bar):
    sys_ = make_system(seed, n_p, n_a)
    v = fallback_witness(sys_)
    print(f"\n{'='*76}\nPART A --- the declared attack set   "
          f"(f = {f} sources, amplitude a_bar = {a_bar})\n{'='*76}")
    base = solve_joint(sys_, v)
    b = row_loads(sys_, base["z"], v)
    Rg, ell = global_Rf(b, f)
    Rr = rowwise_Rf(b, f)
    act = Rr > 1e-12
    print(f"  Week-4 design: {int((base['z']>0.5).sum())} links on, "
          f"mean gamma {base['g'].mean():.4f}")
    print(f"  per-source loads  ell_s = max_i b[i,s]:  "
          f"max {ell.max():.4f}  median {np.median(ell):.4f}")

    print(f"\n{'='*76}\nPARTS B & C --- two certificates for the same attack\n{'='*76}")
    print(f"  GLOBAL   R_f              = {Rg:.4f}    (one number, charged to every row)")
    print(f"  ROWWISE  R_f^(i)          : max {Rr.max():.4f}  "
          f"median {np.median(Rr[act]):.4f}  min {Rr[act].min():.4f}")
    print(f"  ordering R_f^(i) <= R_f   : {bool(np.all(Rr <= Rg + 1e-12))}")
    print(f"  conservatism R_f / median = {Rg/np.median(Rr[act]):.2f}x")

    print(f"\n{'='*76}\nPART D --- does the Week-4 design survive the attack?\n{'='*76}")
    nom = nominal_ratios(sys_, base["m"], base["z"], base["g"], v)
    rg = attacked_rate(sys_, base["m"], base["z"], base["g"], v, f, a_bar, "global")
    rr = attacked_rate(sys_, base["m"], base["z"], base["g"], v, f, a_bar, "rowwise")
    tw = true_worst_rate(sys_, base["m"], base["z"], base["g"], v, f, a_bar)
    print(f"  nominal   max row ratio = {nom.max():.4f}   <= {ALPHA_BAR}: "
          f"{nom.max() <= ALPHA_BAR + 1e-9}")
    print(f"  TRUE worst (brute force over all {f}-subsets) = {tw:.4f}")
    print(f"  rowwise certificate bound                     = {rr.max():.4f}"
          f"   (slack vs truth {rr.max()-tw:+.4f})")
    print(f"  global  certificate bound                     = {rg.max():.4f}"
          f"   (slack vs truth {rg.max()-tw:+.4f})")
    print(f"\n  certifies at a_bar = {a_bar}?   rowwise: {rr.max() <= ALPHA_BAR+1e-9}"
          f"    global: {rg.max() <= ALPHA_BAR+1e-9}")

    print(f"\n  Now DESIGN against each certificate rather than checking afterwards:")
    for mode in ("rowwise", "global"):
        s = solve_with_threat(sys_, v, mode, f, a_bar)
        if s is None:
            print(f"    {mode:8s}: INFEASIBLE at a_bar = {a_bar}")
            continue
        r = attacked_rate(sys_, s["m"], s["z"], s["g"], v, f, a_bar, mode)
        print(f"    {mode:8s}: links {int((s['z']>0.5).sum()):>3}  "
              f"mean gamma {s['g'].mean():.4f}  obj {s['obj']:>9.4f}  "
              f"attacked rate {r.max():.4f}  certified {r.max() <= ALPHA_BAR+1e-9}")
    return sys_


def experiment(n_sys, n_p, n_a, f, a_bar, seed0, out):
    print(f"{'='*76}\nEXPERIMENT --- {n_sys} systems, {n_p} sectors, {n_a} agents, "
          f"f = {f}, a_bar = {a_bar}\n{'='*76}")
    recs = []
    for k in range(n_sys):
        sd = seed0 + k
        sys_ = make_system(sd, n_p, n_a)
        try:
            v = fallback_witness(sys_)
        except RuntimeError:
            continue
        base = solve_joint(sys_, v)
        if base is None:
            continue
        b = row_loads(sys_, base["z"], v)
        Rg, _ = global_Rf(b, f)
        Rr = rowwise_Rf(b, f); act = Rr > 1e-12
        if not act.any():
            continue
        srow = solve_with_threat(sys_, v, "rowwise", f, a_bar)
        sglob = solve_with_threat(sys_, v, "global", f, a_bar)
        def cert(s, mode):
            if s is None:
                return False, None, None
            r = attacked_rate(sys_, s["m"], s["z"], s["g"], v, f, a_bar, mode)
            return bool(r.max() <= ALPHA_BAR + 1e-9), float(s["obj"]), \
                   float(s["g"].mean())
        cr, or_, gr = cert(srow, "rowwise")
        cg, og, gg = cert(sglob, "global")
        recs.append(dict(seed=sd, ratio=float(Rg/np.median(Rr[act])),
                         row_cert=cr, glob_cert=cg,
                         row_obj=or_, glob_obj=og,
                         row_gamma=gr, glob_gamma=gg,
                         base_obj=float(base["obj"])))
        print(f"  [{k+1:>2}/{n_sys}] seed {sd}  conservatism {recs[-1]['ratio']:>5.2f}x   "
              f"rowwise {'OK ' if cr else 'no '}  global {'OK ' if cg else 'no '}")

    import statistics as st
    rt = [r["ratio"] for r in recs]
    nr = sum(r["row_cert"] for r in recs); ng = sum(r["glob_cert"] for r in recs)
    print(f"\n  systems                       : {len(recs)}")
    print(f"  conservatism R_f / median R_f^(i): mean {st.mean(rt):.2f}x  "
          f"median {st.median(rt):.2f}x  min {min(rt):.2f}x  max {max(rt):.2f}x")
    print(f"  certified at a_bar = {a_bar}:  rowwise {nr}/{len(recs)}   "
          f"global {ng}/{len(recs)}")
    both = [r for r in recs if r["row_cert"] and r["glob_cert"]]
    if both:
        cost = [100*(r["glob_obj"]-r["row_obj"])/abs(r["row_obj"]) for r in both]
        print(f"  where BOTH certify, global costs  : median {st.median(cost):+.1f}% "
              f"in objective")
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(recs, indent=2))
        print(f"  written to {out}")


EXERCISES = """
QUESTIONS, AND THE COMMAND THAT ANSWERS EACH

  Q1  Why is  R_f^(i) <= R_f  true by construction? One line. What would have
      to break for it to fail?
          VERIFY:  python3 week6_sharpening.py --verify
      Report the violation count over 25 systems. It should be 0.

  Q2  The top-f epigraph identity is what makes both certificates convex.
      Report the maximum error from --verify. Then explain in two sentences
      why, WITHOUT it, the rowwise constraint would be combinatorial rather
      than linear.

  Q3  How conservative is the global surrogate?
          VERIFY:  python3 week6_sharpening.py --systems 40
      Report mean, median, min and max of the conservatism ratio. Is the
      minimum ever exactly 1.00? Construct on paper a system where it WOULD
      be 1.00 and describe its structure in one sentence.

  Q4  Neither certificate ever becomes infeasible -- switching every link off
      is always available. So the difference is PRICE, not feasibility.
          VERIFY:  sweep --a-bar over 0, 0.5, 1, 2, 4, 8 at --systems 12
      Tabulate median links-on and median objective for each certificate.
      At which amplitude does GLOBAL switch off all communication while
      rowwise still keeps some? Report the objective gap at that point.

  Q5  Where both certify, the global design is worse. Take one system at
      --a-bar 1.0 and report whether the loss is in authority, in links, or in
      mitigation. Numbers, not adjectives.

  Q6  The attack set only AMPLIFIES existing message gains. Modify
      true_worst_rate so the attacker may switch ON a link that the design
      left off. Does either certificate still bound the true worst case?
      It does not. Name precisely which step of Part A fails, and state what
      a certificate for that larger threat would have to constrain.

  Q7  (why the code looks the way it does) Replace solve_with_threat's
      epigraph with the naive version: guess the f worst sources from the
      current z, charge only those, iterate. Show that the LP dodges the
      constraint -- report the attacked rate it returns and explain the
      mechanism in one sentence. This is a real bug that was in this file.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", type=int, default=1)
    ap.add_argument("--n-p", type=int, default=24)
    ap.add_argument("--n-a", type=int, default=12)
    ap.add_argument("--f", type=int, default=F_COMP)
    ap.add_argument("--a-bar", type=float, default=A_BAR)
    ap.add_argument("--seed", type=int, default=20260829)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--exercise", action="store_true")
    args = ap.parse_args()

    if args.verify:
        print(f"{'='*76}\nPART E --- the two identities the whole week rests on\n{'='*76}")
        err = verify_epigraph()
        print(f"  top-f epigraph identity, max error over 4000 draws : {err:.3e}")
        sys_list = [make_system(args.seed + k, args.n_p, args.n_a) for k in range(25)]
        bad, gaps = verify_ordering(sys_list, args.f)
        print(f"  ordering R_f^(i) <= R_f, violations over 25 systems : {bad}")
        print(f"  conservatism ratio on random z: median "
              f"{np.median(gaps):.2f}x   range [{min(gaps):.2f}, {max(gaps):.2f}]")
        print("\n  Both are exact statements, not approximations. If either fails,")
        print("  stop and find the bug before trusting anything else this week.")
        return

    if args.systems == 1:
        report_one(args.seed, args.n_p, args.n_a, args.f, args.a_bar)
    else:
        experiment(args.systems, args.n_p, args.n_a, args.f, args.a_bar,
                   args.seed, args.output)

    if args.exercise:
        print(EXERCISES)
    else:
        print("\n(run with --exercise to see this week's questions)")


if __name__ == "__main__":
    main()
