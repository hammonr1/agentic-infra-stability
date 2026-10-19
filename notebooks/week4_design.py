#!/usr/bin/env python3
"""WEEK 4 --- Stop choosing one variable. Choose all of them, optimally.

Week 3 computed the largest authority gamma* for a FIXED mitigation and message
graph. That is still a one-variable answer. This week you choose all three
families jointly -- mitigation per sector, intensity per candidate link,
authority per agent -- at minimum cost, subject to the same row inequalities.

The key fact carried over from Week 3: at a fixed witness v, every row
inequality is LINEAR in the design variables. So the joint problem is a linear
program. Not a heuristic, not a sweep -- an LP with a certificate attached.

  Part A   build a medium system (24 sectors, 12 agents, ~90 candidate links)
  Part B   assemble the LP: variables, cost, row constraints
  Part C   solve it; compare against Week 3's one-variable answer
  Part D   the price of safety: sweep the contraction target alpha_bar

    python week4_design.py                 # one system, verbose
    python week4_design.py --systems 30    # the medium experiment
    python week4_design.py --exercise
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

ALPHA_BAR = 0.94
KAPPA_M = 0.60          # mitigation effectiveness: P = (1 - KAPPA_M * m) * P0


# ===========================================================================
# Part A --- a medium system
# ===========================================================================
def make_system(seed: int, n_p: int = 24, n_a: int = 12):
    """A medium interdependent system with a candidate communication graph.

    Returns the four building blocks plus the candidate edge list. Compare with
    Week 2's 2x2 blocks: same structure, bigger, and now the message graph is a
    *list of candidate links* each with its own on/off intensity.
    """
    rng = np.random.default_rng(seed)

    # ---- P0: physical interdependency, sparse, nonnegative ----
    P0 = rng.random((n_p, n_p)) * (rng.random((n_p, n_p)) < 0.14)
    np.fill_diagonal(P0, 0.18 + 0.12 * rng.random(n_p))      # self-persistence
    P0 *= 0.62 / max(np.max(np.abs(np.linalg.eigvals(P0))), 1e-9)

    # ---- agent self-retention (diagonal of A; not switchable) ----
    a_diag = 0.26 + 0.10 * rng.random(n_a)

    # ---- candidate message links: (target, source, weight) ----
    cand = [(t, s) for t in range(n_a) for s in range(n_a)
            if s != t and rng.random() < 0.65]
    edge_t = np.array([e[0] for e in cand])
    edge_s = np.array([e[1] for e in cand])
    edge_w = 0.10 + 0.30 * rng.random(len(cand))

    # ---- C0: authority coupling, agent a -> sector i ----
    C0 = rng.random((n_p, n_a)) * (rng.random((n_p, n_a)) < 0.30)
    C0 *= 0.42 / max(C0.max(), 1e-9)

    # ---- D: sensing, sector i -> agent a. NOT a decision variable ----
    D = rng.random((n_a, n_p)) * (rng.random((n_a, n_p)) < 0.25)
    D *= 0.34 / max(D.max(), 1e-9)

    return dict(P0=P0, a_diag=a_diag, edge_t=edge_t, edge_s=edge_s,
                edge_w=edge_w, C0=C0, D=D, n_p=n_p, n_a=n_a, n_e=len(cand))


def assemble(sys_, m, z, g):
    """Build the full (n_p+n_a) x (n_p+n_a) comparison matrix K(m, z, gamma)."""
    n_p, n_a = sys_["n_p"], sys_["n_a"]
    P = (1.0 - KAPPA_M * m)[:, None] * sys_["P0"]
    A = np.diag(sys_["a_diag"]).astype(float)
    A[sys_["edge_t"], sys_["edge_s"]] = sys_["edge_w"] * z
    C = sys_["C0"] * g[None, :]
    return np.block([[P, C], [sys_["D"], A]])


def rho(M):
    return float(np.max(np.abs(np.linalg.eigvals(M))))


def fallback_witness(sys_, alpha=ALPHA_BAR):
    """Strictly positive witness anchored at the design we trust: authority off,
    full mitigation, no messages. Resolvent construction from Week 3."""
    n_p, n_a = sys_["n_p"], sys_["n_a"]
    K0 = assemble(sys_, np.ones(n_p), np.zeros(sys_["n_e"]), np.zeros(n_a))
    n = n_p + n_a
    r = rho(K0)
    if r >= alpha:
        raise RuntimeError(f"fallback itself does not contract: rho={r:.3f}")
    v = np.linalg.solve(alpha * np.eye(n) - K0, np.ones(n))
    return v / v.max()


# ===========================================================================
# Part B --- the LP
# ===========================================================================
def build_lp(sys_, v, alpha_bar=ALPHA_BAR, cost=None):
    """Assemble  min c'x  s.t.  A_ub x <= b_ub,  0 <= x <= 1.

    Variables  x = [ m (n_p) | z (n_e) | gamma (n_a) ].

    Row i of  K(x) v <= alpha_bar * v  expands as follows.

    PHYSICAL row i (i = 0..n_p-1):
        sum_j (1 - KAPPA_M m_i) P0[i,j] v_j  +  sum_a C0[i,a] gamma_a v_{n_p+a}
            <= alpha_bar v_i
      The mitigation term is linear in m_i with a NEGATIVE coefficient
      (mitigation buys margin); the authority term is linear in gamma with a
      positive coefficient (authority spends it).

    AGENT row a (a = 0..n_a-1):
        sum_j D[a,j] v_j  +  a_diag[a] v_{n_p+a}
            + sum_{edges e with target a} w_e z_e v_{n_p + src(e)}
            <= alpha_bar v_{n_p+a}
      Only the message terms are decisions here; sensing and self-retention are
      fixed costs the row must absorb before anything is granted.
    """
    n_p, n_a, n_e = sys_["n_p"], sys_["n_a"], sys_["n_e"]
    nx = n_p + n_e + n_a
    vP, vA = v[:n_p], v[n_p:]

    A_ub = np.zeros((n_p + n_a, nx))
    b_ub = np.zeros(n_p + n_a)

    # --- physical rows ---
    P0v = sys_["P0"] @ vP                       # (n_p,)
    for i in range(n_p):
        A_ub[i, i] = -KAPPA_M * P0v[i]          # mitigation: negative coeff
        A_ub[i, n_p + n_e:] = sys_["C0"][i] * vA
        b_ub[i] = alpha_bar * vP[i] - P0v[i]

    # --- agent rows ---
    Dv = sys_["D"] @ vP                         # (n_a,)
    for a in range(n_a):
        b_ub[n_p + a] = alpha_bar * vA[a] - Dv[a] - sys_["a_diag"][a] * vA[a]
    for e in range(n_e):
        t, s = sys_["edge_t"][e], sys_["edge_s"][e]
        A_ub[n_p + t, n_p + e] += sys_["edge_w"][e] * vA[s]

    # --- the objective ---------------------------------------------------
    # Three terms, and the signs are the whole story:
    #   mitigation  costs money                        -> positive coefficient
    #   a link      carries information worth having,
    #               net of its privacy/bandwidth price -> negative coefficient
    #   authority   is the thing we want to grant       -> negative coefficient
    # Links and authority both BUY something and both SPEND margin. That is the
    # tension the LP resolves, and it is why the answer is a subset of links
    # rather than all or none.
    if cost is None:
        cost = dict(m=1.0, z_price=0.20, z_value=0.55, gamma=2.0)
    c = np.concatenate([
        np.full(n_p, cost["m"]),
        np.full(n_e, cost["z_price"] - cost["z_value"]),
        np.full(n_a, -cost["gamma"]),
    ])
    return c, A_ub, b_ub


def solve_joint(sys_, v, alpha_bar=ALPHA_BAR, cost=None):
    c, A_ub, b_ub = build_lp(sys_, v, alpha_bar, cost)
    nx = len(c)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0.0, 1.0)] * nx,
                  method="highs")
    if not res.success:
        return None
    n_p, n_e = sys_["n_p"], sys_["n_e"]
    x = res.x
    return dict(m=x[:n_p], z=x[n_p:n_p + n_e], g=x[n_p + n_e:],
                obj=float(res.fun), status=res.status)


# ===========================================================================
# Part C --- compare with Week 3's one-variable answer
# ===========================================================================
def week3_gamma_star(sys_, v, m_fixed, z_fixed, alpha_bar=ALPHA_BAR):
    """Week 3's closed form, for the SAME witness and a fixed (m, z)."""
    n_p, n_a = sys_["n_p"], sys_["n_a"]
    vP, vA = v[:n_p], v[n_p:]
    K0 = assemble(sys_, m_fixed, z_fixed, np.zeros(n_a))
    base = (K0 @ v) / v
    dC = np.zeros_like(K0)
    dC[:n_p, n_p:] = sys_["C0"]
    per = (dC @ v) / v
    act = per > 1e-12
    if np.any(base[~act] > alpha_bar + 1e-9):
        return 0.0
    return float(np.min((alpha_bar - base[act]) / per[act]))


def report_one(sys_, v, seed):
    n_p, n_a, n_e = sys_["n_p"], sys_["n_a"], sys_["n_e"]
    print(f"\n{'='*74}\nPART A/B --- system seed {seed}: "
          f"{n_p} sectors, {n_a} agents, {n_e} candidate links")
    print(f"{'='*74}")
    print(f"  design variables: {n_p} mitigations + {n_e} intensities + "
          f"{n_a} authorities = {n_p+n_e+n_a}")
    print(f"  row constraints : {n_p} physical + {n_a} agent = {n_p+n_a}")
    print(f"  a sweep over {n_p+n_e+n_a} variables at 10 points each would be "
          f"10^{n_p+n_e+n_a} designs.")

    sol = solve_joint(sys_, v)
    if sol is None:
        print("  LP infeasible at this witness."); return None
    K = assemble(sys_, sol["m"], sol["z"], sol["g"])
    print(f"\nPART C --- the LP solution")
    print(f"  mean mitigation      m = {sol['m'].mean():.4f}")
    print(f"  links switched on      = {int((sol['z'] > 0.5).sum())} / {n_e}"
          f"   (mean intensity {sol['z'].mean():.4f})")
    print(f"  mean authority   gamma = {sol['g'].mean():.4f}"
          f"   (min {sol['g'].min():.4f}, max {sol['g'].max():.4f})")
    print(f"  certified: max row ratio {float(np.max(K @ v / v)):.6f} "
          f"<= {ALPHA_BAR}   true rho(K) = {rho(K):.4f}")

    # Week 3 comparison: same witness, mitigation and links frozen at the LP's
    # own choice, then the single scalar gamma from the closed form.
    g3 = week3_gamma_star(sys_, v, sol["m"], sol["z"])
    print(f"\n  Week 3 (one scalar authority, same m and z): gamma* = {g3:.4f}")
    print(f"  Week 4 (per-agent authority, jointly chosen): mean = {sol['g'].mean():.4f}")
    if g3 > 1e-9:
        print(f"  ratio = {sol['g'].mean()/g3:.2f}x more authority admitted")
    return sol


# ===========================================================================
# Part D --- the price of safety
# ===========================================================================
def price_of_safety(sys_, v, targets=(0.99, 0.97, 0.94, 0.90, 0.86)):
    print(f"\n{'='*74}\nPART D --- the price of safety\n{'='*74}")
    print(f"  {'alpha_bar':>10} {'mean m':>9} {'links on':>9} "
          f"{'mean gamma':>11} {'LP objective':>13}")
    rows = []
    for ab in targets:
        try:
            vv = fallback_witness(sys_, alpha=min(ab, 0.995))
        except RuntimeError:
            print(f"  {ab:>10.2f}   fallback does not contract at this target")
            continue
        s = solve_joint(sys_, vv, alpha_bar=ab)
        if s is None:
            print(f"  {ab:>10.2f}   infeasible"); continue
        print(f"  {ab:>10.2f} {s['m'].mean():>9.4f} "
              f"{int((s['z']>0.5).sum()):>9d} {s['g'].mean():>11.4f} "
              f"{s['obj']:>13.4f}")
        rows.append(dict(alpha_bar=ab, m=float(s['m'].mean()),
                         links=int((s['z'] > 0.5).sum()),
                         gamma=float(s['g'].mean()), obj=s['obj']))
    print("\n  Tightening the contraction target buys safety by spending")
    print("  mitigation and refusing authority. That trade is the whole design.")
    return rows


EXERCISES = """
EXERCISES (bring to the Week 5 meeting)

  1. In Part C, which agents received the most authority? Cross-check against
     the columns of C0: is the LP granting authority where the authority
     coupling is weakest? Show three numbers that support your answer.

  2. The LP gives per-agent authority; Week 3 gave one scalar for all agents.
     Explain in two sentences why the per-agent version admits more total
     authority even though both satisfy the same row constraints.

  3. Set cost['gamma'] = 0.5 and rerun. What happens to the number of links
     switched on? Explain the mechanism, not just the direction.

  4. Run --systems 30 and report the distribution of (mean gamma, links on).
     Is the number of active links concentrated or spread? Give the IQR.

  5. Pick one system. Remove the mitigation variables (fix m = 0) and re-solve.
     How much authority is lost? This is the dollar value of physical reserve,
     in units of admitted authority.

  6. The LP is solved at ONE witness, anchored at the fallback. Predict, before
     running Week 5: will refining the witness increase or decrease the
     admitted authority, and by roughly how much? Write your prediction down.
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
        sys_ = make_system(args.seed, args.n_p, args.n_a)
        v = fallback_witness(sys_)
        report_one(sys_, v, args.seed)
        price_of_safety(sys_, v)
    else:
        print(f"{'='*74}\nMEDIUM EXPERIMENT --- {args.systems} systems, "
              f"{args.n_p} sectors, {args.n_a} agents\n{'='*74}")
        recs = []
        for k in range(args.systems):
            sd = args.seed + k
            sys_ = make_system(sd, args.n_p, args.n_a)
            try:
                v = fallback_witness(sys_)
            except RuntimeError:
                continue
            s = solve_joint(sys_, v)
            if s is None:
                continue
            K = assemble(sys_, s["m"], s["z"], s["g"])
            g3 = week3_gamma_star(sys_, v, s["m"], s["z"])
            recs.append(dict(seed=sd, mean_m=float(s["m"].mean()),
                             links_on=int((s["z"] > 0.5).sum()),
                             n_e=sys_["n_e"],
                             mean_gamma=float(s["g"].mean()),
                             week3_gamma=float(g3),
                             row_ratio=float(np.max(K @ v / v)),
                             rho=rho(K)))
        import statistics as st
        g = [r["mean_gamma"] for r in recs]
        r3 = [r["week3_gamma"] for r in recs if r["week3_gamma"] > 1e-9]
        gap = [r["row_ratio"] - r["rho"] for r in recs]
        print(f"  solved {len(recs)}/{args.systems} systems")
        print(f"  mean authority admitted : median {st.median(g):.4f}  "
              f"IQR [{np.percentile(g,25):.4f}, {np.percentile(g,75):.4f}]")
        if r3:
            ratio = [r["mean_gamma"]/r["week3_gamma"] for r in recs
                     if r["week3_gamma"] > 1e-9]
            print(f"  vs Week 3 single scalar : median {st.median(ratio):.2f}x")
        print(f"  links on                : median "
              f"{st.median([r['links_on'] for r in recs]):.0f} of "
              f"{st.median([r['n_e'] for r in recs]):.0f}")
        print(f"  all certified at the witness: "
              f"{all(r['row_ratio'] <= ALPHA_BAR + 1e-9 for r in recs)}")
        print(f"  certificate slack (row ratio - rho): median "
              f"{st.median(gap):.4f}   <-- Week 5 closes this")
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(recs, indent=2))
            print(f"  written to {args.output}")

    if args.exercise:
        print(EXERCISES)
    else:
        print("\n(run with --exercise to see this week's exercises)")


if __name__ == "__main__":
    main()
