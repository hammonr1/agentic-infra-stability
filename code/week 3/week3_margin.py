#!/usr/bin/env python3
"""WEEK 3 --- Stop brute-forcing. Compute the margin, then enforce it.

Week 2 found the tipping authority by sweeping. That does not scale: real
designs have hundreds of variables and you cannot sweep them all. This week you
replace the sweep with two things:

  Part A  a WITNESS v > 0 that turns "is rho(K) < 1?" into a set of
          LINEAR INEQUALITIES you can check by hand
  Part B  the same inequalities read as a BUDGET: each row says how much
          margin messages and authority may spend
  Part C  the largest admissible authority, computed in closed form --- no sweep
  Part D  a first fallback: when the budget is exceeded, what do you do?

The point of the witness is that it converts a spectral question (eigenvalues,
nonlinear, global) into row-by-row arithmetic (linear, local, and convex in the
design variables). That is what makes the whole approach optimizable later.

    python week3_margin.py
    python week3_margin.py --exercise
"""

from __future__ import annotations

import argparse

import numpy as np

from week2_toy import assemble, build_blocks, rho, simulate

TARGET = 0.94          # alpha_bar: the contraction level we insist on


# =============================================================================
def perron_witness(K, iters=500):
    """A positive vector v with K v ~ rho(K) v (the Perron vector).

    Power iteration. For a nonnegative primitive matrix this converges to the
    eigenvector of the largest eigenvalue, and it is the BEST witness in the
    sense of Part A below.
    """
    v = np.ones(K.shape[0])
    for _ in range(iters):
        w = K @ v
        if not np.all(w > 0):
            return v
        v = w / w.max()
    return v


def fallback_witness(K0, alpha=TARGET):
    """A STRICTLY POSITIVE witness for a design you already trust.

    Why not the Perron vector? If authority is off, K0 is block lower-triangular
    and its Perron vector has ZEROS in the physical components -- you cannot
    divide by it. The fix is the resolvent:

        v = (alpha I - K0)^{-1} 1      which satisfies   K0 v = alpha v - 1 < alpha v

    For a nonnegative K0 with rho(K0) < alpha, the inverse is a nonnegative
    matrix with positive diagonal, so v > 0 entrywise and the inequality is
    strict. This is the standard way to get a usable witness for a fallback.
    """
    n = K0.shape[0]
    v = np.linalg.solve(alpha * np.eye(n) - K0, np.ones(n))
    assert np.all(v > 0), "rho(K0) must be below alpha for this to work"
    return v / v.max()


def row_ratio(K, v):
    """max_i (K v)_i / v_i.   Always >= rho(K), with equality at the Perron vector.

    This is the Collatz--Wielandt bound. It is the single most useful fact in
    this project: it lets you CERTIFY contraction with arithmetic instead of
    eigenvalues.
    """
    return float(np.max(K @ v / v))


# =============================================================================
def part_a(gamma, z, m):
    print("=" * 74)
    print("PART A --- the witness: eigenvalues out, arithmetic in")
    print("=" * 74)
    P, A, C, D = build_blocks(gamma, z, m)
    K = assemble(P, A, C, D)
    print("Claim: if you can find ANY v > 0 with  K v <= alpha * v  entrywise,")
    print("       then rho(K) <= alpha. No eigenvalue computation needed.\n")

    for name, v in (("all ones (naive guess)", np.ones(4)),
                    ("Perron vector (refined)", perron_witness(K))):
        r = row_ratio(K, v)
        print(f"  witness = {name}")
        print(f"    v            = {np.round(v/v.max(), 3)}")
        print(f"    max (Kv)_i/v_i = {r:.4f}      rho(K) = {rho(K):.4f}")
        print(f"    gap to truth   = {r - rho(K):.4f}"
              f"{'   <-- tight' if r - rho(K) < 1e-6 else '   <-- conservative'}\n")
    print("Lesson: a bad witness makes a TRUE statement that is needlessly")
    print("pessimistic. Refining the witness costs one matrix-vector product and")
    print("buys back real margin. You will do this in every experiment.")
    return P, A, C, D, K


def part_b(P, A, C, D):
    print("\n" + "=" * 74)
    print("PART B --- the same inequalities, read as a budget")
    print("=" * 74)
    K0 = assemble(P, A, np.zeros_like(C), D)      # authority switched off
    # ANCHOR THE WITNESS AT A DESIGN YOU ALREADY TRUST: authority off.
    # C = 0 makes K0 block lower-triangular, so rho(K0) = max(rho(P), rho(A))
    # and we know it is safe. Two traps here, both worth remembering:
    #   (a) anchoring at the gamma = 1 design gives a witness that is wrong for
    #       the rows we are budgeting -- Part C would declare infeasibility;
    #   (b) the Perron vector of a triangular K0 has ZEROS, so you cannot divide
    #       by it. Use the resolvent instead (see fallback_witness).
    v = fallback_witness(K0)
    dC = np.zeros_like(K0)
    nP = P.shape[0]
    dC[:nP, nP:] = C / max(np.max(C), 1e-12)      # unit authority direction

    print(f"rho(K0) with authority OFF = {rho(K0):.4f}  (safe anchor)")
    print(f"Witness v = {np.round(v, 3)}   target alpha_bar = {TARGET}\n")
    print("Row i of  K v <= alpha_bar * v  says:")
    print("     already spent        +   per-unit authority cost  <=  budget")
    print(f"\n  {'row':>4} {'(K0 v)_i/v_i':>13} {'(dC v)_i/v_i':>14} {'alpha_bar':>10} {'slack':>9}")
    base = K0 @ v / v
    per = dC @ v / v
    for i in range(len(v)):
        lab = f"P{i}" if i < nP else f"A{i-nP}"
        print(f"  {lab:>4} {base[i]:>13.4f} {per[i]:>14.4f} {TARGET:>10.2f} {TARGET-base[i]:>9.4f}")
    print("\nThis is the 'shared safety account' from the slides, made literal.")
    print("Every row is one account. Authority and messages withdraw from it;")
    print("mitigation deposits into it. A design is admissible iff NO row goes")
    print("negative. Notice the constraint is LINEAR in the design variables ---")
    print("that is why the full problem will be convex.")
    return v, base, per


def part_c(base, per, z, m):
    print("\n" + "=" * 74)
    print("PART C --- largest admissible authority, in closed form")
    print("=" * 74)
    active = per > 1e-12
    if np.any(base[~active] > TARGET + 1e-12):
        print("  Infeasible even at zero authority: fix the plant first.")
        return None
    ratios = (TARGET - base[active]) / per[active]
    eta = float(np.min(ratios))
    binding = int(np.argmin(ratios))
    print("  For each row, the most authority it can afford is")
    print("        (alpha_bar - already_spent) / per_unit_cost")
    print("  and the design must satisfy ALL rows, so take the minimum:\n")
    for k, r in enumerate(ratios):
        print(f"    row {k}: {r:>8.4f}{'   <-- binding (the tightest row)' if k == binding else ''}")
    print(f"\n  >>> gamma* = {eta:.4f}   (one min over rows; no sweeping) <<<")

    # verify against Week 2's brute-force answer
    tip = None
    for g in np.linspace(0, 1, 2001):
        P, A, C, D = build_blocks(g, z, m)
        if rho(assemble(P, A, C, D)) > TARGET:
            tip = g
            break
    print(f"  brute-force check (2001 points): tips at gamma = {tip:.4f}")
    print(f"  closed form is {'conservative by' if eta <= tip else 'WRONG, larger than'}"
          f" {abs(tip - eta):.4f}")
    print("\n  The closed form is SAFE (never over-admits) but pessimistic, because")
    print("  this witness was computed at a different authority level. Fixing that")
    print("  --- iterating witness and margin together --- is a later topic.")
    return eta


def part_d(eta, z, m):
    print("\n" + "=" * 74)
    print("PART D --- a first fallback")
    print("=" * 74)
    print("Suppose the agents ask for more authority than gamma*. You must refuse.")
    print("But 'refuse' has to mean something concrete and safe.\n")
    asked = 0.9
    granted = min(asked, eta if eta is not None else 0.0)
    print(f"  agents request gamma = {asked:.2f}")
    print(f"  certificate allows   gamma = {granted:.4f}")
    print(f"  action: execute the fallback-blended command with gamma = {granted:.4f}")
    print("          (see the blend equation in the Q&A note)\n")

    for label, g in (("what they asked for", asked), ("what we grant", granted)):
        P, A, C, D = build_blocks(g, z, m)
        K = assemble(P, A, C, D)
        tr = simulate(K, np.array([0.05, 0, 0, 0]), np.array([0.01, 0.01, 0, 0]), 30)
        tot = tr.sum(axis=1)
        print(f"  {label:22s} gamma={g:.4f}  rho={rho(K):.4f}  "
              f"deficit at t=30: {tot[-1]:.4f}")
    print("\n  Same disturbance, same plant. The only difference is how much of the")
    print("  agents' proposal we let through. That single number is the product.")


EXERCISES = """
EXERCISES (bring answers to the Week 4 meeting)

  1. Recompute Part C with the all-ones witness instead of the Perron vector.
     How much authority do you lose by using a lazy witness? Report both numbers.

  2. In Part B, which row is binding? Is it a physical row or an agent row?
     Change m (mitigation) until a DIFFERENT row becomes binding, and report
     the m at which the switch happens. Explain what that means physically.

  3. Mitigation deposits into the account. Compute gamma* for m = 0, 0.2, 0.4,
     0.6 and plot gamma* against m. Is it linear? Should it be?

  4. Write down, in one line each, why the closed form in Part C can never
     over-admit authority, and why it can under-admit. (Hint: Collatz--Wielandt
     is an inequality in one direction only.)

  5. Design question, no code: you have budget for EITHER +0.2 mitigation OR
     turning one message link off. Using Part B's table, which buys more
     authority? State how you decided.

  6. Stretch: the witness in Part C was computed at one authority level but used
     for all of them. Sketch an iteration that fixes this. (You do not need it
     to work yet -- just describe the loop in three steps.)
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--z", type=float, default=1.0)
    ap.add_argument("--m", type=float, default=0.0)
    ap.add_argument("--exercise", action="store_true")
    args = ap.parse_args()

    P, A, C, D, K = part_a(args.gamma, args.z, args.m)
    v, base, per = part_b(P, A, C, D)
    eta = part_c(base, per, args.z, args.m)
    part_d(eta, args.z, args.m)
    if args.exercise:
        print(EXERCISES)
    else:
        print("\n(run with --exercise to see this week's exercises)")


if __name__ == "__main__":
    main()
