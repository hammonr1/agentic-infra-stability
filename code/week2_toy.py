#!/usr/bin/env python3
"""WEEK 2 --- Build the coupled model and find the loop.

Run this after you have read the Week 2 guide. It does four things:

  Part A  builds a tiny 2-sector / 2-agent system and prints the four blocks
  Part B  shows the central fact: both layers can be individually stable while
          the interconnection is unstable
  Part C  simulates the deficit over time and shows bounded vs unbounded
  Part D  sweeps authority gamma and finds where the loop tips over

Nothing here is a certificate yet. Week 2 is about seeing the loop exist.

    python week2_toy.py
    python week2_toy.py --gamma 0.3        # try your own authority level
    python week2_toy.py --exercise 1       # prints the exercise stubs
"""

from __future__ import annotations

import argparse

import numpy as np


# =============================================================================
# The model.  Deficit vector d = [d_P ; d_A]
#   d_P : normalized service deficit in each physical sector   (2 numbers)
#   d_A : action-error magnitude for each agent                (2 numbers)
#
#   d_{t+1}  <=  K d_t + xi        with    K = [[P, C],
#                                              [D, A]]
# Every entry of K is >= 0.  "<=" is entrywise.  K is an UPPER BOUND on
# propagation, not an exact dynamics -- that is what makes the argument work
# without knowing what the agents think.
# =============================================================================
def build_blocks(gamma: float = 1.0, z: float = 1.0, m: float = 0.0):
    """Return P, A, C, D for the toy city.

    Parameters
    ----------
    gamma : authority in [0,1]. How much of an agent's proposal reaches the plant.
    z     : communication intensity in [0,1]. How strongly agents' errors mix.
    m     : mitigation in [0,1]. Physical reserve; shrinks P.

    Sector 0 = power, sector 1 = transportation.
    Agent  0 = power agent, agent  1 = transportation agent.
    """
    # --- P : physical -> physical -------------------------------------------
    # Power shortfall partly becomes a transportation shortfall (charging,
    # signals) and vice versa (transport outage strands crews). Diagonal is
    # self-persistence: a deficit does not vanish in one step.
    P0 = np.array([[0.45, 0.20],
                   [0.30, 0.40]])
    P = (1.0 - 0.6 * m) * P0          # mitigation shrinks every physical gain

    # --- A : agent -> agent --------------------------------------------------
    # Diagonal: an agent's own error persists (it keeps its belief).
    # Off-diagonal: error arrives through MESSAGES, so it scales with z.
    A = np.array([[0.30, 0.50 * z],
                  [0.50 * z, 0.30]])

    # --- C : agent -> physical  (THE AUTHORITY CHANNEL) ----------------------
    # Column a is scaled by agent a's authority. At gamma = 0 this block is
    # zero and an agent error can never become a physical deficit.
    C0 = np.array([[0.384, 0.144],
                   [0.144, 0.384]])
    C = C0 * gamma

    # --- D : physical -> agent  (THE SENSING CHANNEL) ------------------------
    # A deficit is observed, and the agent cannot tell whether it caused it.
    D = np.array([[0.36, 0.12],
                  [0.12, 0.36]])
    return P, A, C, D


def assemble(P, A, C, D):
    return np.block([[P, C], [D, A]])


def rho(M):
    """Spectral radius. For a nonnegative matrix, < 1 means contraction."""
    return float(np.max(np.abs(np.linalg.eigvals(M))))


def loop_gain(P, A, C, D):
    """Gain of one trip around the loop:  agents -> plant -> sensing -> agents.

    (I-A)^-1 D (I-P)^-1 C
      ^        ^   ^      ^
      |        |   |      +-- authority  : error crosses into the plant
      |        |   +--------- physics    : deficit spreads between sectors
      |        +------------- sensing    : deficit is observed as new error
      +---------------------- messages   : error spreads between agents
    """
    nP = P.shape[0]
    nA = A.shape[0]
    return np.linalg.solve(np.eye(nA) - A,
                           D @ np.linalg.solve(np.eye(nP) - P, C))


def simulate(K, d0, xi, steps=40):
    d = d0.astype(float).copy()
    traj = [d.copy()]
    for _ in range(steps):
        d = K @ d + xi
        traj.append(d.copy())
    return np.array(traj)


# =============================================================================
def part_a(gamma, z, m):
    print("=" * 74)
    print(f"PART A --- the four blocks   (gamma={gamma}, z={z}, m={m})")
    print("=" * 74)
    P, A, C, D = build_blocks(gamma, z, m)
    names = {"P (physical -> physical)": P, "A (agent -> agent, messages)": A,
             "C (agent -> physical, AUTHORITY)": C, "D (physical -> agent, SENSING)": D}
    for n, B in names.items():
        print(f"\n{n}:\n{np.round(B, 3)}")
    K = assemble(P, A, C, D)
    print(f"\nFull coupled matrix K (4x4):\n{np.round(K, 3)}")
    print("\nEvery entry is >= 0. That is what lets us reason about the worst case")
    print("without knowing the agents' internals.")
    return P, A, C, D, K


def part_b(P, A, C, D, K):
    print("\n" + "=" * 74)
    print("PART B --- the central fact")
    print("=" * 74)
    print(f"  rho(P)  = {rho(P):.4f}   <- physical layer alone")
    print(f"  rho(A)  = {rho(A):.4f}   <- agent layer alone")
    print(f"  rho(K)  = {rho(K):.4f}   <- the two layers together")
    G = loop_gain(P, A, C, D)
    print(f"  rho(G)  = {rho(G):.4f}   <- one trip around the loop")
    ok_parts = rho(P) < 1 and rho(A) < 1
    ok_whole = rho(K) < 1
    print()
    if ok_parts and not ok_whole:
        print("  >>> BOTH LAYERS ARE STABLE ALONE, THE INTERCONNECTION IS NOT. <<<")
        print("  This is the whole motivation. Certifying the power grid and")
        print("  certifying the agent network separately proves nothing about")
        print("  the system you actually deployed.")
    elif ok_parts and ok_whole:
        print("  Everything contracts here. Raise gamma or z until it does not")
        print("  (try --gamma 1.0 --z 1.0), then re-read the sentence above.")
    else:
        print("  A layer is already unstable on its own; fix that before")
        print("  worrying about the coupling.")


def part_c(K):
    print("\n" + "=" * 74)
    print("PART C --- what it looks like over time")
    print("=" * 74)
    d0 = np.array([0.05, 0.0, 0.0, 0.0])      # one small power-sector deficit
    xi = np.array([0.01, 0.01, 0.0, 0.0])     # small constant disturbance
    traj = simulate(K, d0, xi, steps=30)
    tot = traj.sum(axis=1)
    print("  start: one small deficit in the power sector, nothing else wrong")
    print(f"\n  {'step':>5}  {'total deficit':>14}   profile")
    for t in (0, 1, 2, 3, 5, 10, 20, 30):
        bar = "#" * min(48, int(tot[t] * 12))
        print(f"  {t:>5}  {tot[t]:>14.4f}   {bar}")
    if tot[-1] > tot[0] * 3:
        print("\n  >>> It grows. The single small deficit became a cascade. <<<")
    else:
        print("\n  It settles. The loop is contractive at these settings.")


def part_d(z, m):
    print("\n" + "=" * 74)
    print("PART D --- where does it tip over?")
    print("=" * 74)
    print("  Hold messages and mitigation fixed; sweep AUTHORITY only.\n")
    print(f"  {'gamma':>6} {'rho(K)':>9} {'rho(loop)':>10}  contracts?")
    tip = None
    for g in np.linspace(0.0, 1.0, 21):
        P, A, C, D = build_blocks(g, z, m)
        K = assemble(P, A, C, D)
        r = rho(K)
        ok = r < 1.0
        if tip is None and not ok:
            tip = g
        mark = "yes" if ok else "NO"
        print(f"  {g:>6.2f} {r:>9.4f} {rho(loop_gain(P,A,C,D)):>10.4f}  {mark}")
    print()
    if tip is not None:
        print(f"  >>> The loop stops contracting at about gamma = {tip:.2f}. <<<")
        print("  That number is the thing we will learn to compute and enforce.")
        print("  Right now we found it by brute force; Week 3 computes it directly.")
    else:
        print("  Contractive at every authority level. Turn up z or turn down m.")


EXERCISES = """
EXERCISES (do these before the Week 3 meeting)

  1. Turn the message channel off (--z 0) and sweep gamma again.
     Does the loop still tip over? At what gamma? Explain in one sentence why
     switching off communication does NOT make the system safe.

  2. Turn authority off (--gamma 0) and sweep z by editing part_d.
     Now what happens? State the conclusion in the form:
     "Limiting ___ alone is not sufficient because ___."

  3. Find a pair (gamma, z) with the SAME rho(K) but different design cost,
     where cost = 0.3*m^2 + 0.1*z^2 + 0.2*(1-gamma)^2. Which would you deploy?
     This is your first taste of co-design.

  4. Set m = 0.5 (--m 0.5). How much extra authority does the mitigation buy?
     Report it as "gamma tipped from X to Y".

  5. Break the model on purpose: make D = 0 (edit build_blocks). What happens
     to rho(K), and what does that tell you about the role of sensing?
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma", type=float, default=1.0, help="authority in [0,1]")
    ap.add_argument("--z", type=float, default=1.0, help="communication intensity in [0,1]")
    ap.add_argument("--m", type=float, default=0.0, help="mitigation in [0,1]")
    ap.add_argument("--exercise", action="store_true", help="print the exercises")
    args = ap.parse_args()

    P, A, C, D, K = part_a(args.gamma, args.z, args.m)
    part_b(P, A, C, D, K)
    part_c(K)
    part_d(args.z, args.m)
    if args.exercise:
        print(EXERCISES)
    else:
        print("\n(run with --exercise to see this week's exercises)")


if __name__ == "__main__":
    main()
