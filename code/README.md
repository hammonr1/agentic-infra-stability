# Code for Week 6

NumPy and SciPy only. All five files in one folder — `week6_sharpening.py`
imports `week4_design.py`, which carries the model from Weeks 2–3.

```bash
python3 week6_sharpening.py --verify      # Part E: the two identities. RUN THIS FIRST.
python3 week6_sharpening.py               # one system, Parts A–D
python3 week6_sharpening.py --systems 40 --output out/w6.json
python3 week6_sharpening.py --exercise    # Q1–Q7, each with its command
```

Useful flags: `--f` (attack group size, default 2), `--a-bar` (amplitude,
default 1.0), `--n-p`, `--n-a`, `--seed`.

## What is new this week

| | Weeks 4–5 | Week 6 |
|---|---|---|
| threat | none — nominal matrix only | declared set: ≤ f sources amplify their own message gains by ≤ ā |
| certificate | row ratio ≤ ᾱ | row ratio + ā·top-f(loads) ≤ ᾱ |
| two variants | — | **global** `R_f` (one number for all rows) vs **rowwise** `R_f^(i)` |
| convexity | LP directly | top-f epigraph, so still an LP |

## Expected output

`--verify`: epigraph identity max error ~7e-15; ordering violations 0 of 25;
conservatism on random z median ~2.07×.

Single system (seed 20260829, f=2, ā=1.0): global `R_f` = 0.4959 vs rowwise max
0.3703, median 0.2575 — conservatism 1.93×. The **Week-4 design fails**: nominal
0.9400 but true worst 1.3103. Rowwise bound 1.3103 (exactly tight here), global
1.4359. Designing against each: rowwise keeps 8 links and γ̄ = 0.1831; global
keeps 2 links and γ̄ = 0.1563.

40 systems: conservatism mean 1.96×, median 1.88×, range [1.47, 2.84]. Both
certify 40/40 at ā = 1.0, but global costs a median +22.5% in objective.

## The bug worth knowing about

`solve_with_threat` uses the top-f **epigraph**, not a guessed worst-set. The
first version of this file guessed the f worst sources from the current z,
charged only those, and iterated — and the LP routed around it, returning
designs with attacked rate 1.3103 against a target of 0.94 while reporting
success. Exercise Q7 asks you to reproduce that failure on purpose. A constraint
built on a guessed arg-max is a hint, not a constraint.

## Do not

- Do not raise `ALPHA_BAR` to make an attacked design feasible. Neither
  certificate ever goes infeasible — switching all links off is always
  available — so an infeasibility you see is a bug, not a reason to relax ᾱ.
- Do not conclude the rowwise bound is always tight because it is tight on the
  default seed. Equation (3) is exact per row; the brute force maximises over
  rows *and* subsets. Check, don't assume.
