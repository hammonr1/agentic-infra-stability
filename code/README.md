# Toy code for Weeks 2 and 3

Two scripts, NumPy only. Read the week guide first — the code is there to make
the guide concrete, not the other way round.

```bash
python3 -c "import numpy; print(numpy.__version__)"   # that is all you need

python3 week2_toy.py                 # Week 2 worked example
python3 week2_toy.py --z 0           # Week 2, Exercise 1
python3 week2_toy.py --m 0.5         # Week 2, Exercise 2

python3 week3_margin.py              # Week 3 worked example
python3 week3_margin.py --m 0.5      # Week 3, Exercise 2
```

`week3_margin.py` imports `week2_toy.py`, so keep both files in the same folder.

## The toy city

Two physical sectors (0 = power, 1 = transportation) and two agents (0 = power
agent, 1 = transportation agent). Four blocks:

| Block | Direction | Set by |
|---|---|---|
| `P(m)` | physical → physical | mitigation `m`: `P = (1 − 0.6 m) P0` |
| `A(z)` | agent → agent (messages) | intensity `z` scales the **off-diagonals only** |
| `C(γ)` | agent → physical (**authority**) | `C = C0 · diag(γ)`; at γ = 0 this block is zero |
| `D` | physical → agent (sensing) | **not a decision** — agents observe the plant |

Flags `--gamma`, `--z`, `--m` all take values in [0, 1].

## What each script shows

**`week2_toy.py`**

| Part | Point |
|---|---|
| A | the four blocks and the assembled 4×4 `K` |
| B | ρ(P)=0.6712, ρ(A)=0.8000 — both layers stable — yet ρ(K)=1.2440 |
| C | one 5% power deficit growing 1700× in thirty steps |
| D | authority swept by brute force; tips at γ ≈ 0.30 |

**`week3_margin.py`**

| Part | Point |
|---|---|
| A | same true bound, two witnesses: the lazy one wastes 0.036 of margin |
| B | each row of `K v ≤ ᾱ v` as *already spent + per-unit cost ≤ budget* |
| C | γ\* = 0.0364 from one formula, vs a true tipping point of 0.1480 |
| D | what refusing means: same plant, 211× less deficit |

## Sanity check

```bash
python3 week2_toy.py --gamma 0
```

Must print `rho(K) = 0.8000` and report contraction. With authority off, `C = 0`,
`K` is block lower-triangular, so ρ(K) can only be max{ρ(P), ρ(A)} = max{0.6712,
0.8000}. Anything else is a bug worth chasing.

## Two traps the code walks into on purpose

Both are explained in the source comments. Neither is a mistake.

1. **Anchoring.** Build the witness at γ = 1 and use it to budget the γ = 0 rows,
   and Part C declares the problem infeasible. Anchor the witness at a design you
   already trust — here, authority off.
2. **Degenerate witnesses.** With `C = 0`, `K0` is block triangular and its
   Perron vector has *zeros* in the physical entries, so you cannot divide by it.
   Use the resolvent `v = (ᾱ I − K0)^{-1} 1`, strictly positive whenever
   ρ(K0) < ᾱ. This is how a usable fallback witness is built in general.

## Do not

- **Do not enlarge the toy.** A 4×4 matrix you fully understand beats a 400-node
  model you do not. Size comes in Month 2, after the certificate exists.
- **Do not add a solver yet.** Week 4 introduces the convex program; adding one
  now hides the structure you are meant to be learning.
- **Do not loosen ᾱ = 0.94** to make Part C's conservatism go away. The safety
  level is declared first. Closing that gap honestly is the Week 6–7 topic.
