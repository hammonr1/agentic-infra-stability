---
title: "The proofs"
---

**Notation.** Fix a design and a witness v.

**Loads.** For agent row i and sender s, the load b[i,s] is the sum over s's links into i of w[e]·z[e]·v[s]/v[i]. It is nonnegative and linear in z.

**Worst-row load.** ℓ[s] is the largest of sender s's loads over all rows.

**Top-f sum.** For any vector x, top_f(x) is the sum of its f largest entries.

**The two charges.** The one-number charge is R = top_f(ℓ). Row i's own charge is R(i) = top_f(b[i,·]).

*Figures are illustrations on small made-up examples (f = 2); they are not outputs of the verification code. `figures/make_figs.py` regenerates them.*

# Q1: row by row never exceeds one number

**Lemma 1.** If x ≤ y entry by entry, then top_f(x) ≤ top_f(y).

*Proof.* Let S be the f positions holding the largest entries of x. Then top_f(x) is the sum of x over S, which is at most the sum of y over S. That is a sum of some f entries of y, so it is at most top_f(y). ∎

**Theorem.** R(i) ≤ R for every row i.

*Proof.* ℓ[s] is a maximum over rows, so b[i,s] ≤ ℓ[s] for every sender. Row i's loads therefore sit entry by entry below ℓ. Lemma 1 then gives top_f(b[i,·]) ≤ top_f(ℓ). ∎

![**Figure 1.** (a) A 4×5 table of loads. The shaded cell in each column sets ℓ[s]; the right-hand column is each row's charge R(i), and R = top₂(ℓ) = 11. (b) Row 3 drawn inside the ℓ bars: b[3,s] ≤ ℓ[s] at every sender, so top₂(b[3,·]) = 9 ≤ 11.](figures/fig1_row_by_row.png)

What would have to break is the single inequality b[i,s] ≤ ℓ[s], and a true maximum can't be exceeded. So a violation means ℓ and b were built from different inputs. That could be another design, another witness, another f, or a maximum taken along the wrong axis. Lemma 1 never used nonnegativity, so negative loads wouldn't be a loophole either. That's why Hadi's --verify treats a violation as a bug.

Row by row is exact, not just valid. This is the argument behind the second correction.

Let n(i) be row i's unattacked ratio (Kv)(i)/v(i). Attacking with group B raises it to n(i) plus the sum over s in B of a[s]·b[i,s]. Loads are nonnegative, so the attacker does best with all f agents at full strength ā. The true worst case is therefore:

> T = max over groups B of max over rows i of n(i) + ā · Σ_{s in B} b[i,s]

Both maxima run over one fixed list of pairs (B, i), so they can be taken in either order. Put rows outside and groups inside. For each row, the inner maximum picks that row's f heaviest senders, which gives exactly top_f(b[i,·]). So T equals the maximum over i of n(i) + ā·R(i), which is the row-by-row bound. ∎

The intuition is simple. The worst group for the worst row is itself a legal attack, so letting each row pick its own worst group loses nothing. The handout worries that the brute force maximises "over rows and subsets jointly." A joint maximum over a list is the same as maximising one variable at a time.

What can still be loose is the witness. By Collatz–Wielandt, for nonnegative K and positive v, ρ(K) is at most the largest row ratio (Kv)(i)/v(i). Equality holds only at the Perron vector. Hadi's witness was built for the unattacked fallback design, so its 1.3103 overstates the true 0.9424. That gap is exactly my Finding 1 mistake.

# Q2: the top-f identity

The claim:

> top_f(x) = min over τ, and r ≥ 0 with r[s] ≥ x[s] − τ, of f·τ + Σ_s r[s]

Read τ as a threshold and r[s] as how far x[s] pokes above it.

Every feasible choice is at least top_f(x). Let S be the f largest positions and write each x[s] as τ + (x[s] − τ). Summing over S and using r[s] ≥ x[s] − τ gives top_f(x) ≤ f·τ + Σ over S of r[s]. Adding the remaining r's can only raise the right side, since they are nonnegative.

One choice hits it exactly. Set τ to the f-th largest entry and r[s] = max(x[s] − τ, 0). Only top-f entries have positive excess, so f·τ + Σ r[s] equals the sum of the top f entries. Ties change nothing. ∎

![**Figure 2.** x = (7, 5, 4, 2, 1), f = 2. (a) With τ = 5 (the 2nd-largest entry), only the top entries poke above τ: f·τ + Σr = 2·5 + 2 = 12 = 7 + 5. (b) The cost f·τ + Σ_s max(x[s] − τ, 0) as a function of τ never goes below 12; here the minimum is attained for any τ between the 2nd and 3rd largest entries, i.e. τ ∈ [4, 5].](figures/fig2_topf_identity.png)

This keeps the problem a linear program. The certificate needs n(i) + ā·top_f(b[i,·]) ≤ ᾱ. A minimum on the small side of "≤" turns into "there exist τ and r such that." So τ(i) and r[i,·] become ordinary LP variables with these constraints:

> r[i,s] ≥ b[i,s] − τ(i) and r ≥ 0
>
> n(i) + ā·(f·τ(i) + Σ r[i,s]) ≤ ᾱ

Since b is linear in z, every constraint is linear, and the LP picks the worst group itself.

Without the identity, top_f is a maximum over all groups of f senders. A maximum on the small side of "≤" means "for every group," one linear constraint per group. That's 66 per row here, but about 1.7 × 10¹³ for 100 agents with 10 compromised. The only shortcut is guessing the group, which is the Q7 bug. If you've seen LP duality, here's another view. top_f(x) is the maximum of λ·x over 0 ≤ λ ≤ 1 with Σλ = f, and the identity is that LP's dual.

# Q3: when the ratio is exactly 1

**Lemma 2.** R(i) = R exactly when some group S of f senders does two things. It achieves top_f(ℓ), and b[i,s] = ℓ[s] for every s in S. In words, row i is the worst-hit row for each of the f heaviest senders.

*Proof.* Suppose such an S exists. Then R(i) is at least the sum of b[i,s] over S, which equals the sum of ℓ[s] over S, which is R. The theorem gives the other direction. Conversely, let S be row i's f heaviest senders. Then R = R(i) = Σ over S of b[i,s] ≤ Σ over S of ℓ[s] ≤ R. Both inequalities must be equalities, which is the condition. ∎

The ratio divides R by the median row charge. Every row charge is at most R, so the median reaches R only if more than half the rows do. With 12 rows, that means at least 7. Combined with Lemma 2, the ratio is 1 exactly when a majority of agent rows are each hit hardest by the same f heaviest senders. That's the one-sentence structure for the hand-in.

**Symmetric city.** Every agent hears from every other at the same load c. Every ℓ[s] is c, R is f·c, and every row also reaches f·c, so the ratio is exactly 1. It's fragile. Making one link 10% heavier lifts R to 2.1c while 11 rows stay at 2c, giving 1.05.

**Two broadcasters.** Two agents send to everyone at load c, and all other loads are below c. The ten other rows each see both broadcasters, so they reach R = 2c. Only the broadcasters' own rows fall short, since no agent messages itself. Ten of twelve is a majority, so the ratio stays exactly 1 however the lighter links are arranged.

In random cities each sender's heaviest load tends to land on a different row. So no row collects two heavy senders' maxima, which is why the ratio never reached 1. Table H reproduces all three constructions with Hadi's functions.

![**Figure 3.** Scaled-down 6-agent versions of the three constructions (f = 2; the proof's 12-row case works the same way with a majority of 7). Left: loads, with an orange box on the cell that sets ℓ[s]. Right: row charges R(i) against R; blue bars reach R. (a) Symmetric city and (b) two broadcasters (4 of 6 rows reach R, a majority): ratio exactly 1. (c) A random city: the maxima land on different rows, no row reaches R, and the ratio is above 1 (1.27 for this seed).](figures/fig3_ratio_equals_one.png)

# Q6: which step fails, and why no design survives

§2.2 substitutes the attack into row i and gets an extra cost equal to the sum over s in B of a[s]·b[i,s]. That extra is proportional to the design's own link intensities, so a link left off carries no attack. Everything downstream leans on this, including the fact that switching every link off always certifies.

If the attacker can switch a link on, the attacked entry becomes (1 + a[s])·w[e] whatever z[e] was. The extra turns into a sum of ((1 + a[s]) − z[e])·w[e]·v[s]/v[i] over the attacker's links into i. That is positive even at z = 0, so switching links off stops being a defence.

Now take the safest design: backup everywhere, no authority, every link off. Under the same attack, any other design has every matrix entry at least as large:

- Less backup only enlarges the sector block.
- Authority and honest links only add entries.
- The attacker's links are identical either way.

For nonnegative matrices, entrywise larger means a spectral radius at least as large, by Perron–Frobenius monotonicity. On this city the safest design reaches 1.083 when agents A1 and A5 switch their links on. So every design is unstable under that attack, and by Collatz–Wielandt no witness can certify any of them. The same holds on 31 of 40 cities. A certificate for this threat must therefore constrain the candidate graph itself, meaning which links can physically exist, enforced outside the agents.

![**Figure 4.** (a) The extra cost an attack adds on one link, as a function of the design's intensity z[e] (drawn with a = 0.5, w = 1). A design-proportional attack costs nothing at z = 0; an attacker who can switch the link on still costs (1 + a)·w there. (b) Under that attack the safest design already has ρ = 1.083 (the figure from the text); every other design's matrix is entrywise at least as large, so its ρ is at least 1.083, and nothing lands in the certifiable region ρ < 1.](figures/fig4_why_no_design_survives.png)
