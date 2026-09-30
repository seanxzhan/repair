# Prototype plan: the fastest path that still proves the thesis

**Status:** execution plan for a deadline-bound proof of concept. Companion to [proposal.md](proposal.md).

## The thesis, in one sentence

The static capacity of a repair interface is a staircase in the interface parameters (every step is a contact-topology change), the optimum sits on a step, and a learned surrogate gives a usable gradient there where the classical solver gives none.

Everything below exists to produce **one figure** that shows this, plus the two supporting results that make it credible:

1. **The staircase and the surrogate.** A 1D slice through the true capacity landscape (steps) with the surrogate overlaid (smooth), at a damage field the surrogate never saw.
2. **The optimizer.** Gradient descent on the surrogate reaches an interface the true solver confirms is near the best available, while gradient descent on the true solver stalls on plateaus.
3. **Family selection.** Running the optimizer once per interface family and keeping the best gives "which joint" as a by-product, with no hand-made labels.

If time runs out, ship in that order. Result 1 alone is a defensible talk.

## Decisions fixed before writing any code

| Decision | Choice | Why |
|---|---|---|
| Dimension | **2D cross-section** | 3D contact sets, solvers and rendering are weeks. The argument is about topology change, which 2D already exhibits. |
| Label | **Frictional-equilibrium capacity** (max moment the interface transfers), not stress | Milliseconds per label, no meshing, no material model. "Max stress" needs FEM and is mesh-dependent at corners. |
| Damage | **Enters the network as an input**, as a few front parameters | Without it the optimizer cannot trade dead contacts for saved wood, which is the novel claim. |
| Coupling | **Coupled**: the interface may leave dead contacts behind to save sound wood | The decoupled version (cut at the front, then optimize in sound wood) reduces damage to an offset and kills the story. Same effort in 2D. |
| Objective | capacity − λ · sound wood removed, subject to capacity ≥ required load | Matches the proposal. λ and the required load are stated constants, with a one-line sensitivity sweep. |
| Network scope | **One small MLP per family**, native parameters in | A shared representation across families is a follow-up slide. Per-family MLPs train in minutes. |
| Parameter count | **At least 4 per family** | With 2 parameters, grid search is trivially better and a smoothing kernel does the surrogate's job. The surrogate has to earn its place in higher dimension. |

## Steps

### Step 1. Interface families and damage (half a day)

Define the families over one rectangular block, each as a small parameter vector with stated bounds. The four MiGumi splices the prototype draws on are all end-to-end connection joints (tsugi), and each has one cross-section that carries its mechanism. In 2D they become:

| Family | MiGumi joint | What the 2D section keeps | Parameters |
|---|---|---|---|
| plain cut | | a vertical cut; carries no moment | 1 |
| mortise and tenon | | Figure 1 of the proposal | 4 |
| dovetail | CJ_AT, Ari Tsugi | a tenon whose cheeks flare toward the tip | 5 |
| hooked scarf | CJ_DT, Daimochi Tsugi | long shallow scarf with a 45° hook step and short end shoulders | 6 |
| tenon, dovetail and hooked scarf, flipped | | the same joint cut the other way round: the tongue on the retained wood, or the scarf's retained wedge above instead of below | as above |

Orientation is treated as a separate family rather than a parameter. A flip is a reflection of the interface plus a swap of which side is retained, so in sound wood it carries exactly its original's moment, and only the damage field distinguishes the two. That makes the flips a clean test of whether the pipeline actually uses the damage input: a selector that never prefers a flip under a leaning front has not learned the damage.

Two joints were tried and dropped: CJ_AKT, a bowtie key in half the depth, is in 2D just a half-depth dovetail; and CJ_IT, two mirrored plain scarfs, holds a couple only through friction and only below the friction angle, so at its traditional proportions the rigid-body model gives it zero capacity.

Each family needs only two functions: contact samples with normals for a parameter vector, and the removed region as a mask. Faces are sampled at a handful of points each; a sample is live only if the retained wood just behind it is sound.

Damage is a severity field on a grid, not only the parametric front of Figure 1. The generator is erosion from the boundary: seed a run of the end face and at most one surface pocket, take the anisotropic distance from the seeds (rot runs several times further along the grain than across), displace that distance by a smooth noise so the front is ragged without fragmenting, keep only the rot connected to the end face, and pass it through a logistic. Four knobs, uniform over stated ranges: reach, the seeded fraction of the end face, anisotropy, and pocket size (zero for none). Front width, noise and the threshold are fixed constants, because they change how the raster looks and not what the label sees. Everything else about a field, where the run and the pocket sit and the noise realization, comes from its seed. No calibration to real decay. Rot is one connected region by construction: a detached pocket the cut does not touch would be invisible to the objective, so it is not generated. Future work, as a pair: allow more than one connected component (detached pockets, leak spots) **and** add the retained-rot term to the objective, so that leaving a pocket behind costs something. Neither makes sense without the other. An earlier Gaussian-blur version coupled the front's sharpness to its reach and produced speckled islands near the threshold for deep rot; the distance form fixes that. The contact flags and the sound-wood mask threshold the field exactly as they thresholded the front, so the LP still sees only live and dead. The network's damage input is the raster of the window the interfaces live in, x from 6 to 12 over the full height, at 96 by 32, so that a damage field is a picture it has to read rather than three numbers it can memorize. The parametric front stays as a second implementation for Figure 1, the viewers and the sanity checks. Do **not** spend time on a rot model beyond this.

### Step 2. Data generation (half a day, runs in minutes)

For each family:

- Sample parameters uniformly within bounds and damage parameters uniformly within a range where the front crosses the interface region. Reject nothing.
- Label each sample with the true capacity from the solver, and record the sound wood removed and the number of live contacts.
- Tens of thousands of samples per family. Each label is a small linear program, so this is minutes on a laptop and seconds with a process pool.
- **Split by damage field**, not by row. Hold out a set of damage parameter tuples entirely so "unseen damage" means what it says.

Sanity check before training: the capacity of the plain cut is zero everywhere, and capacity is monotone in tenon length inside one active set.

### Step 3. Surrogate (half a day)

- Input: normalized interface parameters concatenated with normalized damage parameters. Output: capacity, scaled to the dataset peak.
- A 4-layer MLP with smooth activations (SiLU or GELU, never ReLU, since the point is a smooth gradient). Adam, a few minutes on CPU.
- Report test RMSE and R² on the held-out damage fields.
- **The figure.** Fix a held-out damage field and the other parameters, sweep one parameter finely, plot the true capacity (staircase) and the surrogate (smooth). Then a second panel with the surrogate's derivative and the true solver's finite-difference derivative: the latter is zero almost everywhere with spikes at the steps.

Also train the surrogate on a small dataset once to check it does not simply memorize steps. If the surrogate reproduces the staircase exactly, it is too big or too well fed; a slightly under-fit model is the one you want.

### Step 4. Optimization (one day)

The removed sound wood is made differentiable directly: a soft mask for the removed region times a soft mask for sound wood, integrated on a fixed grid. No network needed for that term.

Objective per family: surrogate capacity − λ · sound removed − penalty · max(0, required load − surrogate capacity)².

Run four methods on the **same** set of held-out damage fields, each from the same multi-start seeds:

| Method | Role |
|---|---|
| Gradient descent through the surrogate | the proposed method |
| Grid search on the true solver | upper bound on what is achievable; affordable at 4 parameters and coarse resolution |
| Gradient descent on the true solver via finite differences | the classical baseline; expected to stall on plateaus |
| A derivative-free optimizer on the true solver (Nelder–Mead or CMA-ES) | the strong baseline a reviewer will ask about |

**Always re-evaluate every method's final interface with the true solver** and report that number, never the surrogate's own estimate. The optimizer will exploit surrogate error if allowed, and the gap between surrogate estimate and true value at the optimum is itself a useful honesty column.

Report: true objective at the optimum for each method, gap to grid search, and the number of solver calls used. Draw the surrogate's descent trajectory on top of the true landscape with its cliffs.

### Step 5. Family selection (half a day)

For each held-out damage field, run Step 4 for every family and keep the best true objective. Plot which family wins against damage extent. The expected story: plain cut never wins, the tenon wins when the rot is shallow, the scarf wins when the rot leans or runs deep.

**Nested families do not compete fairly.** Butt ⊂ tenon ⊂ dovetail (zero length, then zero flare), so the dovetail's optimum is at least the tenon's by construction, and in this model flare raises capacity monotonically in sound wood at little sound-wood cost, so the optimizer pushes it to its bound. "Dovetail wins" then reports the bound, not the physics: the rigid-body model has no short-grain or crushing limit to stop the flare. The scarf is not nested with the dovetail, but the larger search space still favours it in the same way. The flips are the one clean comparison: same parameter count, same capacity in sound wood, so any difference is damage.

**Retained rot is not scored.** Damage enters the label only through contacts backed by rot; rot left in the retained wood away from the interface costs nothing, so the objective prefers leaving it. The generator keeps rot connected to the end face so this is a bay along a face at worst, but the part of a bay left of the shoulder is still uncounted. Future work, together with multi-component damage fields: a third objective term, the area of rot retained, on the same grid as the sound-wood integral, with its own weight; large weight is the "remove all rot" rule, small weight lets a shallow bay stay if removing it costs too much sound wood.

For now: **report the optimum's parameters and flag any that sit on a bound.** A parameter on its bound is the model saying it lacks a constraint, and the slide says so rather than calling the family the winner.

Noted for later, not done now: **a per-family fabrication cost in the objective**, so the extra parameters of a richer family have to buy capacity beyond what the simpler family gives before it can win. The cost would be a fixed term per family (or a term on flare, hook size and similar features), stated as a design choice and swept like λ. This is the honest version of "which joint" for nested families, and also what a shop would actually charge for. A cheaper middle step is to bound flare physically, for instance to a fraction of the neck thickness, so the dovetail's search space carries a strength rule the statics do not.

If time remains, fit a classifier from damage parameters (or a rasterized damage image, if you want the voxel picture) to the winning family. This is an amortized selector trained on labels the optimizer produced, not on labels anyone invented. Say that on the slide.

### Step 6. Presentation (half a day)

Figures, in slide order:

1. Input: a block with a damage field and no interface drawn.
2. Staircase versus surrogate, one held-out damage field.
3. Optimization: trajectories on the landscape, then a bar chart of true objective per method.
4. Family selection against damage extent.
5. One 3D render of a MiGumi joint from the dataset as "what this family looks like in 3D", labeled as future work, with nothing computed in 3D.

## Time budget

| Step | Effort | Cumulative |
|---|---|---|
| 1. Families and damage | 0.5 d | 0.5 d |
| 2. Data | 0.5 d | 1 d |
| 3. Surrogate and the figure | 0.5 d | 1.5 d |
| 4. Optimization and baselines | 1 d | 2.5 d |
| 5. Family selection | 0.5 d | 3 d |
| 6. Slides | 0.5 d | 3.5 d |

If only two days exist: Steps 1–3 and the surrogate-descent versus finite-difference comparison from Step 4, one family.

## Questions to have answers for

- **Why not grid search?** It is the upper bound here because 2D and 4 parameters are cheap. The interface representation in the proposal has dozens of parameters, and the damage field in 3D is not three scalars. The prototype shows the gradient exists and points the right way; it does not claim to beat grid search at 4 parameters.
- **Why not smooth the true landscape with a kernel?** In 2 parameters that works and the proposal figure did it. It does not generalize to unseen damage or to more parameters without evaluating the solver on the whole neighbourhood, which is the cost the surrogate amortizes.
- **Isn't the surrogate only as good as the solver?** Yes. The contribution is the gradient, not accuracy. Every result is re-evaluated with the solver.
- **Doesn't the family with the most parameters always win?** For nested families, yes, and the report says which parameters sat on their bounds rather than declaring a winner. A per-family fabrication cost is the fix, noted in Step 5 and not yet in the objective.
- **Where is the material model?** Deliberately absent. Capacity is rigid-body static equilibrium with friction. Stress and fracture are the next project.

## Not doing

Realistic rot models, voxel CNNs as the primary input, FEM or any stress label, one network shared across families, 3D contact sets or solvers, preload, obstacle contexts, insertability. Each is a follow-up slide and none is a prerequisite for the thesis figure.
