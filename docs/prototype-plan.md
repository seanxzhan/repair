# Prototype plan: what the 2D proof of concept is for

**Status:** execution plan for a deadline-bound proof of concept. Companion to [proposal.md](proposal.md).

## The goal

The full pipeline is: damage field in, repair joint out, with a network in the loop. The network earns its place when the evaluator is an expensive 3D contact simulation and a joint has dozens of parameters. Neither is true here. The evaluator is a 2.5 ms linear program, each family has four or five parameters, and grid search on the LP is the best optimizer. The prototype cannot prove the network is necessary, and does not try.

What it can prove is that **the learning problem is solvable**: a network can read a damage field it has never seen and predict (a) what the solver says about a candidate joint and (b) which joint the solver-driven optimizer would choose. If that fails on cheap 2D fields where every answer can be checked exactly, it will fail worse with FEM in 3D. If it works, the expensive version inherits a tested recipe.

Headline results, in priority order:

1. **Damage in, joint out.** A network trained on solver-produced optima predicts family and parameters for held-out damage fields, and the solver confirms its answers are near the best available.
2. **Capacity on unseen damage.** A surrogate of the evaluator predicts capacity on held-out fields. Figure: a 1D parameter sweep on one held-out field, the LP's staircase against the surrogate's curve.
3. **The choice depends on the damage.** Which family wins, and whether a flipped orientation wins, changes with the field. Flips are identical in sound wood, so preferring one is proof the pipeline read the damage.

If time runs out, ship in that order. Result 1 alone is the talk.

### Two roles for a network

| | A. Surrogate of the evaluator | B. Designer |
|---|---|---|
| Learns | (joint parameters, damage) → capacity | damage → best family and parameters |
| Training rows | any evaluation, good or bad | one solved optimum per damage field |
| Cost per row | one solver call | one optimization, many solver calls |
| Used how | inside an optimizer | one forward pass replaces the optimizer |

B is the product, and A and B compose at inference: B's guess is the starting point, A's gradient refines it, the solver confirms the result. B is also what makes the start sensible: a random start in a five-parameter space usually lands in the rot, where no route has a slope in capacity. A is what makes B's labels affordable once the evaluator is FEM: thousands of optima at hundreds of FEM calls each is the expensive step, and running the optimizer on A instead makes each label cheap. With the LP, B's labels come straight from the solver. A is still built, because the future pipeline needs it and its generalization is worth testing now.

On differentiability: the LP already has a gradient almost everywhere through its duals, and the cliffs in the capacity landscape come from a hard threshold that could be softened inside the LP. A learned gradient pays off only with many parameters, an evaluator without duals, or a gradient with respect to the damage itself. Those are 3D and FEM arguments for the proposal, not claims this prototype makes.

## Decisions

| Decision | Choice | Why |
|---|---|---|
| Dimension | 2D cross-section | 3D is weeks of work and the learning question does not need it. |
| Label | Frictional-equilibrium capacity, not stress | Milliseconds per label, no mesh, no material model. |
| Damage | A severity field on a grid, fed to the network as a raster | Three front numbers can be memorized. A picture has to be read. |
| Coupling | The interface may leave dead contacts behind to save sound wood | Cutting at the front first reduces damage to an offset and kills the story. |
| Objective | capacity − λ · sound wood removed, with capacity ≥ required load | As in the proposal. λ and the load are stated constants with a sensitivity sweep. |
| Label optimizer | Whatever is cheapest on the true LP | Honest at this scale. The surrogate route is reported alongside, not instead. |
| Network scope | One model per family for A; a family classifier plus per-family regressors for B | A shared representation is a follow-up. |

## Steps

### Step 1. Families and damage (done)

Each family maps a small parameter vector to the region of the block the new wood replaces. Contact faces are read off that region's edges. Four MiGumi splices were examined; two survive as 2D sections, each in two orientations.

| Family | MiGumi joint | 2D section | Parameters |
|---|---|---|---|
| plain cut | | vertical cut; carries no moment | 1 |
| mortise and tenon | | Figure 1 of the proposal | 4 |
| dovetail | CJ_AT, Ari Tsugi | tenon with cheeks flaring at a dovetail angle of 6° to 15°, so no dovetail is a tenon | 5 |
| hooked scarf | CJ_DT, Daimochi Tsugi | shallow scarf with a 45° hook mid-chord and equal shoulders | 4 |
| the three above, flipped | | the same joint cut the other way round | as above |

A flip reflects the interface and swaps which side is retained. In sound wood it carries exactly the original's moment, so only damage separates the two. A selector that never picks a flip has not learned the damage.

Dropped: CJ_AKT (a bowtie key, in 2D just a half-depth dovetail) and CJ_IT (two mirrored scarfs, zero capacity at traditional proportions because the slope exceeds the friction angle). The hooked scarf needs both shoulders and the hook, and carries nothing without friction.

Damage is a severity field on a grid. The generator is erosion from the boundary: seed a run of the end face and at most one surface pocket, take the anisotropic distance from the seeds, displace it with smooth noise so the front is ragged, keep only rot connected to the end face, pass through a logistic. Four knobs: reach, seeded fraction of the end face, anisotropy, pocket size (zero for none). Front width, noise and the threshold are constants, since they change the picture but not the label. Placements and the noise realization come from the seed. Not calibrated to real decay. The network sees the window x from 6 to 12 over the full height as a 96 by 32 raster.

Rot is one connected region by construction. A detached pocket the cut does not touch is invisible to the objective, so none is generated. Future work, as a pair: multi-component fields **and** a retained-rot term in the objective. Neither makes sense alone.

### Step 2. Data for A (done)

2000 fields, 20 interfaces per family on each, 40k rows per family. Parameters are uniform within bounds, resampled when infeasible so rows are not piled on the feasible boundary. Each row stores the parameters, the field index, capacity, sound wood removed and live count per face. 200 fields are held out entirely.

These rows are deliberately not good joints. A has to learn what a dead cut and a wasteful cut look like. Good data for A is coverage; good data for B is correct optima, which is Step 4.

### Step 3. The surrogate, A (half a day)

- Input: the damage raster through a small convolutional encoder, plus the normalized parameters. Output: capacity. One model per family.
- A quarter to a half of rows have zero capacity. Weight or oversample the rest, or the network learns "zero".
- Report test RMSE and R² on held-out fields.
- Figure: one held-out field, one parameter swept, the LP's staircase and the surrogate's curve.

### Step 4a. The optimizer and the comparison table (done, first pass)

Objective per family: capacity / M_ref − λ · sound removed / R_ref − penalty · max(0, required load − capacity)². M_ref is the family's training peak, R_ref half the block.

Four routes, all in the family's normalized parameter space, all projected to feasibility after every step, all re-scored by the hard LP at the end:

| Route | Gradient | Role |
|---|---|---|
| surrogate | the network's autograd for capacity; finite differences for the sound-wood term, which is smooth | the route the future pipeline needs |
| soft LP | the LP with softened contact flags (every contact live with a budget scaled by a sigmoid of its severity), finite differences on that continuous landscape | the hand-made smooth solver; the fair comparison |
| hard FD | finite differences on the hard LP | the staircase as a classical optimizer feels it |
| reference | none: random search plus Nelder-Mead on the hard LP | the best available; the others are measured against it |

Starts: in the prototype each start is the best of 16 random LP probes, charged to every route, because a plain random start lands in the rot 71% of the time and every route then stalls on the dead plateau. In the pitch the start is the designer B's guess for the field: B proposes, the gradient route refines, the solver confirms. The comparison table measures the refinement step on its own.

The table reports, per route: mean true objective, gap to the reference, how often the end point has zero capacity, how far the route's own estimate is from the LP's score, LP calls, and how often a parameter sits on a bound. Routes share their random starts. A polyscope viewer runs any route on any held-out field and scrubs the trajectory.

### Step 4b. B's labels (next)

For every field, train and held-out, run the reference route for every family: a few thousand LP calls per family per field, about an hour for all fields on this machine. Record the best parameters, the true objective, and which parameters sit on a bound. This table is B's training set.

### Step 5. Family selection and the designer, B (one day)

From the Step 4 table, the winner per field is the family with the best true objective. Plot winners against reach and pocket size. Count how often a flip beats its original. Expected: the plain cut never wins, tenon or dovetail wins on shallow rot, a flip wins when a pocket sits where the original's tongue would go.

Train B on the table: a classifier from the raster to the winning family, and per family a regressor from the raster to the parameters. Evaluate on held-out fields by re-solving B's prediction with the LP. Report the objective gap to the optimizer, family accuracy, and top-2 accuracy, since two near-equal families are a tie, not an error.

Two caveats to state on the slide:

- **Bigger families win by construction.** Butt ⊂ tenon, and a larger search space favours the dovetail even though it is no longer a superset. In this model flare raises capacity at little cost, so the dovetail angle goes to its bound. Report parameters that sit on a bound; that is the model admitting a missing constraint. The fix, noted for later, is a per-family fabrication cost in the objective.
- **Retained rot is not scored.** Rot left away from the interface costs nothing, so the objective prefers leaving it. Connectivity keeps this to a bay along a face. The retained-rot term is the fix, paired with multi-component fields.

### Step 6. Slides (half a day)

1. Input: a block with a damage field, no interface drawn.
2. Damage in, joint out: held-out fields with B's joint drawn on each, the solver's score for B's choice next to the optimizer's.
3. Family and orientation against damage, including the flips' share.
4. Capacity on unseen damage: staircase and surrogate on one held-out field, with the test error.
5. One 3D MiGumi render as future work, and one slide on what changes at scale: FEM labels, dozens of parameters, A as the engine for B's labels.

## Time budget

| Step | Effort | Cumulative |
|---|---|---|
| 1. Families and damage | done | |
| 2. Data for A | done | |
| 3. Surrogate A | 0.5 d | 0.5 d |
| 4a. Optimizer and comparison table | done | |
| 4b. Label table on every field | 0.5 d | 1 d |
| 5. Selection and B | 1 d | 2 d |
| 6. Slides | 0.5 d | 2.5 d |

With two days: Step 4b's table, Step 5 with the classifier only, figures 1 to 3.

## Questions to have answers for

- **If the LP is that cheap, why a network?** Because the real evaluator will not be. The prototype tests whether the learning works while the ground truth is still cheap enough to check every answer.
- **Why not grid search?** It produces the labels here, and the plan says so. It stops being affordable with dozens of parameters or an FEM evaluator.
- **Why a learned gradient?** Not needed here, and the LP has duals anyway. It matters for many parameters, evaluators without duals, and sensitivity to the damage. Those are proposal claims, not prototype results.
- **Isn't the surrogate only as good as the solver?** Yes. Everything is re-evaluated with the solver.
- **Doesn't the biggest family always win?** For nested ones, yes. The report shows which parameters sat on bounds instead of declaring a winner. A fabrication cost is the fix.
- **Several joints can be equally good. What does B predict?** A family classifier with top-2 reported, then parameters within the family. A generative model is the fuller answer, later.
- **Where is the material model?** Absent on purpose. Capacity is rigid-body equilibrium with friction. Stress and fracture are the next project.

## Not doing

Realistic rot models, FEM or stress labels, one network shared across families, 3D, preload, obstacles, insertability, generative designers, the retained-rot term, multi-component damage. Each is a follow-up slide, none is needed for the headline results.
