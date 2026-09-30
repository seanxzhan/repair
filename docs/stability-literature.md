# Joint stability: literature summaries

One summary per paper, all in the same format: problem, why it matters, contributions, key intuitions, the technical crux explained simply, results, limitations, and relevance to this repo. Every summary was written from the full text; each file's `Source read:` line gives the PDF used. Claims that are the summary writer's own, not the authors', are marked *(our observation)*.

Each summary carries figures. Most are **computed**, not drawn: the paper's method runs on a small example (an LP with friction cones, an eigenmode, a blocking test) and the plot shows what came out, so a caption's numbers can be checked by re-running the script. The scripts are in [figures/](figures/), one per summary, sharing [figures/_style.py](figures/_style.py); run them from that directory with `python3 <slug>.py` to regenerate the PNGs.

"Stability" breaks into three separate questions:

- **Kinematic**: can the parts move apart?
- **Static**: do they stay put under gravity or a load, with friction?
- **Structural**: does the wood itself break, crush, or bend?

## Suggested reading order

1. [wang2021-star-assemblies-rigid-parts](wang2021-star-assemblies-rigid-parts.md): the map of the field. Skim [song2022-computational-assemblies-tutorial](song2022-computational-assemblies-tutorial.md) for the slide material the survey lacks.
2. [yao2017-decorative-joinery](yao2017-decorative-joinery.md): closest to this project. Glue-free joints plus a static analysis that finds sliding and hinging.
3. [larsson2020-tsugite](larsson2020-tsugite.md), then [ganeshan2025-migumi](ganeshan2025-migumi.md): the line of work our dataset comes from. MiGumi optimizes fit, not load-bearing.
4. The kinematic, static, or structural track below, depending on the question.

## Kinematic: can it come apart?

| Paper | In one line |
|---|---|
| [wang2018-desia](wang2018-desia.md) | Blocking graphs per direction; interlocking test in polynomial time instead of checking every subset of parts |
| [wang2019-topological-interlocking](wang2019-topological-interlocking.md) | Adds rotations; interlocking ⇔ frictionless equilibrium under any load (Farkas' lemma) |
| [wang2021-mocca](wang2021-mocca.md) | Joints that allow a *cone* of removal directions; trades ease of assembly against stability |
| [song2012-recursive-interlocking-puzzles](song2012-recursive-interlocking-puzzles.md) | A local three-piece locking rule that provably gives global interlocking |
| [fu2015-interlocking-furniture-assembly](fu2015-interlocking-furniture-assembly.md) | Picks a single-axis joint for each connection so the whole furniture frame locks |
| [song2017-reconfigurable-interlocking-furniture](song2017-reconfigurable-interlocking-furniture.md) | One part set, several interlocked forms. The full-size prototypes still needed pins to carry load |
| [chen2022-high-level-interlocking-puzzles](chen2022-high-level-interlocking-puzzles.md) | Puzzles needing up to 27 moves before the first piece comes out |

## Static: does it hold under load?

| Paper | In one line |
|---|---|
| [mosemann1997-stability-assemblies-friction](mosemann1997-stability-assemblies-friction.md) | Stability with friction as a linear program; enumerates all stable orientations |
| [whiting2009-structurally-sound-masonry](whiting2009-structurally-sound-masonry.md) | Rigid-block equilibrium; "how much glue would be needed" as a smooth measure of instability |
| [whiting2012-structural-optimization-masonry](whiting2012-structural-optimization-masonry.md) | A torque-based energy whose gradient points to useful geometry changes |
| [kao2022-coupled-rigid-block-analysis](kao2022-coupled-rigid-block-analysis.md) | Why force-only analysis is wrong for non-planar contacts, and how adding kinematics fixes it |
| [nadeau2024-robustness-frictional-contact](nadeau2024-robustness-frictional-contact.md) | The largest push an assembly survives, split into slipping and toppling |
| [wang2025-learning-to-assemble](wang2025-learning-to-assemble.md) | A learned assembly-sequence planner in which every partial assembly must be stable |

## Structural: does the material hold?

| Paper | In one line |
|---|---|
| [umetani2012-guided-exploration-furniture](umetani2012-guided-exploration-furniture.md) | Toppling and nail-joint failure as the valid parameter range shown while editing |
| [liu2022-worst-case-rigidity](liu2022-worst-case-rigidity.md) | One eigen-solve gives the softest deformation mode and the load that causes it |
| [janikova2025-rta-eccentric-joint-strength](janikova2025-rta-eccentric-joint-strength.md) | Lab strength tests of hardware corner joints in particleboard; shows what empirical testing does and doesn't tell us |
