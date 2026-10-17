# Foveated ViT plan

Oct 17, 2026 · @Oliver Holland

Test whether per-head, attention-guided foveation can cut ViT attention cost at high resolution while keeping DINOv3 features, in four phases from a no-training check to optimised kernels.

## Background and goal

**The problem.** A vision transformer (ViT) cuts an image into square patches, turns each patch into a token, and lets every token attend to every other token. Attention cost grows with the square of the token count. Doubling an image's width and height gives 4 times the tokens and 16 times the attention cost. A 4K image (3840 by 2160) has about 32,400 tokens at 16px patches and about 518,400 at 4px patches, which makes dense attention impractical at the finer size.

**The idea.** Human vision is sharp only at the fovea. Here, each attention head gets a fixed budget of K key tokens and spends it according to its own attention: fine tokens where the head's attention is high, coarse tokens elsewhere. Every position still produces an output, but each head reads only a K-token summary of the image, so attention cost grows linearly with image size instead of quadratically.

**The goal.** Adapt the pretrained DINOv3 model so that it keeps its feature quality at 16px (Phases 0 and 1), then use the saved compute to see 4px detail wherever the heads want it (Phase 2).

**What success looks like.**

- Phase 1: probe results close to dense DINOv3 at the same resolution, at much lower attention cost.
- Phase 2: better results than dense 16px DINOv3 on tasks that depend on small details.
- Phase 3: the compute saving shows up as real wall-clock speed.

**Starting values and open targets.**

| Parameter | Starting value | Status |
| --- | --- | --- |
| Key budget K per head | 1,024 | Value used in the cost estimates; tuned in Phase 0. Since Oct 17, 2026 K counts the CLS and register tokens |
| Starting grid | 64px or 32px cells | Ablated in Phase 1. At 4K, 64px start cells alone exceed K = 1,024, so 4K uses 128px |
| Patch size, Phases 0 and 1 | 16px | Fixed, matches DINOv3 |
| Minimum patch size, Phase 2 | 4px | Fixed for the first attempt |
| Image sizes | 1K to 4K | Phase 0 repeats the check at several sizes |
| Accuracy tolerance vs dense DINOv3 | Not set | Open: to be agreed |
| Benchmarks | Segmentation and depth probes | Phase 0 uses DIV2K validation (100 2K photos, in data/). DIS5K is the candidate fine-detail segmentation probe. Depth set not chosen |
| Speedup target | Not set | Open: to be agreed |

## Glossary

| Term | Meaning here |
| --- | --- |
| Token, patch | One square piece of the image (16px or 4px on a side) and its vector representation |
| Head | One of several parallel attention units in a layer, each with its own query, key and value projections |
| Query, key, value | Each token produces a query (what it looks for), a key (what it offers) and a value (what it passes on). Attention weights come from query-key matches and are applied to the values |
| Attention map, attention mass | The grid of weights saying how much each query attends to each key. Attention mass is the total weight a key receives across queries |
| Foveation | Using fine resolution only where it is needed and coarse resolution elsewhere |
| Budget K | The fixed number of key tokens each head may use, whatever the image size |
| Quadtree pyramid | The image's tokens repeatedly averaged 2 by 2 into coarser levels. Each coarse cell has four children one level finer |
| Splitting | Replacing a coarse cell in a head's key set with its four finer children |
| Log-count bonus | A correction added to a coarse token's attention score, based on how many patches it covers, so it is not under-attended |
| RoPE (rotary position embedding) | A way of encoding position by rotating queries and keys, so attention depends on relative offset. Coarse tokens use the coordinates of their cell's centre |
| Residual stream | The running vector each token carries through the network. It stays at full resolution, so detail is not lost where keys are coarse |
| Distillation | Training a student model to reproduce a frozen teacher's outputs. Here the teacher is dense DINOv3 |
| LoRA | Fine-tuning by training small low-rank corrections to existing weights while the originals stay frozen |
| Probe | A small model trained on frozen features to measure their quality on a task such as segmentation or depth |

## Phases at a glance

Each phase builds on the one before and ends at a gate. If a gate fails, stop and revisit the design before spending more on later phases.

| Phase | What changes | Deliverable | Gate to continue |
| --- | --- | --- | --- |
| 0. Feasibility | Nothing is trained. Foveated key sets are simulated on pretrained DINOv3 | Plots of attention mass captured against K, per head and per image size | A modest K captures most of the mass for most heads |
| 1. Foveation at 16px | Foveated attention wrapper plus distillation from dense DINOv3 | Trained student and a results table against dense, uniform-pooling and random baselines | Close to dense on probes, clearly ahead of the two baselines |
| 2. Down to 4px | 4px patch embedding and residual stream | 4px model and results on detail-heavy tasks and 4K images | Measurable gain over the Phase 1 model on small-detail tasks |
| 3. Optimisation | Fused kernels and cheaper scoring | Wall-clock benchmarks against dense attention | Real speedup at 4K without losing accuracy |

## Cost estimate

At 4K, the foveated design is roughly 50 to 95 times cheaper than a dense model at the same 4px detail, and about 5 to 6 times cheaper than dense DINOv3 at 16px. These are my own back-of-envelope estimates, not measurements.

Assumptions: ViT-B size (768 dimensions, 12 layers), a 3840 by 2160 image, K = 1,024 keys per head, counting only attention, projections and MLPs. A range means scoring rounds use pooled queries (low end) or all queries (high end).

| Model | Tokens | Approx. compute (TFLOPs) |
| --- | --- | --- |
| Dense DINO at 256px, 16px patches | 256 | 0.05 |
| Dense ViT at 4K, 16px patches | 32,400 | 44 |
| Foveated, 16px minimum | 32,400 | 7 to 9 |
| Dense ViT at 4K, 4px patches | 518,400 | 10,000 |
| Foveated, 4px minimum | 518,400 | 107 to 205 |

Two things limit the saving. Attention becomes cheap, but the per-token projections and MLPs still run on every token and form a floor, which is why the 16px foveated model is only about 5 to 6 times cheaper. And real wall-clock speed will be lower than these FLOP ratios until the gather, pooling and selection steps are fused (Phase 3).

## Design

Each head builds its own mixed-resolution key set from the same fine token grid, using that head's own attention to decide where to look.

- The residual stream stays on the 16px grid in Phases 0 and 1, so pretrained DINOv3 weights are used as trained.
- Per layer, keys and values are projected once on the fine grid, then mean-pooled 2x2 level by level into a quadtree pyramid (pool after projection and norm).
- Each head starts from a coarse grid (64px or 32px cells) and runs scoring rounds with no gradients: attend to the current key set, sum the attention each key receives, split the highest-mass cells into four children, repeat until the budget of K keys is spent.
- The real attention then runs with gradients over that final set. Queries stay on all fine tokens, so each head's output lands directly on the tokens and nothing needs scattering back.
- Pooled tokens get rotary positions at their true centroid, a log-count bonus on their logits (a token covering m patches is not under-attended), and a scale signal.
- Gradients flow through the selected tokens and attention values, but not through how the layout was chosen.
- Attention cost per head becomes about N times K, linear in token count N, against N squared for dense attention.

## Phase 0: feasibility check, no training

Goal: find out whether a fixed K captures most of what each head attends to, before building anything.

- Run pretrained DINOv3 on high-resolution images (1K to 4K, 16px patches).
- For every head, compare the full attention map against a top-K key set built by the split procedure (coarse start, no-grad scoring, split until the budget is spent).
- Measure the fraction of attention mass captured, per head, against K.
- Repeat at several image resolutions to see how the K you need grows with image size.
- Note which heads are too diffuse to foveate. Those get a fixed global role or a larger budget.

Gate: proceed only if a modest K captures most of the mass for most heads.

## Phase 1: foveation at 16px, distilled from DINOv3

Goal: show the foveated model matches dense DINOv3 at a fraction of the attention cost, with no loss of detail since nothing finer than 16px exists yet.

- Student: DINOv3 with the per-head foveation layer from the Design section, trained with LoRA or full fine-tuning.
- Teacher: frozen dense DINOv3 on the same image. Loss: match CLS and patch features.
- Baselines: dense DINOv3 at the same resolution; uniform pooling to K keys with no adaptivity; random selection of K cells. The last two show whether attention-guided layout matters.
- Metrics: feature similarity to the teacher, linear-probe segmentation and depth, and attention mass captured.
- Ablations: K, the starting grid (64px or 32px), the number of refinement rounds, and a few heads fixed as global coarse or local fine.

Gate: within an agreed tolerance of dense DINOv3 on probes, and clearly ahead of uniform and random baselines.

## Phase 2: drop the minimum patch size to 4px

Goal: show that finer tokens placed by the heads improve high-detail tasks, at the same K.

- The pyramid and residual stream move to the 4px grid. This needs a new 4px patch embedding, since the pretrained one is 16px. How to initialise it from the pretrained embedding is an open question.
- Evaluate on detail-heavy tasks: small-object segmentation, high-resolution depth, and probes on 4K images.
- Compare against dense 16px DINOv3 and the Phase 1 model.
- A 16px teacher cannot reward fine detail by itself. Pool student features to 16px for the distillation loss, and add a task loss or probe-based objective so the 4px tokens have a reason to be useful.
- Expect the per-token projections and MLPs to dominate cost, since the token count is 16 times higher than at 16px. An 8px residual stream is the fallback compromise.

Gate: a measurable gain on detail-heavy tasks over the Phase 1 model.

## Phase 3: optimisations

Only worth doing if Phases 1 and 2 show promise, since the aim is wall-clock speed, not new capability.

- Store tokens in Morton (Z-order) order so the four children of a cell are adjacent, and fuse the pyramid build into one Triton kernel (or try torch.compile on the reshape-and-mean chain first).
- Score with pooled queries in the no-grad rounds, and try a streaming column-max statistic (subtract each query's own max first) instead of a two-pass attention-mass sum.
- Run the final gradient-carrying pass with standard flash attention over the gathered per-head key set. Fold the log-count bonus into the kernel via an extra feature, or use FlexAttention.
- Try per-query-block layouts with block-sparse attention if per-head layouts prove too coarse.
- Try smoothing layouts across layers, optional merge operations, and an 8px residual stream as a compute compromise.

Expect a real speedup well below the FLOP saving until the gather, pooling and selection rounds are fused.

## Implementation notes

The aim is a minimal change to DINOv3: replace each block's attention module with a wrapper that holds the original module and reuses its parameters by reference, with no copies and no new parameters in the first version.

**Where the code and weights come from.** Meta's repo is [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3), and the model is also in Hugging Face transformers ([docs](https://huggingface.co/docs/transformers/model_doc/dinov3)). The [weights](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m/blob/main/README.md) are gated behind a form and use a custom DINOv3 license, so read the terms before use beyond research. The model card lists patch size 16, 4 register tokens for the transformer models, and RoPE. The Hugging Face output order is CLS, registers, then patches.

**What the wrapper's forward pass does.**

1. Project the incoming tokens once on the fine grid with the original q, k and v projections, then split into heads. Keep keys unrotated at this stage.
2. Build one flat pyramid buffer per head by mean-pooling keys and values 2 by 2 level by level. Store each cell's centroid, size, token count and child indices, so a key is just an integer index into the buffer.
3. Run the selection loop with no gradients. Each head starts with all cells at the coarsest level. Each round, pooled queries attend to the current key set, key-side attention is summed, and the top cells are replaced by their four children. A fixed schedule of splits per round gives exactly K keys, so shapes stay static.
4. Gather keys and values by the final indices, apply rotary at each gathered token's centroid, and run scaled dot-product attention with heads as a batch dimension and all fine tokens as queries. Then apply the original output projection.

Gradients reach the selected tokens through the gather and through pooling back to the fine keys and values. The index tensors are integers, so nothing flows through the layout choice.

**Things to check in the DINOv3 code before writing the wrapper.**

- The attention submodule's name, and whether q, k and v are one fused layer or three. The Hugging Face config has separate query, key and value bias flags, which hints at separate layers, but I have not confirmed it.
- That the rotary code accepts arbitrary float coordinates, since pooled tokens sit at cell centres.
- Position-embedding shift, jitter and rescale options that the Hugging Face docs say apply only in training mode. Turn these off for the student so its coordinates match the frozen teacher's (my inference from the docs).
- CLS and register tokens stay outside the pyramid as always-present keys and are never pooled.

**First-version shortcuts.** The log-count bonus needs an attention mask, which may disable the fast flash-attention path. Accept the slower path now and fix it in Phase 3. Score with pooled queries on a coarse grid to keep the extra rounds cheap.

**First test.** With K at least the total number of cells, the wrapper must reproduce the original module's output. Run this before any training, since it validates the rotary and indexing code.

## Design decisions

Choices made while implementing, to revisit once Phase 0 and Phase 1 results are in.

| Decision | Choice | Why | Alternatives to revisit |
| --- | --- | --- | --- |
| Selection order (Oct 17, 2026) | Level by level, coarsest first. At each level, only the cells the level above created compete, and the top ones split | Attention mass grows with cell area, so ranking cells of different sizes together always favours splitting the biggest. In the first Phase 0 run this left only about 6% of attention on 16px keys at K = 1,024 | Global priority queue: better allocation but sequential. This also fixes the number of scoring rounds at one per level, so the Phase 1 "number of refinement rounds" ablation no longer applies |
| Global, area-normalised order (Oct 17, 2026) | Tried as `order="global"`: each round ranks every splittable cell by score per 16px patch | Lets cells of different sizes compete directly, so the data decides how much budget each level gets | Pilot, 2 images, untrained, end to end: slightly worse than per-level everywhere (final CLS cosine 0.837 vs 0.846 at K = 1,024). Keep as an option; revisit with more images |
| Split score (Oct 17, 2026) | Tried as `score="detail"`: attention mass × RMS spread of the cell's patch values, a rough estimate of the output error from leaving the cell unsplit | Attention mass alone says where a head looks, not whether splitting changes anything; flat regions are cheap to keep coarse. Layouts showed some heads foveating borders or sweeping the image top to bottom | Pilot, same setup: best of all variants. Final CLS cosine 0.867 vs 0.846 for attention mass and 0.824 for random at K = 1,024; 0.948 vs 0.928 and 0.901 at K = 2,048. Same ranking on both images. **Default since Oct 17, 2026**; recheck on a real image set |
| Splits per level (Oct 17, 2026) | **Provisional:** spare budget shared equally between levels. Splits a level cannot take, because the level above created too few cells, pass down. HD at K = 1,024 with 64px start cells: 86 splits at 64px, 85 at 32px | A neutral default between breadth and depth | Coarse-first: as many splits as possible at the top, giving a wide medium-detail view. Fine-first: split just enough at the top to reach 16px everywhere chosen, giving fewer but sharper regions. Ablate in Phase 1. Equal-slope allocation (below) replaces the fixed shares altogether |
| Split criterion, tested (Oct 17, 2026) | Compared criteria against each cell's true output error: the change in a head's output, over 256 sampled queries, when only that cell is pooled. Run with `split_criteria.py` | Attention mass says where a head looks, not what pooling a cell costs. Rate-distortion and first-order error analysis suggest better scores | 10 DIV2K images, HD, all heads, 32px and 64px cells. Rank correlation with true error: bound 0.830, KL 0.817, first-order covariance 0.801, mass × value spread 0.789, mass 0.633, random 0.00. The bound (query norm × mass × key spread × value spread / √d, rotated keys) beat mass × value spread on all 10 images. Middle layers are hardest for every criterion |
| Equal-slope allocation (Oct 17, 2026) | Implemented as `order="tree"`, with `score="bound"`. One scoring pass of the pooled queries over every 16px key gives every cell's error estimate at once, with masses summed up the pyramid. Each head bisects on its own price per key, so it splits a cell exactly when the error removed per added key beats that price. Leftover slots are filled greedily | Lets each head choose its own mix of cell sizes, which fixed per-level quotas cannot. Rate-distortion's equal-slope rule, the same logic as cost-complexity pruning in decision trees | Smoke photo, end to end, K = 1,024: final CLS cosine 0.913 vs 0.894 for per-level + detail and 0.823 for random; 0.966 vs 0.959 at K = 2,048. Most of the gain is the criterion (equal-slope + detail: 0.896). Scoring costs about 3% of dense attention at HD and stays linear in image size only while the number of scoring queries stays fixed. Not yet the default: time it and compare on more images first |
| Budget includes CLS and registers (Oct 17, 2026) | `budget` is the exact key count per head, CLS and 4 registers included. Unused slots hold empty cells with zero attention | A fixed, kernel-friendly key count such as 1,024 | Splits add 3 keys, so the cell count is start cells + 3 × splits and up to 2 slots stay empty: at HD, 1,017 cells + 5 = 1,022 real keys of 1,024. The register count was left at DINOv3's pretrained 4: removing registers would hurt features, new ones need training, and the remainder depends on image size anyway. Partial splits (one cell into fewer tokens) could fill the last slots if they ever matter |
| Pyramid speed-ups (Oct 17, 2026) | Layout and RoPE sin/cos computed once per forward pass and shared by all layers; queries pooled only to the scoring level; optional `torch.compile` of the per-layer pooling (`compile=True`, recompiles per image size) | Building the pyramid cost more than the attention it saved at HD | See "Results so far". Next: fold the log-count bonus into the attention kernel instead of passing it as a mask (Phase 3) |

## Results so far

All on pretrained ViT-L/16 with no training, so they measure the starting gap that Phase 1 distillation has to close.

**Phase 0 on DIV2K** (100 validation images at HD, per-layer check, per-level order with the detail score):

| K | Attention on 16px keys | Output cosine vs dense | Heads with cosine ≥ 0.95 |
| --- | --- | --- | --- |
| 1,024 | 10% ± 2% | 0.948 ± 0.019 | 63% |
| 2,048 | 25% ± 3% | 0.971 ± 0.012 | 83% |

Middle layers (5 to 11) are weakest. Running every layer foveated compounds the errors through the middle of the network, so end-to-end checks matter more than per-layer ones.

**Speed** (`benchmark.py`, bf16, RTX 3080, K = 1,024, compiled pooling):

| Image | Dense | Foveated | Speedup | Attention FLOPs, dense vs foveated |
| --- | --- | --- | --- | --- |
| HD, 120×68 patches, 64px start | 278 ms | 216 ms | 1.3× | 6.6 vs 0.9 TFLOPs |
| 4K, 240×135 patches, 128px start | 2,359 ms | 775 ms | 3.0× | 103 vs 3.3 TFLOPs |

The projections and MLPs run on every token and set a floor: at HD they are most of the remaining cost. At 4K the rest is mainly the pyramid build (122 ms) and the final attention (167 ms), which uses a slower kernel because the log-count bonus is passed as a mask. Measured with per-level order; equal-slope order not timed yet. A 128px start is no faster at HD.

## Risks and open questions

- Fixed K may stop capturing enough attention mass as images grow. Phase 0 measures how the needed K scales with resolution.
- Mean-pooled coarse keys can smear sharp features, so a cell worth splitting may look unremarkable. Try max-pooled keys or mass per unit area within a level for scoring.
- Open (Oct 17, 2026): the per-level schedule fixes how many splits each level gets, the same for every head. A head that cares about only a few cells still has to spend its full quota at the coarse level. In the HD pilot, layer 1, head 8 had 482 of its 510 64px cells score exactly zero and only one above a millionth of the maximum, so 85 of its 86 64px splits went to zero-score cells in index order. 4 of 384 heads did this, all in layer 1; it wastes budget but does not change outputs. The goal is to let each head choose its own mix of cell sizes, for example a flat head turning the whole image into 32px cells and a focused head spending everything on 16px cells in one region, while keeping static shapes for batching. Area-normalised global order (`order="global"`) is a first step but still fixes splits per round, and scored slightly worse in the pilot. Revisit after Phase 0 runs on a real image set. Update: equal-slope allocation (`order="tree"`) addresses this; on the smoke photo every head fills its budget and heads with little to split spend it at their own mix of sizes. Confirm on more images.
- The layout choice is not differentiable. The coarse scorer may need an auxiliary loss, such as matching the full-resolution attention distribution.
- Heads may collapse to similar layouts. Fixing a few heads as global coarse or local fine may stabilise training.
- Pretrained DINOv3 MLPs expect 16px semantics. Changing the key set is a light adaptation, but a 4px stream is a heavy one.
- Attention mass can be high on regions where values barely vary, such as flat sky. Weighting mass by value spread within a cell is one fix to test.
- I have not searched for prior art on per-head, per-layer foveation of a pretrained ViT. Closest relatives I recall, unverified: QuadTree Attention, Shunted Transformer, ViT-Adapter, MSViT and the selection branches in sparse-attention LLMs. Check before claiming novelty.
- Open: how to initialise the 4px patch embedding from the pretrained 16px one.
