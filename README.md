# Hive

`hive.py` is a single-file implementation of a modular causal language model built from many small transformer experts (`cubes`) plus a `planner` that selects which experts to activate at each layer.

The training story in the code is:

- tokenize and chunk the corpus;
- cluster the chunks separately for each Hive layer;
- pretrain the shared token interface;
- train each cube on its own shard;
- train the planner on top of frozen cubes;
- jointly fine-tune the assembled system;
- run inference through a sparse per-prompt execution graph.

## What `hive.py` contains

The file includes:

- configuration and automatic dimension derivation from vocabulary size;
- FHRR holographic binding/unbinding for shared complex K/V buses;
- `Cube`, the expert block;
- `Planner`, the routing module;
- `ABI`, the shared token/logit interface;
- `HiveModel`, the assembled model;
- dataset reading, chunking, clustering, and shard export;
- Stage A / B / C training;
- post-Stage-A cluster refinement;
- inference, perplexity evaluation, cluster evaluation, and smoke tests;
- CLI for the full pipeline.

## Architecture

The assembled model is:

- `ABI`
- `dense_pre`
- a stack of Hive layers
- `dense_post`
- optional `PredictiveCodingHead`
- `planner`
- optional `CellularSheaf` per Hive layer

Each `cube` sees the shared residual stream, writes its K/V into a shared FHRR bus, reads the bus back through its own slot, proposes a residual delta, gates that delta, and adds it back to the stream.

The core update in the code is

$$
h \leftarrow h + w_{l,c}\, g_{l,c}(h,\Delta)\, \alpha_{l,c}\, \Delta_{l,c},
$$

where:

- $h$ is the current residual stream;
- $w_{l,c}$ is the planner weight of cube $c$ at layer $l$;
- $g_{l,c}$ is the token-level gate;
- $\alpha_{l,c}$ is the learned residual scale;
- $\Delta_{l,c}$ is the cube's proposed delta.

If a cube is irrelevant, either $w_{l,c}$ or $g_{l,c}$ goes small, so the cube becomes close to identity.

## ASCII overview

```text
raw text / dataset
        |
        v
 tokenizer + chunking
        |
        v
balanced clustering per layer
        |
        +------------------------------+
        |                              |
        v                              v
 cluster shards                 concept centroids
(layer_i/cube_j)                for symbolic prior
        |                              |
        +---------------+--------------+
                        |
                        v
                   Stage 0
            pretrain ABI + dense
                        |
                        v
                   Stage A
         train each cube on its shard
                        |
                        v
            optional cluster refinement
                        |
                        v
                   Stage B
          train planner on frozen cubes
                        |
                        v
                   Stage C
        joint fine-tune assembled model
                        |
                        v
                assembled checkpoint
                        |
                        v
                     inference
```

And one forward pass:

```text
tokens
  |
  v
ABI.embed -> ABI.to_ui -> dense_pre
  |
  v
planner -> route weights per layer
  |
  v
for each Hive layer:
  residual h
    |
    +--> active cube 1: write K/V to FHRR bus ----+
    +--> active cube 2: write K/V to FHRR bus ----+--> shared bus
    +--> active cube N: write K/V to FHRR bus ----+
                                                   |
    +<-- cube 1 unbind/read bus -> gated delta ----+
    +<-- cube 2 unbind/read bus -> gated delta ----+
    +<-- cube N unbind/read bus -> gated delta ----+
                                                   |
                        optional cross-cube mixer / sheaf
                                                   |
                                                   v
                                          updated residual h
  |
  v
dense_post -> optional PC head -> ABI logits
```

## Main formulas

### FHRR bus

Slots are unit-modulus complex phasors:

$$
s = e^{i\theta}.
$$

Binding and unbinding are elementwise:

$$
\mathrm{bind}(s, x) = s \odot x,
\qquad
\mathrm{unbind}(y, s) = y \odot \overline{s}.
$$

Because $|s| = 1$, the code relies on:

$$
\mathrm{unbind}(\mathrm{bind}(s, x), s) = x.
$$

Multiple cubes write into one bus by summation:

$$
K_{\text{bus}} = \sum_{c \in A_l} w_{l,c}\, \mathrm{bind}(s_{l,c}, K_{l,c}),
\qquad
V_{\text{bus}} = \sum_{c \in A_l} w_{l,c}\, \mathrm{bind}(s_{l,c}, V_{l,c}).
$$

Cube $c$ reads its own view by unbinding:

$$
\widetilde{K}_{l,c} = \mathrm{unbind}(K_{\text{bus}}, s_{l,c}),
\qquad
\widetilde{V}_{l,c} = \mathrm{unbind}(V_{\text{bus}}, s_{l,c}).
$$

### ABI

The shared interface is:

$$
e = \mathrm{Embed}(t),
\qquad
h_0 = P_{\text{in}} e.
$$

The output projection is tied:

$$
P_{\text{out}} = P_{\text{in}}^\top.
$$

So logits are:

$$
\ell = E^\top P_{\text{in}}^\top \mathrm{RMSNorm}(h),
$$

where $E$ is the embedding matrix.

### Planner

The planner maps embeddings and dense context to routing logits. After several bidirectional blocks it pools:

$$
\rho = \frac{1}{T}\sum_{t=1}^T z_t,
\qquad
\widehat{\rho} = \frac{\rho}{\|\rho\|}.
$$

Its learned head is:

$$
\ell_{\text{learned}} = \widehat{\rho} W^\top + b.
$$

With the symbolic prior enabled, the final cube logits are:

$$
\ell_c =
\ell_{\mathrm{learned},c} +
s_{\mathrm{prior}}
\frac{\cos(\widehat{\rho}, \widehat{C}_c)}
{\max(\tau_{\mathrm{prior}}, 0.1)}.
$$

where $C_c$ is the centroid-derived concept vector for cube $c$, $s_{\mathrm{prior}}$ is `prior_scale`, and $\tau_{\mathrm{prior}}$ is `prior_temp`.

Per-layer route weights are then:

$$
w_l = \mathrm{sparsemax}(\ell_l)
\quad\text{or}\quad
w_l = \mathrm{softmax}(\ell_l),
$$

with `top_x` limiting how many cubes remain active at inference.

### Cube gate and residual delta

For each active cube:

$$
\Delta_{l,c} = f_{l,c}(h) - h.
$$

The token-level gate is:

$$
g_{l,c}(h,\Delta) = \sigma\!\left(\frac{W_{l,c}[h;\Delta] + b_{l,c}}{T_{\text{gate}}}\right).
$$

The cube contribution is:

$$
\delta h_{l,c} = w_{l,c}\, g_{l,c}(h,\Delta)\, \alpha_{l,c}\, \Delta_{l,c}.
$$

### Sheaf consensus

If `use_sheaf=True`, the code lifts per-cube deltas to a stalk space and performs iterative sheaf diffusion:

$$
X \leftarrow X - \varepsilon L_{\mathcal{F}} X.
$$

For an edge $(u,v)$ the discrepancy is:

$$
d_{uv} = R_{uv} X_u - R_{vu} X_v.
$$

The code also tracks an obstruction-like energy:

$$
\mathcal{L}_{\text{obs}} \propto \sum_{(u,v)} \|d_{uv}\|^2.
$$

This enters Stage C as an auxiliary regularizer.

### Predictive coding head

If `use_pc_head=True`, before unembedding the model performs:

$$
z_0 = W_{\text{enc}} h,
$$

then iterates

$$
\hat{h}_k = W_{\text{dec}} z_k,
\qquad
e_k = h - \hat{h}_k,
$$

$$
z_{k+1} =
z_k + \eta \left(e_k W_{\mathrm{dec}} - \lambda z_k\right).
$$

and outputs

$$
h' = W_{\text{out}} z_K.
$$

## Data preparation

`--prepare` does the following:

1. Reads text from:
   - on-disk HuggingFace Arrow datasets;
   - or normal files/directories (`txt`, `md`, `jsonl`, `json`, `csv`, `tsv`).
2. Tokenizes documents.
3. Chunks them into windows of length `seq_len`.
4. Builds cheap TF-IDF/SVD document features.
5. Clusters chunks independently for each layer into `cubes[layer]` clusters.
6. Writes:
   - `tokens.pt`
   - `clusters.pt`
   - `doc_ids.pt`
   - `concept_feats.npy`
   - per-cube shards like `layer_X/cube_Y/tokens.pt`
7. Optionally writes shifted chunk views for Stage A via `sliding_window_views`.

## Training stages

### Stage 0: `--pretrain-abi --dataset PREP`

Trains only:

- `ABI`
- `dense_pre`
- `dense_post`

Its task loss in the code is standard next-token cross-entropy plus z-loss:

$$
\mathcal{L}_{\mathrm{stage0}} =
\mathcal{L}_{\mathrm{LM}} +
\lambda_z \mathbb{E}\!\left[\left(\log \sum_j e^{\ell_j}\right)^2\right].
$$

Artifact:

- `out_dir/abi.pt`

### Stage A: `--train --stage A --dataset PREP --layer L --cube C`

Trains one cube in isolation on its own cluster shard.

Implementation details:

- the rest of the model is frozen;
- `abi.final_norm` can optionally be unfrozen;
- positives come from the cube's own cluster;
- negatives are sampled from other cubes in the same layer;
- nearest-cluster hard negatives are used;
- upper layers can receive cascaded inputs from the strongest cubes below;
- optional upstream noise is injected to approximate deployment-time inputs.

For layers above zero, the code estimates an input-noise scale:

$$
\sigma_l \approx
\gamma_{\mathrm{noise}}
\frac{\sqrt{n_{\mathrm{terms}}}}{\sqrt{2(l+1)}}
\mathrm{RMS}(h_0),
$$

with $\gamma_{\mathrm{noise}} =$ `stageA_noise_scale`.

The Stage A loss is:

$$
\mathcal{L}_A =
\mathcal{L}_{\mathrm{task}} +
\lambda_{\mathrm{margin}}^{\mathrm{eff}} \mathcal{L}_{\mathrm{margin}} +
\lambda_{\mathrm{open}} \mathcal{L}_{\mathrm{open}} +
\lambda_{\mathrm{close}} \mathcal{L}_{\mathrm{close}},
$$

where, if `stage_a_margin_adaptive=True`,

$$
\lambda_{\mathrm{margin}}^{\mathrm{eff}} =
\lambda_{\mathrm{margin}}
\frac{\mathcal{L}_{\mathrm{task}}}{\mathcal{L}_{\mathrm{task}} + 1}.
$$

Where:

$$
\mathcal{L}_{\mathrm{margin}} =
\mathbb{E}\!\left[\max(0, m - g^+_{\mathrm{logit}})\right] +
\mathbb{E}\!\left[\max(0, m + g^-_{\mathrm{logit}})\right].
$$

$$
\mathcal{L}_{\mathrm{open}} = -\log\!\left(\mathbb{E}[g^+] + \varepsilon\right).
$$

$$
\mathcal{L}_{\mathrm{close}} = \mathbb{E}\!\left[\left(\mathbb{E}[g^-]\right)^2\right].
$$

Artifact:

- merged into `out_dir/hive.pt`

There is also a parallel Stage A path via `parallel_cube_teach`.

### Cluster refinement: `--refine-clusters`

After Stage A, the code can reassign each chunk to the cube whose gate opens most strongly:

$$
c^*(x) = \arg\max_c g_c(x).
$$

This overwrites `clusters.pt`. The intended next step is to rerun Stage A.

### Stage B: `--train --stage B --dataset PREP`

Trains only the `planner` while cubes stay frozen.

The loss is:

$$
\mathcal{L}_B =
\mathcal{L}_{\mathrm{task}} +
\lambda_{\mathrm{bce}} \mathcal{L}_{\mathrm{route}} +
\lambda_{\mathrm{cap}} \mathcal{L}_{\mathrm{cap}} +
\lambda_{\mathrm{balance}} \mathcal{L}_{\mathrm{bal}}.
$$

The route supervision is per-layer cross-entropy against prepared cluster labels:

$$
\mathcal{L}_{\mathrm{route}} =
\frac{1}{L}\sum_{l=1}^{L} \mathrm{CE}(\ell_l, y_l).
$$

The load-balance term uses mean actual route weights:

$$
\mathcal{L}_{\mathrm{bal}} =
\sum_l \sum_c \bar{w}_{l,c} \log\!\left(\bar{w}_{l,c} |C_l| + 10^{-9}\right).
$$

The route-cap penalty is:

$$
\mathcal{L}_{\mathrm{cap}} =
\sum_l \mathbb{E}\!\left[\max(0, |A_l| - \mathrm{top\_x})^2\right].
$$

Stage B uses differentiable sparse routing via `route_weight_matrix(..., soft_topk=True)`.

### Stage C: `--train --stage C --dataset PREP`

This is a joint fine-tune of:

- planner parameters;
- cube gate parameters and `alpha`;
- cube core parameters;
- optional sheaf parameters;
- optional predictive-coding head.

The loss used in the code is:

$$
\mathcal{L}_C =
\mathcal{L}_{\mathrm{task}} +
\lambda_{\mathrm{bce}} \mathcal{L}_{\mathrm{route}} +
0.01\,\mathcal{L}_{\mathrm{obs}}.
$$

No extra sparsification term is currently part of the described training path here.

## Inference

`--infer` does:

1. tokenize the prompt;
2. run `planner` once on the prompt;
3. build a sparse route per layer;
4. run `prefill()` over the prompt;
5. continue generation with `decode_step()`.

The cache is not a standard dense transformer KV cache. It stores the per-layer complex FHRR bus accumulated across positions.

## Evaluation and checks

### `--eval-ppl DIR`

Computes per-file perplexity over `.txt` files. Each chunk is routed the same way as at inference time.

### `--eval`

Runs the cluster suite:

- single-cluster perplexity;
- pair composition perplexity;
- triple composition perplexity;
- mixed composition perplexity;
- route summaries written to `cluster_eval.json`.

### `--smoke-test`

Checks:

- exact FHRR bind/unbind;
- equivalence of Hermitian and stacked-real scores;
- identity behavior when `alpha = 0`;
- Stage A/B/C forward/backward paths;
- equivalence of route forward and weighted forward;
- equivalence of incremental decode and full forward.

## CLI

Main commands:

```bash
python hive.py --prepare RAWDIR --config config.yaml --output PREP
python hive.py --train-tokenizer RAWDIR --config config.yaml --tokenizer-out TOKENIZER_JSON
python hive.py --pretrain-abi --config config.yaml --dataset PREP
python hive.py --train --config config.yaml --dataset PREP --stage A --layer 0 --cube 0
python hive.py --refine-clusters --config config.yaml --dataset PREP
python hive.py --train --config config.yaml --dataset PREP --stage B
python hive.py --train --config config.yaml --dataset PREP --stage C
python hive.py --infer --config config.yaml --checkpoint hive_runs/hive.pt --prompt "Hello"
python hive.py --eval-ppl EVAL_DIR --config config.yaml --checkpoint hive_runs/hive.pt
python hive.py --eval --config config.yaml --dataset PREP --checkpoint hive_runs/hive.pt
python hive.py --smoke-test
```

End-to-end run:

```bash
python hive.py --run-all RAWDIR --config config.yaml --output PREP
```

This runs:

- prepare
- tokenizer training if needed
- Stage 0
- all of Stage A
- optional refinement
- Stage B
- Stage C
- cluster evaluation

Additional CLI flags:

```bash
python hive.py --run-all RAWDIR --config config.yaml --output PREP --reset-out
python hive.py --run-all RAWDIR --config config.yaml --output PREP --refine-clusters
python hive.py --train --config config.yaml --dataset PREP --stage A --layer 0 --cube 0 --stage-a-out tmp.pt --stage-a-no-marker
python hive.py --infer --config config.yaml --checkpoint hive_runs/hive.pt --prompt "Hello" --max-new 128
python hive.py --prepare RAWDIR --config config.yaml --device cuda
```

## Output artifacts

Typical `out_dir`:

```text
out_dir/
  abi.pt
  hive.pt
  cluster_eval.json
  tokenizer.json
  .stage0_abi.done
  .stageA_l0_c0.done
  .stageB.done
  .stageC.done
```

Typical prepared dataset directory:

```text
prepared/
  tokens.pt
  clusters.pt
  doc_ids.pt
  concept_feats.npy
  meta.json
  tokens_offset_*.pt
  doc_ids_offset_*.pt
```

Clusterized shards:

```text
<clusterized_dir>/
  meta.json
  concept_feats.npy
  layer_0/
    meta.json
    cube_0/
      tokens.pt
      indices.pt
      meta.json
```

## Config

The YAML controls:

- number of layers;
- cubes per layer;
- `top_x`;
- widths (`d_cube`, `d_emb`, `d_router`, `d_ff`);
- cube and planner depth;
- sheaf and predictive-coding options;
- tokenizer choice;
- per-stage step counts;
- learning rates;
- gate parameters;
- prepare/eval limits;
- output directories.

If dimensions are set to `auto`, they are derived from vocabulary size and model structure.

## Dependencies

Minimum:

- `torch`
- `numpy`
- `pyyaml`

Optional depending on workflow:

- `transformers` for HuggingFace tokenizers
- `tokenizers` for local `tokenizer.json`
- `scikit-learn` for TF-IDF / SVD / KMeans
- `datasets`
