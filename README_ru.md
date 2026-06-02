# Hive

`hive.py` — это однофайловая реализация модульной causal language model, собранной из множества маленьких transformer-экспертов (`cubes`) и `planner`, который выбирает, какие эксперты активировать на каждом слое.

Логика обучения в коде такая:

- токенизировать и нарезать корпус на чанки;
- отдельно кластеризовать чанки для каждого Hive-слоя;
- предобучить общий токенный интерфейс;
- обучить каждый `cube` на своем шарде;
- обучить `planner` поверх замороженных кубов;
- аккуратно дообучить собранную систему целиком;
- на инференсе запускать разреженный граф экспертов, построенный один раз на prompt.

## Что есть в `hive.py`

Файл содержит:

- конфиг и автоподбор размерностей от размера словаря;
- FHRR holographic binding/unbinding для общей комплексной K/V-шины;
- `Cube`, базовый экспертный блок;
- `Planner`, модуль маршрутизации;
- `ABI`, общий интерфейс токенов и логитов;
- `HiveModel`, полную сборку модели;
- чтение датасета, chunking, clustering и экспорт шардов;
- обучение Stage A / B / C;
- refinement кластеров после Stage A;
- inference, perplexity evaluation, cluster evaluation и smoke tests;
- CLI для полного pipeline.

## Архитектура

Собранная модель состоит из:

- `ABI`
- `dense_pre`
- стека Hive-слоев
- `dense_post`
- опционального `PredictiveCodingHead`
- `planner`
- опционального `CellularSheaf` на каждом Hive-слое

Каждый `cube` получает общий residual stream, записывает свои K/V в общую FHRR-шину, читает шину через собственный slot, предлагает residual delta, пропускает ее через gate и добавляет обратно в поток.

Базовое обновление в коде:

$$
h \leftarrow h + w_{l,c}\, g_{l,c}(h,\Delta)\, \alpha_{l,c}\, \Delta_{l,c},
$$

где:

- $h$ — текущий residual stream;
- $w_{l,c}$ — вес куба $c$ на слое $l$, заданный planner;
- $g_{l,c}$ — token-level gate;
- $\alpha_{l,c}$ — обучаемый residual scale;
- $\Delta_{l,c}$ — дельта, предложенная кубом.

Если куб нерелевантен, малым становится либо $w_{l,c}$, либо $g_{l,c}$, и куб ведет себя почти как identity.

## ASCII схема

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

И один forward pass:

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

## Основные формулы

### FHRR bus

Slots — это комплексные phasor-векторы единичного модуля:

$$
s = e^{i\theta}.
$$

Binding и unbinding поэлементные:

$$
\mathrm{bind}(s, x) = s \odot x,
\qquad
\mathrm{unbind}(y, s) = y \odot \overline{s}.
$$

Поскольку $|s| = 1$, код опирается на точное равенство:

$$
\mathrm{unbind}(\mathrm{bind}(s, x), s) = x.
$$

Несколько кубов пишут в одну bus-шину суммированием:

$$
K_{\text{bus}} = \sum_{c \in A_l} w_{l,c}\, \mathrm{bind}(s_{l,c}, K_{l,c}),
\qquad
V_{\text{bus}} = \sum_{c \in A_l} w_{l,c}\, \mathrm{bind}(s_{l,c}, V_{l,c}).
$$

Куб $c$ читает свою проекцию через unbind:

$$
\widetilde{K}_{l,c} = \mathrm{unbind}(K_{\text{bus}}, s_{l,c}),
\qquad
\widetilde{V}_{l,c} = \mathrm{unbind}(V_{\text{bus}}, s_{l,c}).
$$

### ABI

Общий интерфейс задается так:

$$
e = \mathrm{Embed}(t),
\qquad
h_0 = P_{\text{in}} e.
$$

Выходная проекция связана с входной:

$$
P_{\text{out}} = P_{\text{in}}^\top.
$$

Тогда логиты равны:

$$
\ell = E^\top P_{\text{in}}^\top \mathrm{RMSNorm}(h),
$$

где $E$ — embedding matrix.

### Planner

`Planner` отображает embeddings и dense context в routing logits. После нескольких bidirectional-блоков он пулингует:

$$
\rho = \frac{1}{T}\sum_{t=1}^T z_t,
\qquad
\widehat{\rho} = \frac{\rho}{\|\rho\|}.
$$

Его обучаемая голова:

$$
\ell_{\text{learned}} = \widehat{\rho} W^\top + b.
$$

При включенном symbolic prior итоговые логиты кубов:

$$
\ell_c =
\ell_{\mathrm{learned},c} +
s_{\mathrm{prior}}
\frac{\cos(\widehat{\rho}, \widehat{C}_c)}
{\max(\tau_{\mathrm{prior}}, 0.1)}.
$$

где $C_c$ — centroid-derived concept vector куба $c$, $s_{\mathrm{prior}}$ — это `prior_scale`, а $\tau_{\mathrm{prior}}$ — это `prior_temp`.

После этого на каждом слое вычисляются route weights:

$$
w_l = \mathrm{sparsemax}(\ell_l)
\quad\text{или}\quad
w_l = \mathrm{softmax}(\ell_l),
$$

а `top_x` ограничивает число реально активных кубов на инференсе.

### Cube gate и residual delta

Для каждого активного куба:

$$
\Delta_{l,c} = f_{l,c}(h) - h.
$$

Token-level gate:

$$
g_{l,c}(h,\Delta) = \sigma\!\left(\frac{W_{l,c}[h;\Delta] + b_{l,c}}{T_{\text{gate}}}\right).
$$

Вклад куба:

$$
\delta h_{l,c} = w_{l,c}\, g_{l,c}(h,\Delta)\, \alpha_{l,c}\, \Delta_{l,c}.
$$

### Sheaf consensus

Если `use_sheaf=True`, код поднимает per-cube дельты в stalk-space и выполняет итеративную sheaf diffusion:

$$
X \leftarrow X - \varepsilon L_{\mathcal{F}} X.
$$

Для ребра $(u,v)$ discrepancy:

$$
d_{uv} = R_{uv} X_u - R_{vu} X_v.
$$

Код также считает obstruction-like energy:

$$
\mathcal{L}_{\text{obs}} \propto \sum_{(u,v)} \|d_{uv}\|^2.
$$

Она входит в Stage C как вспомогательный регуляризатор.

### Predictive coding head

Если `use_pc_head=True`, перед unembedding модель выполняет:

$$
z_0 = W_{\text{enc}} h,
$$

затем итерации

$$
\hat{h}_k = W_{\text{dec}} z_k,
\qquad
e_k = h - \hat{h}_k,
$$

$$
z_{k+1} =
z_k + \eta \left(e_k W_{\mathrm{dec}} - \lambda z_k\right).
$$

и на выходе

$$
h' = W_{\text{out}} z_K.
$$

## Подготовка данных

`--prepare` делает следующее:

1. Читает текст из:
   - HuggingFace Arrow datasets на диске;
   - или обычных файлов/директорий (`txt`, `md`, `jsonl`, `json`, `csv`, `tsv`).
2. Токенизирует документы.
3. Режет их на окна длины `seq_len`.
4. Строит дешевые TF-IDF/SVD признаки документов.
5. Независимо кластеризует чанки для каждого слоя в `cubes[layer]` кластеров.
6. Пишет:
   - `tokens.pt`
   - `clusters.pt`
   - `doc_ids.pt`
   - `concept_feats.npy`
   - per-cube shards вида `layer_X/cube_Y/tokens.pt`
7. При необходимости пишет сдвинутые chunk views для Stage A через `sliding_window_views`.

## Стадии обучения

### Stage 0: `--pretrain-abi --dataset PREP`

Обучает только:

- `ABI`
- `dense_pre`
- `dense_post`

Его loss в коде — обычный next-token cross-entropy плюс z-loss:

$$
\mathcal{L}_{\mathrm{stage0}} =
\mathcal{L}_{\mathrm{LM}} +
\lambda_z \mathbb{E}\!\left[\left(\log \sum_j e^{\ell_j}\right)^2\right].
$$

Артефакт:

- `out_dir/abi.pt`

### Stage A: `--train --stage A --dataset PREP --layer L --cube C`

Обучает один куб изолированно на его cluster shard.

Что важно в реализации:

- остальная модель заморожена;
- `abi.final_norm` можно опционально разморозить;
- positives идут из собственного кластера куба;
- negatives семплируются из других кубов того же слоя;
- используются nearest-cluster hard negatives;
- верхние слои могут получать cascaded inputs от самых сильных нижних кубов;
- опционально добавляется upstream noise, чтобы приблизить train-time input к deploy-time input.

Для слоев выше нулевого код оценивает масштаб входного шума:

$$
\sigma_l \approx
\gamma_{\mathrm{noise}}
\frac{\sqrt{n_{\mathrm{terms}}}}{\sqrt{2(l+1)}}
\mathrm{RMS}(h_0),
$$

где $\gamma_{\mathrm{noise}} =$ `stageA_noise_scale`.

Stage A loss:

$$
\mathcal{L}_A =
\mathcal{L}_{\mathrm{task}} +
\lambda_{\mathrm{margin}}^{\mathrm{eff}} \mathcal{L}_{\mathrm{margin}} +
\lambda_{\mathrm{open}} \mathcal{L}_{\mathrm{open}} +
\lambda_{\mathrm{close}} \mathcal{L}_{\mathrm{close}},
$$

где при `stage_a_margin_adaptive=True`

$$
\lambda_{\mathrm{margin}}^{\mathrm{eff}} =
\lambda_{\mathrm{margin}}
\frac{\mathcal{L}_{\mathrm{task}}}{\mathcal{L}_{\mathrm{task}} + 1}.
$$

Где:

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

Артефакт:

- merge в `out_dir/hive.pt`

Также есть параллельный Stage A через `parallel_cube_teach`.

### Cluster refinement: `--refine-clusters`

После Stage A код может переприсвоить каждый chunk тому кубу, у которого gate открывается сильнее всего:

$$
c^*(x) = \arg\max_c g_c(x).
$$

Это перезаписывает `clusters.pt`. Дальше по задумке нужно снова прогнать Stage A.

### Stage B: `--train --stage B --dataset PREP`

Обучает только `planner`, пока кубы заморожены.

Loss:

$$
\mathcal{L}_B =
\mathcal{L}_{\mathrm{task}} +
\lambda_{\mathrm{bce}} \mathcal{L}_{\mathrm{route}} +
\lambda_{\mathrm{cap}} \mathcal{L}_{\mathrm{cap}} +
\lambda_{\mathrm{balance}} \mathcal{L}_{\mathrm{bal}}.
$$

Route supervision — это per-layer cross-entropy против подготовленных cluster labels:

$$
\mathcal{L}_{\mathrm{route}} =
\frac{1}{L}\sum_{l=1}^{L} \mathrm{CE}(\ell_l, y_l).
$$

Load-balance term использует средние фактические route weights:

$$
\mathcal{L}_{\mathrm{bal}} =
\sum_l \sum_c \bar{w}_{l,c} \log\!\left(\bar{w}_{l,c} |C_l| + 10^{-9}\right).
$$

Route-cap penalty:

$$
\mathcal{L}_{\mathrm{cap}} =
\sum_l \mathbb{E}\!\left[\max(0, |A_l| - \mathrm{top\_x})^2\right].
$$

Stage B использует differentiable sparse routing через `route_weight_matrix(..., soft_topk=True)`.

### Stage C: `--train --stage C --dataset PREP`

Это joint fine-tune для:

- параметров planner;
- gate-параметров кубов и `alpha`;
- core-параметров кубов;
- опциональных sheaf-параметров;
- опционального predictive-coding head.

Loss в коде:

$$
\mathcal{L}_C =
\mathcal{L}_{\mathrm{task}} +
\lambda_{\mathrm{bce}} \mathcal{L}_{\mathrm{route}} +
0.01\,\mathcal{L}_{\mathrm{obs}}.
$$

Дополнительный sparsification term в описываемом тренировочном пути здесь не учитывается.

## Инференс

`--infer` делает:

1. токенизирует prompt;
2. один раз прогоняет `planner` по prompt;
3. строит sparse route на каждом слое;
4. делает `prefill()` по prompt;
5. продолжает генерацию через `decode_step()`.

Кеш здесь не dense transformer KV cache. Он хранит накопленную по позициям комплексную FHRR bus на каждом слое.

## Evaluation и проверки

### `--eval-ppl DIR`

Считает per-file perplexity по `.txt` файлам. Каждый chunk маршрутизируется так же, как на обычном инференсе.

### `--eval`

Запускает cluster suite:

- single-cluster perplexity;
- pair composition perplexity;
- triple composition perplexity;
- mixed composition perplexity;
- route summaries в `cluster_eval.json`.

### `--smoke-test`

Проверяет:

- exact FHRR bind/unbind;
- эквивалентность Hermitian и stacked-real scores;
- identity-поведение при `alpha = 0`;
- forward/backward paths для Stage A/B/C;
- эквивалентность route forward и weighted forward;
- эквивалентность incremental decode и full forward.

## CLI

Основные команды:

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

Сквозной запуск:

```bash
python hive.py --run-all RAWDIR --config config.yaml --output PREP
```

Он делает:

- prepare
- tokenizer training при необходимости
- Stage 0
- весь Stage A
- optional refinement
- Stage B
- Stage C
- cluster evaluation

Дополнительные CLI-флаги:

```bash
python hive.py --run-all RAWDIR --config config.yaml --output PREP --reset-out
python hive.py --run-all RAWDIR --config config.yaml --output PREP --refine-clusters
python hive.py --train --config config.yaml --dataset PREP --stage A --layer 0 --cube 0 --stage-a-out tmp.pt --stage-a-no-marker
python hive.py --infer --config config.yaml --checkpoint hive_runs/hive.pt --prompt "Hello" --max-new 128
python hive.py --prepare RAWDIR --config config.yaml --device cuda
```

## Выходные артефакты

Типичный `out_dir`:

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

Типичный prepared dataset directory:

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

## Конфиг

YAML управляет:

- числом слоев;
- числом кубов на слой;
- `top_x`;
- ширинами (`d_cube`, `d_emb`, `d_router`, `d_ff`);
- глубиной кубов и planner;
- опциями sheaf и predictive coding;
- выбором tokenizer;
- количеством шагов по стадиям;
- learning rates;
- параметрами gate;
- prepare/eval limits;
- output directories.

Если размерности стоят в `auto`, они выводятся из vocabulary size и структуры модели.

## Зависимости

Минимум:

- `torch`
- `numpy`
- `pyyaml`

Опционально по workflow:

- `transformers` для HuggingFace tokenizers
- `tokenizers` для локального `tokenizer.json`
- `scikit-learn` для TF-IDF / SVD / KMeans
- `datasets` и `pyarrow` для Arrow datasets

## Короткое чтение

На высоком уровне `hive.py` — это не "один большой transformer". Это pipeline:

- построить общее пространство представлений;
- независимо обучить экспертов;
- обучить router поверх этих экспертов;
- дообучить собранную систему;
- на инференсе запускать только sparse subset экспертов.

Именно такую систему реально реализует файл.
