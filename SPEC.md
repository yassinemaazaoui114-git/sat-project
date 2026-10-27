# 📌 PROJECT SPEC — Show, Attend and Tell (SAT) Reproduction

> **Version:** 2.1
> **Last updated:** 2026-10-27
> **Status:** 🟢 Modeling done (in a Kaggle notebook) · 🟢 Repo built and pushed to GitHub (scope A, §8) · 🟢 Class submission finished

Legend: ✅ **Confirmed** (verified in the files) · ❓ **Assumption** (not confirmed yet)

---

## 1. Project context

| Item | Value | Status |
|---|---|---|
| Paper | *Show, Attend and Tell: Neural Image Caption Generation with Visual Attention*, Xu et al., ICML 2015 ([arXiv 1502.03044](https://arxiv.org/abs/1502.03044)) | ✅ |
| Companion paper | *Reproducing "Show, Attend and Tell"*, Liu & Brailsford, J. Phys.: Conf. Ser. 2589 (2023) 012012, open access (CC BY 3.0) | ✅ |
| Type | University deep learning project (paper reproduction) | ✅ |
| Academic deliverables | Annotated notebook + report + slides | ✅ (from spec v1) |
| Academic deadline | Week 14 of the semester; spec v1 written 2026-04-23, so most likely already passed | ❓ |
| Team | Ayoub Rebhi, Mahmoud Ben Khelil, Medali Mraihi, Molka Boukhris, Semah Ayachi, Yassine Maazaoui (same team as RLT) | ✅ |
| Current goal | A clean GitHub repo for the CV | ✅ |
| GitHub repo | `yassinemaazaoui114-git/sat-project` (contains only spec v1) | ✅ |
| Local folder | `C:\Users\mam83\OneDrive\Desktop\ML projects\2- Deep Learning - SAT` | ✅ |

---

## 2. What exists today

### Local folder

| File | What it is | Status |
|---|---|---|
| `deep-learning 30K.ipynb` (11 MB) | **The full implementation.** 72 cells, run on Kaggle with a Tesla T4 GPU, outputs and 11 figures saved | ✅ |
| `Cell Explainer/` (12 `.txt` files) | Plain-text explanations of notebook cells 1–13 (study / report notes). **Written for the Flickr8k / Colab variant**: same model code, but Flickr8k sizes, RMSProp, vocab 2,660, `/content` paths (see issue #7) | ✅ |
| `Article Deep Learning.pdf` | The original SAT paper | ✅ |
| `Liu_2023.pdf` | The reproduction paper (Liu & Brailsford 2023) | ✅ |
| `.ipynb_checkpoints/` | Jupyter autosave, not needed | ✅ |
| `DL_Flicker_8K.ipynb` (Colab, shared separately) | A copy of the same notebook pointed at **Flickr8k**. Only the data and training cells were actually run on Flickr8k (best **validation** BLEU-4 = 21.22, early stop at epoch 14, ~5.5 min/epoch). **Every output after training (test BLEU, captions, hard attention, figures) is a stale copy from the Flickr30k run** (Flickr30k image IDs, 3,179 test images, `/kaggle` paths). Decision: **not used in the repo** | ✅ |

### GitHub repo
On GitHub: only `SPEC.md` v1 (1 commit, 2026-04-23). Its plan (Flickr8k, ResNet-101, soft attention only, 7 phases, `src/` modules) **no longer matches** what was actually built. ✅

Local clone (`sat-project/`, not yet committed): README, LICENSE, requirements, `.gitignore`, this spec, the cleaned notebook, 11 exported figures and the 12 cell notes as Markdown (layout in §6). ✅

### Not in the folder

| Item | Where it is | Status |
|---|---|---|
| Trained weights: `best_model.pth` (113 MB, soft) and `best_hard_attention_model.pth` (116 MB, hard) | **Lost**: no longer on Kaggle | ✅ |
| `vocab.json`, `test_caption_predictions.csv`, `evaluation_results.json`, attention PNGs, comparison CSVs | **Lost** as files, but the numbers and figures are still embedded in the notebook outputs | ✅ |
| Report and slides | Not found | ❓ |

---

## 3. What the notebook implements ✅

| Part | Implementation |
|---|---|
| Dataset | **Flickr30k** (Kaggle), split **by image** 80 / 10 / 10 (3,179 test images), `random_state` seed |
| Text | Lowercased and cleaned; vocabulary built from training captions only (min frequency 5, max length 30); `<pad> <start> <end> <unk>` tokens |
| Batching | Length-based bucketing sampler, as in the paper |
| Encoder | **VGG19**, ImageNet-pretrained and frozen, trailing max-pool dropped so the map is **14×14×512**, giving 196 annotation vectors (shape checked in the notebook) |
| Soft attention | Additive attention + LSTM decoder with the **β gate**, plus **doubly-stochastic regularisation** (λ = 1.0) |
| Hard attention | `HardAttentionDecoder`: Categorical sampling + **REINFORCE** with a moving-average baseline + entropy bonus + soft-mix trick; paper-style deep output layer |
| Training | Adam, lr 4e-4, grad clip 5, dropout 0.5, embed 256 / decoder 512 / attention 256; early stopping on **validation BLEU-4** (patience 3). Soft: 8 epochs (~20 min each); hard: 5 epochs (~22 min each) |
| Evaluation | Corpus BLEU-1…4 (NLTK, smoothing method1) + METEOR (NLTK) + average caption length; greedy decoding. Validation BLEU for early stopping uses a fixed 200-image subset of the validation set |
| Visualisation | Per-word 14×14 attention maps, upsampled and smoothed (Gaussian), overlaid on the image, for both models |
| Error analysis (from Liu 2023) | VGG19 top-5 ImageNet labels; wrong noun/verb analysis with POS tags; GPT-2 caption scoring; dependency parsing (spaCy); YOLO planned but skipped (unavailable on Kaggle) |
| Reporting | Paper-vs-notebook comparison table and a report-ready summary |

---

## 4. Results ✅

**Test set** (Flickr30k, 3,179 images). Paper numbers are from Table 1 of Xu et al. (2015):

| Model | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR |
|---|---|---|---|---|---|
| **Ours, soft attention** | 65.50 | 44.05 | 29.84 | **20.52** | 39.70* |
| Paper, soft attention | 66.7 | 43.4 | 28.8 | 19.1 | 18.49 |
| **Ours, hard attention** | 62.53 | 39.50 | 25.17 | **16.42** | 35.09* |
| Paper, hard attention | 66.9 | 43.9 | 29.6 | 19.9 | 18.46 |

Validation BLEU-4 per epoch: soft attention went from 17.15 to a best of **22.58 at epoch 6**; hard attention reached a best of **16.95 at epoch 2** and stopped early at epoch 5.

**How to read these numbers (must be stated in the README):**
- \*Our **METEOR is not comparable** with the paper's. NLTK's METEOR is a different implementation from the official METEOR 1.5 the paper used, and it scores much higher.
- BLEU is only **roughly comparable**. We use a random 80/10/10 split instead of the Karpathy split, NLTK smoothing, and greedy decoding. "Soft attention matches or beats the paper on BLEU-2 to BLEU-4" is fair to say only with these caveats.
- Hard attention is **below the paper**: 5 epochs on a T4 GPU versus days of training on a Titan Black.

---

## 5. Issues found ✅

1. ✅ *Fixed in the repo copy.* **Notebook cell 37 misquotes the paper.** It says the paper's soft attention gets BLEU-4 ≈ 19.9, but 19.9 is **hard** attention; soft is **19.1**. It also says "we expect lower numbers", but the soft model actually scored higher.
2. ✅ *Fixed.* **Noise in the outputs:** repeated `AssertionError: can only test a child process` messages, a harmless DataLoader-worker warning on Kaggle (3 stderr outputs, in cells 28 and 60, removed).
3. ✅ *Fixed.* **Title and intro** say "(Soft Attention)", but the notebook also implements hard attention. Cells 38 ("left as future work") and 51 (conclusion: "focuses on soft attention") had the same stale claim and were fixed too.
4. ⏸️ *Kept as is.* **Paths are Kaggle-only** (`/kaggle/input`, `/kaggle/working`). The README explains how to run on Kaggle and what to change elsewhere.
5. ✅ *Closed.* **The weights (113 MB + 116 MB) are too big for normal GitHub** (100 MB file limit). Moot: the weights are lost. `*.pth` is in `.gitignore`.
6. ✅ *Fixed.* **Spec v1 is out of date:** Flickr8k → Flickr30k, ResNet-101 → VGG19, soft only → soft + hard.
7. ⚠️ **The Cell Explainer notes describe the Flickr8k / Colab variant**, not the final Flickr30k run (8,091 images, vocab 2,660, RMSProp, `/content` paths, Flickr8k example outputs). The model code they explain is the same. They were copied as written, with a disclaimer at the top of each note and in `docs/cell_explanations/README.md`.

---

## 6. Repo layout (built, scope A)

```
sat-project/
├── README.md                 ← results vs paper, method, figures, how to run
├── SPEC.md                   ← this file
├── LICENSE                   ← MIT, the 6 team members
├── requirements.txt          ← from the notebook's imports
├── .gitignore                ← *.pth, .ipynb_checkpoints/, __pycache__/
├── notebooks/
│   └── show_attend_tell_flickr30k.ipynb   ← cleaned version of "deep-learning 30K.ipynb" (not re-run)
└── docs/
    ├── cell_explanations/    ← the 12 Cell Explainer notes as .md + an index README
    └── figures/              ← the 11 figures exported from the notebook outputs
```

No `results/` (the Kaggle output files are lost; the numbers are in the notebook and README) and no `src/` (scope A).

---

## 7. Open questions (need your decision)

| # | Question | Options | My suggestion |
|---|---|---|---|
| Q1 | ~~Team members?~~ | ✅ Answered: same team as RLT (see §1) | — |
| Q2 | ~~Is the class submission finished?~~ | ✅ Answered: **Yes**, so this repo is CV polish only (no report/slides work) | — |
| Q3 | ~~Can you still get the Kaggle outputs?~~ | ✅ Answered: **No**, they're gone from Kaggle. Results and figures exist only inside the notebook | — |
| Q4 | ~~Where do the weights go?~~ | ✅ Closed: there are no weights to publish. The README says they can be retrained with the notebook (~4.5 h on a T4) | — |
| Q5 | ~~Scope~~ | ✅ **A:** cleaned notebook + README. B/C (modules, demo) later; a demo would need retraining first | — |
| Q6 | ~~Include the Cell Explainer notes?~~ | ✅ Answered: **Yes**, in `docs/cell_explanations/` as Markdown (with a Flickr8k disclaimer, see issue #7) | — |
| Q7 | ~~Include the papers' PDFs?~~ | ✅ Answered: **link both, include neither** (arXiv + PMLR for SAT, DOI for Liu 2023) | — |

---

## 8. Plan once decided (scope A)

- [x] 1. Clean the notebook (JSON edit, not re-run): fix cell 37, fix the title/intro (and the stale "soft only" text in cells 38 and 51), remove the noisy error outputs, keep the results and figures.
- [x] 2. Export the figures to `docs/figures/` (all 11).
- [x] 3. Copy the Cell Explainer notes to `docs/cell_explanations/` (as Markdown).
- [x] 4. Write the README: paper summary, architecture, results vs paper (with the caveats in §4), soft vs hard, attention figures, error analysis, how to run on Kaggle, structure, citations.
- [x] 5. Add `requirements.txt`, the MIT LICENSE and `.gitignore`.
- [x] 6. ~~Publish the weights~~ → no weights exist; the README explains how to retrain them with the notebook.
- [x] 7. Update this spec.
- [x] 8. Commit to `main` and push (approved by Yassine).

Same approach as RLT: a Claude Code prompt run in a local clone, then pushed.

---

## 9. Change log

| Date | Change | Why |
|---|---|---|
| 2026-04-23 | v1 written: Flickr8k, ResNet-101, soft attention only, 7 phases | Initial plan |
| 2026-10-27 | **v2:** rewritten from the actual notebook. Dataset → Flickr30k, encoder → VGG19 (as in the paper), hard attention + Liu 2023 error analysis added, real results recorded, issues and open questions listed | The implementation went beyond v1, and v1 no longer matched the code |
| 2026-10-27 | Team confirmed (same 6 members as RLT); Q1 closed | Answered by Yassine |
| 2026-10-27 | Kaggle outputs lost → no weights; scope A chosen; Q3–Q5 closed | Answered by Yassine; short on time |
| 2026-10-27 | Flickr8k Colab notebook reviewed and excluded | Its post-training outputs are copies of the 30k run, so they aren't real 8k results |
| 2026-10-27 | **v2.1:** repo built locally (plan steps 1–7): cleaned notebook, 11 figures, cell notes as Markdown, README, LICENSE, requirements, `.gitignore`. Issues #1–3, #5, #6 marked fixed; Q6, Q7 closed | Scope A carried out; nothing was re-trained or re-run |
| 2026-10-27 | Notebook cells 38 and 51 also corrected (besides 0 and 37) | They still said hard attention was "future work" / not implemented, which contradicts Sections 17–21 |
| 2026-10-27 | Issue #7 added: the Cell Explainer notes describe the Flickr8k/Colab variant; disclaimer added to each note | Found while converting them; copied as written so as not to rewrite the team's notes |
| 2026-10-27 | §3 and README note that validation BLEU uses a fixed 200-image subset | Found in the code (`VAL_BLEU_NSAMPLES = 200`); explains why val BLEU-4 (22.58) is above test (20.52) |
| 2026-10-27 | Q2 closed (class submission finished); repo committed and pushed to `main` | Answered by Yassine |
