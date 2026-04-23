# 📌 PROJECT SPEC — Show, Attend and Tell (SAT) Reproduction
> **Version:** 1.0  
> **Created:** 2026-04-23  
> **Author:** [Your Name]  
> **Status:** 🟡 In Progress — Phase 0 not started  

---

## 1. Project Context

### Academic Objective
Reproduce the paper **"Show, Attend and Tell: Neural Image Caption Generation with Visual Attention"** (Xu et al., 2015, ICML) as part of a Deep Learning university project.

- **Validation deadline:** Week 14 of the semester
- **Deliverables (academic):** Annotated Jupyter Notebook + Report + Slides
- **Deliverable (personal):** A working web app prototype (post-academic)

### Personal Objective
Build a fully functional, well-understood image captioning system that can be turned into a real web application. Every line of code must be understood — no black boxes.

---

## 2. The Paper — Summary

### What it does
Given an input image, the model automatically generates a natural language caption describing its content. While generating each word, the model **attends** to a different region of the image — mimicking how humans scan a scene.

### Key Idea: Visual Attention
Instead of compressing the whole image into a single vector, the model uses an **attention mechanism** that produces a weighted map over image regions at each decoding step. This means when it generates the word "bird", it's actually looking at the bird region.

### Two Attention Variants
| Variant | Type | Training |
|---|---|---|
| **Soft Attention** | Deterministic — weighted average of all regions | Standard backpropagation |
| **Hard Attention** | Stochastic — picks one region at a time | REINFORCE (RL-based) |

> We will implement **Soft Attention** — it is differentiable and trainable end-to-end. Hard Attention is much harder to train and not required for the project.

### Architecture Pipeline
```
Input Image
    ↓
[Encoder] Pretrained CNN (ResNet-101)
    → Produces a 14×14 feature map (196 spatial locations × 2048 features)
    ↓
[Attention] At each timestep t:
    → Computes a weight αti for each of the 196 locations
    → Weighted sum = context vector ẑt
    ↓
[Decoder] LSTM
    → Takes: context vector + previous word embedding + previous hidden state
    → Outputs: probability distribution over vocabulary
    → Picks the most likely next word
    ↓
Output Caption (word by word)
```

### Datasets Used in Paper
| Dataset | Images | Captions/image | Size |
|---|---|---|---|
| Flickr8k | 8,000 | 5 | ~1GB |
| Flickr30k | 31,000 | 5 | ~3GB |
| MS COCO | ~120,000 | 5 | ~20GB |

> **Our strategy:** Train and evaluate on **Flickr8k** (fits in free Colab). Describe Flickr30k and COCO in the notebook with explanations of what would differ.

### Evaluation Metrics
- **BLEU-1 to BLEU-4** — n-gram precision between generated and reference captions
- **METEOR** — considers synonyms and morphological variants
- **Loss curves** — cross-entropy training/validation loss

---

## 3. Technical Stack

| Tool | Purpose |
|---|---|
| Python 3.10+ | Language |
| PyTorch | Deep learning framework |
| torchvision | Pretrained ResNet, image transforms |
| NLTK | BLEU score computation |
| Matplotlib / Seaborn | Plots and attention visualizations |
| Pillow (PIL) | Image handling |
| Google Colab | Training environment (free GPU) |
| Google Drive | Persistent file storage |
| Git / GitHub | Version control |
| Gradio | Demo interface (Colab-hosted) |
| FastAPI + React (later) | Web app prototype (post-academic) |

---

## 4. Project Structure

```
sat-project/
│
├── SPEC.md                  ← This file
├── PHASE_0.md               ← Phase notes
├── PHASE_1.md
├── ...
│
├── notebooks/
│   └── SAT_main.ipynb       ← The annotated deliverable notebook
│
├── src/                     ← Clean Python modules (for web app later)
│   ├── dataset.py
│   ├── encoder.py
│   ├── attention.py
│   ├── decoder.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
│
├── data/
│   ├── flickr8k/
│   │   ├── images/
│   │   └── captions/
│   └── README.md            ← How to download datasets
│
├── checkpoints/             ← Saved model weights (.pth files)
│
├── results/
│   ├── plots/               ← Loss curves, BLEU charts
│   └── attention_maps/      ← Visualization outputs
│
└── report/
    ├── report.pdf
    └── slides.pptx
```

---

## 5. The 7 Phases

### Phase 0 — Setup & PyTorch Foundations
**Goal:** Get the environment running. Understand PyTorch tensors, autograd, and the training loop pattern before writing any model code.  
**Key concepts:** tensors, .backward(), DataLoader, GPU/CPU device management, Google Drive mounting  
**Output:** Working Colab notebook shell, Drive folder structure, PyTorch basics cell  
**Status:** 🔴 Not started  
**File:** `PHASE_0.md`

---

### Phase 1 — Dataset Loading & Preprocessing
**Goal:** Download Flickr8k. Parse captions. Build vocabulary. Write a PyTorch Dataset class. Visualize samples.  
**Key concepts:** tokenization, vocabulary building, `<START>` / `<END>` / `<PAD>` / `<UNK>` tokens, DataLoader batching  
**Output:** `dataset.py`, working DataLoader, vocab stats  
**Status:** 🔴 Not started  
**File:** `PHASE_1.md`

---

### Phase 2 — Encoder (CNN Feature Extractor)
**Goal:** Load pretrained ResNet-101. Remove the classification head. Extract 14×14 spatial feature maps. Understand what these feature maps represent.  
**Key concepts:** transfer learning, feature maps vs. pooled vectors, fine-tuning vs. freezing  
**Output:** `encoder.py`, shape verification tests  
**Status:** 🔴 Not started  
**File:** `PHASE_2.md`

---

### Phase 3 — Attention Mechanism
**Goal:** Implement soft attention. At each timestep, compute attention weights over the 196 image regions. Visualize the weights as a heatmap.  
**Key concepts:** additive attention (Bahdanau), softmax normalization, context vector, doubly stochastic regularization  
**Output:** `attention.py`, attention heatmap visualization  
**Status:** 🔴 Not started  
**File:** `PHASE_3.md`

---

### Phase 4 — Decoder (LSTM + Attention)
**Goal:** Wire the LSTM with the attention mechanism. Implement teacher forcing. Build the baseline model (LSTM without attention) for comparison.  
**Key concepts:** LSTM gates, teacher forcing, embedding layer, output projection  
**Output:** `decoder.py`, baseline_decoder.py  
**Status:** 🔴 Not started  
**File:** `PHASE_4.md`

---

### Phase 5 — Training Loop
**Goal:** Write the full training loop. Track train/val loss. Save best checkpoints. Handle Colab disconnection (saving to Drive).  
**Key concepts:** cross-entropy loss, Adam optimizer, gradient clipping, learning rate scheduling, early stopping  
**Output:** `train.py`, loss curve plots, saved `.pth` checkpoint  
**Status:** 🔴 Not started  
**File:** `PHASE_5.md`

---

### Phase 6 — Evaluation & Visualization
**Goal:** Compute BLEU scores. Generate captions on test images. Visualize attention maps overlaid on images. Compare SAT vs. baseline.  
**Key concepts:** BLEU-1 to BLEU-4, greedy decoding vs. beam search, attention map upsampling  
**Output:** `evaluate.py`, BLEU comparison table, attention visualizations  
**Status:** 🔴 Not started  
**File:** `PHASE_6.md`

---

## 6. How to Resume This Project in a New Conversation

If this conversation is lost, start a new one with Claude and paste the following:

```
I am working on reproducing the "Show, Attend and Tell" paper for a Deep Learning project.
Here is my full project spec: [paste SPEC.md content]
Here is my current phase file: [paste current PHASE_X.md content]

I am at: [describe exactly where you stopped]
My current question / next step is: [describe it]
```

---

## 7. Key Decisions Log

| Date | Decision | Reason |
|---|---|---|
| 2026-04-23 | Use Flickr8k only for training | Free Colab memory/time constraints |
| 2026-04-23 | Implement Soft Attention only | Differentiable, standard backprop, easier to debug |
| 2026-04-23 | Use ResNet-101 as encoder | Better than VGG for feature quality, widely available pretrained |
| 2026-04-23 | Baseline model = LSTM without attention | Direct comparison to prove attention's value |
| 2026-04-23 | Framework = PyTorch | Best ecosystem for this paper's reproduction |
| 2026-04-23 | Demo = Gradio first, then FastAPI+React | Gradio for academic demo, web app is personal project after |

---

## 8. Resources

| Resource | Link |
|---|---|
| Original paper | https://arxiv.org/abs/1502.03044 |
| Flickr8k dataset | Kaggle or professor's link |
| PyTorch docs | https://pytorch.org/docs |
| BLEU evaluation | NLTK corpus.bleu |
| Reference implementation | https://github.com/kelvinxu/arctic-captions (original, Theano) |
| PyTorch reference | https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning |

---

*Last updated: 2026-04-23 | Next step: Start PHASE_0.md*
