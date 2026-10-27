# CELL 01 — Imports & Global Setup

> **Note.** These study notes were first written for an earlier Flickr8k / Google Colab
> version of our notebook. The code they explain is the same as in
> [`show_attend_tell_flickr30k.ipynb`](../../notebooks/show_attend_tell_flickr30k.ipynb),
> but dataset sizes, paths, the optimiser and the example outputs can differ from our final
> Flickr30k run. See the [README](../../README.md) for the actual Flickr30k results.

## WHAT IS THE PROBLEM THIS CELL SOLVES?

Before doing anything — loading data, building a model, training — Python needs to
know which tools to use. Every library must be imported first. Also, deep learning
involves a lot of randomness (random weight initialisation, random data shuffling).
Without fixing the random seed, running the notebook twice would give different
results, making it impossible to compare experiments fairly.

## WHAT DOES THIS CELL ACTUALLY DO?

Two things:

```text
1. Loads every library the notebook will use.
   Think of it like gathering all your tools on the table before starting work.
   If a library is missing here, every cell that uses it will crash.
```

```text
2. Fixes the random seed to 42 and detects the GPU.
   The seed is passed to Python's random module, NumPy, and PyTorch (both CPU and
   GPU). This means every "random" choice made later — weight initialisation, data
   shuffling, dropout — will be identical on every run.
```

## KEY LIBRARIES AND WHY WE NEED THEM

```text
torch / nn / F      The deep learning engine. nn contains all layer types (Linear,
                    LSTM, Conv2d). F contains functions like softmax and relu.
```

```text
torchvision         Pre-trained computer vision models. We use VGG19 as the encoder.
```

```text
transforms          Image preprocessing pipeline. Resizes, crops, and normalises
                    images so they match what VGG19 expects.
```

```text
Dataset / DataLoader PyTorch tools for packaging data into batches and feeding them
                    to the model efficiently during training.
```

```text
pad_sequence        Pads captions of different lengths to the same length within
                    a batch so they can be stacked into a single tensor.
```

```text
nltk                Natural Language Toolkit. Used later to compute BLEU and METEOR
                    evaluation scores.
```

```text
tqdm                Draws a progress bar during training so you can see how many
                    batches are done and how long is left.
```

## PAPER REFERENCE

The paper trained on an NVIDIA Titan Black GPU (Section 4.3). Colab T4 is a modern
equivalent and is sufficient for Flickr8k.

## EXPECTED OUTPUT

```text
CUDA available : True
Device         : cuda
GPU            : Tesla T4
```

If CUDA available shows False, the GPU is not enabled in Colab.
Go to Runtime → Change runtime type → T4 GPU and restart.
