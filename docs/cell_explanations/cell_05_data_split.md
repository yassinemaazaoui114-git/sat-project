# CELL 05 — Train / Validation / Test Split

> **Note.** These study notes were first written for an earlier Flickr8k / Google Colab
> version of our notebook. The code they explain is the same as in
> [`show_attend_tell_flickr30k.ipynb`](../../notebooks/show_attend_tell_flickr30k.ipynb),
> but dataset sizes, paths, the optimiser and the example outputs can differ from our final
> Flickr30k run. See the [README](../../README.md) for the actual Flickr30k results.

## WHAT IS THE PROBLEM THIS CELL SOLVES?

We have 8,091 images. The model needs to learn from some of them, be monitored
on others during training, and be tested on a completely separate set at the end.
If the model is tested on images it has already seen during training, the test
results are meaningless — the model has just memorised them.

## WHAT DOES THIS CELL ACTUALLY DO?

```text
Splits the 8,091 unique images into three non-overlapping groups:
- 80% for training   (6,472 images, 32,360 captions)
- 10% for validation (810 images, 4,050 captions)
- 10% for testing    (809 images, 4,045 captions)
```

Then builds a separate DataFrame for each group.

## THE CRITICAL DESIGN DECISION — SPLIT BY IMAGE NOT BY ROW

The dataset has 40,455 rows (8,091 images × 5 captions each).

WRONG approach: split the 40,455 rows randomly.
Result: the same image might have 3 captions in train and 2 in test.
The model would have seen that image during training.
Test scores would be artificially inflated — the model is not generalising,
it is partially memorising.

CORRECT approach: split the 8,091 unique image IDs first.
Then assign all 5 captions of each image to the same split.
This guarantees zero overlap between train, val, and test.

## WHY THREE SPLITS AND NOT TWO?

```text
Train    The model updates its weights based on this data only.
```

```text
Val      Checked after every epoch. The BLEU-4 score here tells you
         whether the model is improving or has plateaued. Early stopping
         watches this. The model never trains on val — it only evaluates on it.
         This prevents overfitting to the training set.
```

```text
Test     Touched exactly once at the very end. This is the honest score.
         If you look at test results during development and make decisions
         based on them, you have effectively trained on test and the score
         is no longer trustworthy.
```

## PAPER REFERENCE

Section 5.2: "In our reported results, we use the predefined splits of Flickr8k."
The paper uses official Karpathy splits. This notebook uses a random 80/10/10
split instead — a standard and acceptable alternative. Results may differ
slightly from Table 1 of the paper for this reason.

## EXPECTED OUTPUT

```text
Unique images   : 8091
  train images  : 6472  (32360 captions)
  val   images  : 810   (4050 captions)
  test  images  : 809   (4045 captions)
```
