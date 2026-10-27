# CELL 04b — Visualise One Sample Image

> **Note.** These study notes were first written for an earlier Flickr8k / Google Colab
> version of our notebook. The code they explain is the same as in
> [`show_attend_tell_flickr30k.ipynb`](../../notebooks/show_attend_tell_flickr30k.ipynb),
> but dataset sizes, paths, the optimiser and the example outputs can differ from our final
> Flickr30k run. See the [README](../../README.md) for the actual Flickr30k results.

## WHAT IS THE PROBLEM THIS CELL SOLVES?

You have loaded 40,455 rows of data and cleaned all the text. But have you actually
verified that the images open correctly? That the captions look sensible? That the
filenames match the images? This cell answers all of those questions with one look.

## WHAT DOES THIS CELL ACTUALLY DO?

Takes the very first image in the dataset, opens it from disk using PIL,
displays it using matplotlib, and prints all 5 of its human-written captions
underneath. That is all. No model, no training, no computation.

## WHY IS THIS USEFUL?

Before spending hours training you want to verify three things:

1. Images open without errors — the path is correct and files are not corrupted.
2. Captions look clean — the cleaning step did not break anything.
3. The right captions are matched to the right image — the DataFrame is correct.

If any of these are wrong, you see the problem immediately here instead of
discovering it after 3 hours of training produces garbage output.

## WHAT THE OUTPUT TEACHES YOU

The image shows a wooden playhouse with stairs. Five different humans described
the exact same scene and produced five different captions:

```text
Caption 1  "a child in a pink dress is climbing up a set of stairs in an entry way"
Caption 2  "a girl going into a wooden building"
Caption 3  "a little girl climbing into a wooden playhouse"
Caption 4  "a little girl climbing the stairs to her playhouse"
Caption 5  "a little girl in a pink dress going into a wooden cabin"
```

Notice: one annotator says "cabin", another says "playhouse", another says
"building". All are valid descriptions of the same image. There is no single
correct caption. This is fundamental to understanding how BLEU evaluation works —
it compares the model output against ALL 5 references simultaneously, not just one.
If the model generates "a little girl near a wooden structure" it should score
reasonably against all five even though it matches none of them exactly.

## PAPER REFERENCE

This cell directly illustrates what Figure 4 in the paper shows — images paired
with captions. The variety across 5 human annotations is exactly why Section 5.1
specifies using all 5 references for BLEU evaluation.
