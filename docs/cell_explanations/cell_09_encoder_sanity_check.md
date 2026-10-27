# CELL 09 — Sanity Check: Encoder Shape Verification

> **Note.** These study notes were first written for an earlier Flickr8k / Google Colab
> version of our notebook. The code they explain is the same as in
> [`show_attend_tell_flickr30k.ipynb`](../../notebooks/show_attend_tell_flickr30k.ipynb),
> but dataset sizes, paths, the optimiser and the example outputs can differ from our final
> Flickr30k run. See the [README](../../README.md) for the actual Flickr30k results.

## WHAT IS THE PROBLEM THIS CELL SOLVES?

The encoder architecture was just defined in the previous cell. But defining
a class does not prove it works correctly. The shapes of the intermediate tensors
must exactly match what the paper describes. If something is wrong — for example
if the MaxPool removal did not work and the feature map is 7×7 instead of 14×14 —
you want to know NOW, not after 3 hours of training a broken model.

## WHAT DOES THIS CELL ACTUALLY DO?

Instantiates a temporary encoder, runs two real images through it, and prints
the shape of every intermediate tensor. Then checks two hard assertions:

- Raw feature map must be (B, 512, 14, 14)
- Annotation vectors must be (B, 196, 512)

If either assertion fails, the cell crashes immediately with a clear error.
If both pass, the encoder is confirmed correct and the temporary model is
deleted to free GPU memory.

## WHY IS THIS IMPORTANT?

Shape errors in deep learning are silent killers. A wrong shape does not always
cause an immediate crash — sometimes it propagates through many cells and only
surfaces as a confusing error deep inside the training loop. Catching it here
with an explicit assert guarantees the rest of the notebook builds on correct
foundations.

## UNDERSTANDING THE OUTPUT LINE BY LINE

```text
Raw conv feature map : (2, 512, 14, 14)
  2 = batch size (two images were passed)
  512 = number of feature channels from VGG19
  14×14 = spatial grid size
  This confirms the trailing MaxPool layers were removed correctly.
  If they were still there, this would show (2, 512, 7, 7) — wrong.
```

```text
Adaptive-pooled : (2, 512, 14, 14)
  The AdaptiveAvgPool2d layer ensures the output is always exactly 14×14
  regardless of input image size. Shape unchanged — correct.
```

```text
Annotation vectors : (2, 196, 512)
  196 = 14 × 14 (the spatial grid flattened into a list of locations)
  512 = dimension of each annotation vector
  These are exactly the vectors a_1...a_L from §3.1.1 of the paper,
  where L=196 and D=512.
```

```text
Pretrained weights OK : True
  VGG19 ImageNet weights loaded successfully. The encoder starts from a
  strong visual understanding built from millions of ImageNet images.
  If this shows False, the encoder is randomly initialised and results
  will be poor. Re-run the cell or check internet connectivity.
```

## THE CLEANUP AT THE END

del _enc_check, raw_feat, pooled, annot
torch.cuda.empty_cache()

Deletes the temporary encoder and clears GPU memory immediately.
GPU memory on Colab T4 is limited. Keeping a second copy of the encoder
in memory while the real model trains would cause out-of-memory errors.

## PAPER REFERENCE

§3.1.1: "The extractor produces L vectors, each of which is a D-dimensional
representation corresponding to a part of the image." L=196, D=512.

§4.3: "we use the 14×14×512 feature map of the fourth convolutional layer
before max pooling."

Both claims are verified directly by this cell.

## ACTUAL OUTPUT

```text
Raw conv feature map   : (2, 512, 14, 14)  (expected: [B, 512, 14, 14])
Adaptive-pooled        : (2, 512, 14, 14)
Annotation vectors     : (2, 196, 512)     (expected: [B, 196, 512])
Pretrained weights OK  : True
```

✔ Encoder shape matches Show, Attend and Tell.
