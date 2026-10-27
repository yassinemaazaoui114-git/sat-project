# CELL 07 — Dataset, DataLoader & Length-Based Bucketing

> **Note.** These study notes were first written for an earlier Flickr8k / Google Colab
> version of our notebook. The code they explain is the same as in
> [`show_attend_tell_flickr30k.ipynb`](../../notebooks/show_attend_tell_flickr30k.ipynb),
> but dataset sizes, paths, the optimiser and the example outputs can differ from our final
> Flickr30k run. See the [README](../../README.md) for the actual Flickr30k results.

## WHAT IS THE PROBLEM THIS CELL SOLVES?

At this point we have a vocabulary and DataFrames with filenames and captions.
But the model cannot train from a DataFrame. PyTorch needs data delivered in
a very specific way — in batches of tensors, preprocessed, on the GPU.
This cell builds the bridge between the raw data and the model.

## WHAT DOES THIS CELL ACTUALLY DO?

Three things:

1. Defines image preprocessing transforms (train_tfm and eval_tfm)
2. Wraps the DataFrames into PyTorch Dataset objects (FlickrDataset)
3. Builds DataLoaders with a length-based bucketing sampler

## WHAT FLOWS IN AND OUT

```text
Raw JPEG on disk
      ↓  FlickrDataset.__getitem__()
Tensor [3, 224, 224]  +  caption [1, 5, 23, 8, 2]  +  length 5
      ↓  DataLoader + collate_fn
Batch of 64 images [64, 3, 224, 224]  +  captions [64, 12]
      ↓
Ready to feed into VGG19 encoder
```

## PART 1 — IMAGE PREPROCESSING

VGG19 was trained on ImageNet with images of a specific size and normalisation.
Feeding it raw images produces garbage features. The transforms make every
image look exactly like what VGG19 was trained on.

```text
Resize(256)        Makes the shortest side 256 pixels, keeps aspect ratio.
RandomCrop(224)    For training: crops a random 224×224 patch each time.
                   This means the model sees slightly different versions of
                   the same image across epochs.
CenterCrop(224)    For eval: always crops the centre 224×224.
                   Deterministic — same crop every time for reproducible results.
Normalize(...)     Subtracts ImageNet mean, divides by ImageNet std.
                   Centres pixel values around zero, which helps training stability.
```

## CHANGES FROM THE FLICKR30K VERSION

```text
Batch size          Changed from 32 to 64.
                    Paper Section 4.3 explicitly states "mini-batch of size 64."
```

```text
RandomHorizontalFlip  Removed.
                    The paper does not mention any data augmentation.
                    The Flickr30k version added this independently.
                    Removed here to stay faithful to the paper.
```

## PART 2 — FLICKRDATASET CLASS

PyTorch needs a Dataset object. Think of it as a smart list.
When the training loop asks for item number 5, the Dataset:

- Opens image number 5 from disk
- Applies the preprocessing transform
- Converts its caption to a list of integers using the vocabulary
- Returns the image tensor, the caption tensor, and the caption length

The Dataset handles one sample at a time.
The DataLoader handles batching multiple samples together.

## PART 3 — LENGTH-BASED BUCKETING (PAPER-ALIGNED)

This is the most important part of this cell.

Normal DataLoader problem:
A batch might contain captions of lengths 5, 12, 19, 27.
All must be padded to length 27. The model processes 27 timesteps
but 22 of them are just padding for some captions. Wasted computation.

Paper solution (Section 4.3):
"In preprocessing we build a dictionary mapping the length of a sentence
to the corresponding subset of captions. Then, during training we randomly
sample a length and retrieve a mini-batch of size 64 of that length."

```text
LengthBucketSampler does exactly this:
- Groups all 32,360 training captions by their token length
- When building a batch, picks one length and fills the batch with
  64 captions all of that exact length
- Zero padding is needed because every caption is already the same length
- Paper confirms: "We found that this greatly improved convergence speed
  with no noticeable diminishment in performance."
```

## PAPER REFERENCES

```text
Section 4.3:  Batch size 64, length bucketing strategy, convergence speed.
Section 5.4:  "resized so shortest side is 256... centre-cropped 224×224"
```

## EXPECTED OUTPUT

```text
Using length-based bucketing for training (paper §4.3).
Train batches : 493
Val batches   : 64
Test batches  : 64
Image batch shape   : torch.Size([64, 3, 224, 224])
Caption batch shape : torch.Size([64, 12])
Lengths in batch    : [12]
```

## UNDERSTANDING THE OUTPUT

```text
[64, 3, 224, 224]   64 images, 3 RGB channels, 224×224 pixels.
                    Exactly what VGG19 expects as input. ✓
```

```text
[64, 12]            64 captions, each exactly 12 tokens long.
                    Zero padding in this batch. Bucketing is working. ✓
```

```text
Lengths: [12]       One single value in the set. Every caption in this
                    batch is the same length. Perfect bucketing. ✓
```

```text
493 train batches   32,360 captions ÷ 64 per batch = ~505 possible batches.
                    Minus batches dropped by drop_last when a bucket does not
                    have enough captions to fill a complete batch of 64.
```
