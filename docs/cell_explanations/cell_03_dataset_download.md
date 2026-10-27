# CELL 03 — Dataset Download & Path Detection

> **Note.** These study notes were first written for an earlier Flickr8k / Google Colab
> version of our notebook. The code they explain is the same as in
> [`show_attend_tell_flickr30k.ipynb`](../../notebooks/show_attend_tell_flickr30k.ipynb),
> but dataset sizes, paths, the optimiser and the example outputs can differ from our final
> Flickr30k run. See the [README](../../README.md) for the actual Flickr30k results.

## WHAT IS THE PROBLEM THIS CELL SOLVES?

The model needs 8,091 images and 40,455 captions to learn from. None of this data
is inside Colab — it lives on Kaggle. This cell downloads it, unpacks it, and
finds where everything is on disk so the rest of the notebook knows where to look.

## WHAT DOES THIS CELL ACTUALLY DO?

Three things in sequence:

```text
1. Sets up Kaggle API credentials so Colab has permission to download.
   Kaggle requires authentication — your kaggle.json file contains your
   username and API key.
```

```text
2. Downloads and unzips Flickr8k from Kaggle into /content/flickr8k.
   Total size is about 1GB. Takes 2-3 minutes on Colab.
```

```text
3. Walks the directory tree to find the Images/ folder and captions.txt
   automatically. Sets three global variables that every later cell uses.
```

## WHY AUTO-DETECT PATHS INSTEAD OF HARDCODING?

If you hardcode the path as /content/flickr8k/Images and the zip file unpacks
into a slightly different folder structure, every cell that uses the path breaks.
The find_dataset_root() function searches for a folder that contains both an
Images/ directory and a captions.txt file, so it works regardless of how
the dataset is structured on disk.

## THE THREE KEY VARIABLES SET BY THIS CELL

```text
DATASET_ROOT    /content/flickr8k
                The top-level folder everything lives in.
```

```text
IMAGES_DIR      /content/flickr8k/Images
                The folder containing all 8,091 .jpg image files.
                This is what the Dataset class uses to open images during training.
```

```text
CAPTIONS_FILE   /content/flickr8k/captions.txt
                The CSV file containing image filenames and their captions.
                This is what the next cell reads.
```

## PAPER REFERENCE

Section 5.1 of the paper: "We report results on the widely-used Flickr8k dataset...
Each image in the Flickr8k/30k dataset have 5 reference captions."
8,091 images × 5 captions = 40,455 total pairs.

## EXPECTED OUTPUT

```text
Download complete.
DATASET_ROOT  : /content/flickr8k
IMAGES_DIR    : /content/flickr8k/Images   exists = True
CAPTIONS_FILE : /content/flickr8k/captions.txt exists = True
```

Number of jpg files : 8091

The number 8,091 is a direct check against the paper.
Any other number means the download was incomplete — re-run the cell.
