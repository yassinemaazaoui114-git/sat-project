# CELL 04 — Load & Clean Captions

> **Note.** These study notes were first written for an earlier Flickr8k / Google Colab
> version of our notebook. The code they explain is the same as in
> [`show_attend_tell_flickr30k.ipynb`](../../notebooks/show_attend_tell_flickr30k.ipynb),
> but dataset sizes, paths, the optimiser and the example outputs can differ from our final
> Flickr30k run. See the [README](../../README.md) for the actual Flickr30k results.

## WHAT IS THE PROBLEM THIS CELL SOLVES?

The captions.txt file is raw text. It contains inconsistencies — uppercase letters,
special characters, extra spaces. The model cannot learn from inconsistent text
because "Dog" and "dog" would be treated as two completely different words,
doubling the vocabulary size for no reason. This cell reads the file, standardises
all text, and defines the four special control tokens the model uses.

## WHAT DOES THIS CELL ACTUALLY DO?

Two things:

```text
1. Reads captions.txt into a table (pandas DataFrame) with two columns:
   image (filename) and caption (text). Handles different separators
   automatically — comma, pipe, or tab — so it works with any version
   of the Flickr8k captions file.
```

```text
2. Cleans every caption with three operations:
   - Lowercase everything  ("A Dog" → "a dog")
   - Remove unusual characters  (keep only letters, numbers, basic punctuation)
   - Collapse extra spaces  ("a  dog" → "a dog")
```

## THE FOUR SPECIAL TOKENS

These are not real words. They are control signals that the LSTM uses
to know when to start, when to stop, and what to do with unknowns.
They occupy the first four indices in the vocabulary — always.

```text
<pad>    index 0    Fills short captions to match the longest one in a batch.
                    The loss function is told to ignore pad positions completely.
                    Without this, all captions in a batch would need to be the
                    same length, which is very wasteful.
```

```text
<start>  index 1    The very first input fed to the LSTM at timestep t=0.
                    It tells the model: "begin generating a caption now."
                    The model learns that after seeing <start> it should
                    output the first real word of the caption.
```

```text
<end>    index 2    Generation stops when the model predicts this token.
                    During training it marks where the caption should end.
                    During inference the model stops generating as soon as
                    it predicts index 2.
```

```text
<unk>    index 3    Replaces any word that does not make it into the
                    vocabulary (words appearing fewer than 5 times).
                    The model learns to handle unknown words gracefully.
```

## PAPER REFERENCE

Section 5.1: "We applied only basic tokenization to MS COCO so that it is
consistent with the tokenization present in Flickr8k and Flickr30k."
The paper uses the same simple approach — lowercase, basic punctuation only.

## EXPECTED OUTPUT

```text
Number of captions     : 40455
Number of unique images: 8091
Average captions/image : 5.0
```

These three numbers are the verification check.
40,455 = 8,091 × 5 exactly. If not, something went wrong reading the file.
