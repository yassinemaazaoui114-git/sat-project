# CELL 06 — Vocabulary & Tokenisation

> **Note.** These study notes were first written for an earlier Flickr8k / Google Colab
> version of our notebook. The code they explain is the same as in
> [`show_attend_tell_flickr30k.ipynb`](../../notebooks/show_attend_tell_flickr30k.ipynb),
> but dataset sizes, paths, the optimiser and the example outputs can differ from our final
> Flickr30k run. See the [README](../../README.md) for the actual Flickr30k results.

## WHAT IS THE PROBLEM THIS CELL SOLVES?

```text
The LSTM model cannot process words. It only processes numbers.
"a dog is running" means nothing to a neural network.
[4, 42, 9, 87] is what the model actually reads.
This cell builds the lookup table that converts between words and numbers
in both directions.
```

## WHAT DOES THIS CELL ACTUALLY DO?

Creates a Vocabulary object by scanning every caption in the training set,
counting how often each word appears, and assigning a unique integer index
to every word that appears at least 5 times. The result is a two-way
mapping: word → number (stoi) and number → word (itos).

## WHY TRAINING CAPTIONS ONLY?

If you built the vocabulary from all captions including validation and test,
words that only appear in test would get an index. At inference time the model
would have seen those words during vocabulary construction, giving it an
unfair advantage. Building from training data only mirrors real deployment —
at inference time the model only knows words it was trained on.

## THE TWO KEY HYPER-PARAMETERS

```text
MIN_FREQ = 5    A word must appear at least 5 times in training captions to
                enter the vocabulary. Words appearing fewer times are almost
                always typos, rare proper nouns, or noise. Filtering them out
                keeps the vocabulary small and clean. Rare words become <unk>
                at inference time anyway.
```

```text
MAX_LEN = 30    Captions longer than 30 tokens are truncated. This keeps
                LSTM sequence lengths manageable and matches the distribution
                of Flickr8k captions (most are under 20 tokens).
```

## THE FOUR METHODS OF THE VOCABULARY CLASS

```text
build(sentences)       Counts all words, assigns index to those >= MIN_FREQ.
                       Called once on training captions. Indices 0-3 are reserved
                       for the four special tokens. Real words start at index 4.
```

```text
numericalize(text)     Converts a caption string to a list of integers.
                       Prepends <start> (index 1), appends <end> (index 2).
                       Unknown words become <unk> (index 3).
                       Truncates at MAX_LEN.
```

```text
decode(ids)            Converts a list of integers back to a string.
                       Stops at <end> token. Skips <pad> and <start>.
                       Used during inference to read the model output.
```

```text
tokenize(text)         Simple whitespace split. "a dog" → ["a", "dog"].
                       Same basic tokenisation used in the paper (Section 5.1).
```

## UNDERSTANDING THE ACTUAL OUTPUT

Vocabulary size: 2,660. The paper uses 10,000 because it combined multiple
larger datasets. Flickr8k is smaller so fewer words pass the MIN_FREQ=5 threshold.

The word "entry" in the example caption became `<unk>` (index 3) in the encoded
output. This means "entry" appeared fewer than 5 times in the 32,360 training
captions and was filtered out. This is completely normal. The decoded output
correctly shows `<unk>` in place of "entry", confirming the round-trip works.

Why does "a" always get index 4?
"a" is the most common word in English captions. It definitely passes MIN_FREQ=5
and gets assigned the first available real word index (4), right after the
four reserved special tokens at indices 0, 1, 2, 3.

## PAPER REFERENCE

Section 5.1: "For all our experiments, we used a fixed vocabulary size of 10,000."
Our vocabulary is smaller (2,660) because we build from Flickr8k training captions
only rather than combining all three datasets.

## EXPECTED OUTPUT

```text
Vocabulary size : 2660
```

```text
Original  : a child in a pink dress is climbing up a set of stairs in an entry way .
Encoded   : [1, 4, 5, 6, 4, 7, 8, 9, 10, 11, 4, 12, 13, 14, 6, 15, 3, 16, 17, 2]
Decoded   : a child in a pink dress is climbing up a set of stairs in an <unk> way .
```
