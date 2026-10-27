# CELL 02 — Runtime Controls

> **Note.** These study notes were first written for an earlier Flickr8k / Google Colab
> version of our notebook. The code they explain is the same as in
> [`show_attend_tell_flickr30k.ipynb`](../../notebooks/show_attend_tell_flickr30k.ipynb),
> but dataset sizes, paths, the optimiser and the example outputs can differ from our final
> Flickr30k run. See the [README](../../README.md) for the actual Flickr30k results.

## WHAT IS THE PROBLEM THIS CELL SOLVES?

A deep learning notebook has many moving parts — soft attention training, hard
attention training, debug mode for quick testing, early stopping, REINFORCE
hyper-parameters. If these settings are scattered across 20 different cells, changing
one thing requires hunting through the whole notebook. This cell puts every
on/off switch and every numerical setting in one single place.

## WHAT DOES THIS CELL ACTUALLY DO?

Defines all the flags and hyper-parameters that control the entire experiment.
Nothing is trained or computed here — this cell just sets variables.
Every other cell reads from these variables, so changing one value here
affects the whole notebook automatically.

## WHAT EACH FLAG DOES

```text
RUN_SOFT_ATTENTION       True/False. Whether to train the soft (deterministic)
                         attention model described in Section 4.2 of the paper.
```

```text
RUN_HARD_ATTENTION       True/False. Whether to train the hard (stochastic)
                         attention model described in Section 4.1 of the paper.
```

```text
DEBUG_MODE               If True, uses only 500 images per split so the whole
                         pipeline runs in minutes instead of hours. Useful for
                         checking that the code works before a full training run.
```

```text
SOFT/HARD_NUM_EPOCHS     Maximum number of times the model sees the full training
                         set. Early stopping may halt training before this.
```

```text
PATIENCE = 3             If the validation BLEU-4 score does not improve for 3
                         epochs in a row, training stops early. The paper uses
                         BLEU on the validation set for model selection (§4.3).
```

```text
REINFORCE_WEIGHT (r)     Scales the reward signal in the REINFORCE gradient
                         estimate for hard attention. Called r in Equation 7
                         of the paper.
```

```text
ENTROPY_WEIGHT (e)       Adds an entropy bonus to the hard attention gradient
                         to reduce variance. Called e in Equation 7 of the paper.
```

```text
BASELINE_MOMENTUM        Controls how fast the moving average reward baseline
                         b_k updates. Set to 0.9 in the paper (Section 4.1).
                         Formula: b_k = 0.9 × b_(k-1) + 0.1 × log p(y|s,a)
```

```text
SOFT_MIX_PROB = 0.5      Paper trick from Section 4.1: during hard attention
                         training, with probability 0.5 replace the randomly
                         sampled attention location with its expected value α.
                         This makes training more stable.
```

```text
USE_PAPER_OUTPUT_LAYER   Use the deep output layer from Equation 2 of the paper
                         instead of a simple linear projection.
```

## WHY BOTH SOFT AND HARD ATTENTION?

The paper presents both as its core contribution. Table 1 reports results for
both on Flickr8k. Hard attention actually scores slightly higher (BLEU-4: 21.3
vs 19.5 for soft). Implementing both allows direct comparison against the paper.

## EXPECTED OUTPUT

```text
RUN_SOFT_ATTENTION          = True
RUN_HARD_ATTENTION          = True
DEBUG_MODE                  = False
SOFT_NUM_EPOCHS             = 20
HARD_NUM_EPOCHS             = 20
PATIENCE                    = 3
REINFORCE_WEIGHT            = 1.0
ENTROPY_WEIGHT              = 0.01
BASELINE_MOMENTUM           = 0.9
HARD_ATTENTION_USE_SOFT_MIX = True
```
