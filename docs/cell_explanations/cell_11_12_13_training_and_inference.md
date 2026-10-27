# CELL 11 — BLEU Evaluation Helper

> **Note.** These study notes were first written for an earlier Flickr8k / Google Colab
> version of our notebook. The code they explain is the same as in
> [`show_attend_tell_flickr30k.ipynb`](../../notebooks/show_attend_tell_flickr30k.ipynb),
> but dataset sizes, paths, the optimiser and the example outputs can differ from our final
> Flickr30k run. See the [README](../../README.md) for the actual Flickr30k results.

## WHAT IS THE PROBLEM THIS CELL SOLVES?

During training we need to check after every epoch whether the model is actually
generating better captions. Loss going down is a good sign but it does not
directly measure caption quality. The paper observed this in §4.3:
"We observed a breakdown in correlation between the validation set log-likelihood
and BLEU in the later stages of training."

This cell provides compute_val_bleu() — the function that generates captions for
200 validation images and returns BLEU-1 through BLEU-4 scores. These scores drive
the early stopping decision in the training loop.

## WHAT DOES compute_val_bleu() DO?

1. Randomly samples 200 images from the validation set (VAL_BLEU_NSAMPLES)
2. For each image: generates a greedy caption using greedy_caption()
3. Compares each generated caption against all 5 human reference captions
4. Returns BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores

## WHY SAMPLE 200 INSTEAD OF ALL 810 VALIDATION IMAGES?

Running greedy_caption() on 810 images after every epoch is slow.
200 images gives a reliable estimate of BLEU in much less time.
The same 200 are sampled consistently (fixed SEED) so scores are comparable
across epochs.

## THE REFERENCE DICT

val_refs and test_refs map each image filename to its list of 5 reference captions.
Built once here, reused many times. Format:
{"image.jpg": ["caption 1", "caption 2", "caption 3", "caption 4", "caption 5"]}

SmoothingFunction().method1
Short captions sometimes have zero 4-gram matches, which would make BLEU-4 = 0.
Method1 adds a small smoothing value to avoid this mathematical breakdown.
Standard choice for sentence-level BLEU evaluation.

## PAPER REFERENCE

§4.3: "Since BLEU is the most commonly reported metric, we used BLEU on our
validation set for model selection."

## EXPECTED OUTPUT

```text
Val images for BLEU monitoring : 809
```

(809 not 810 because one image had no surviving captions after cleaning.
Off by one is normal and does not affect anything.)

---

# CELL 12 — Training Loop

## WHAT IS THE PROBLEM THIS CELL SOLVES?

This is the cell where actual learning happens. All previous cells were setup.
This cell runs the model through the training data repeatedly, updating weights
after each batch, until either the model stops improving or 20 epochs complete.

## WHAT DOES run_one_epoch() DO?

Runs the model through the entire dataset once. For each batch:

Training mode:

1. Zero the gradients
2. Encode images through frozen VGG19 → [B, 196, 512]
3. Run decoder with teacher forcing → predictions, alphas, betas
4. Compute total loss (see loss function below)
5. Backpropagate gradients
6. Clip gradients to prevent explosions
7. Update weights with RMSProp

Validation mode:
Same steps 1-4 but no backpropagation. Just measure loss.

## TEACHER FORCING

During training the decoder receives the ground truth previous word at each
step, not its own prediction. Example: to predict word 3, the model is given
the true word 2, not whatever it predicted for word 2.
This makes training stable and fast. During inference (greedy_caption) the
model uses its own predictions — this is the train/test discrepancy that
motivates beam search for final evaluation.

## THE LOSS FUNCTION — PAPER EQUATION 9

Total loss = cross_entropy_loss + LAMBDA_ATT × attention_regularisation

Cross-entropy loss:
Penalises wrong word predictions. At each timestep the model outputs scores
for all 2,660 vocabulary words. Cross-entropy is high when the correct word
gets low probability. Loss is only computed on non-padding positions
(ignore_index = `<pad>`).

Attention regularisation (doubly-stochastic):
attention_reg = mean( (1 - sum_over_time(alpha_i))^2 )

For each image location i, the alphas summed over all timesteps should equal 1.
If a location gets ignored completely (sum = 0) or attended to too much (sum > 1),
this penalty increases. It forces the model to look at every part of the image
at least once across the whole caption. Paper §4.2.1 fixes lambda=1 (τ=1).

## GRADIENT CLIPPING

nn.utils.clip_grad_norm_(decoder.parameters(), GRAD_CLIP=5.0)
If the gradient vector has norm > 5.0, it is scaled down to 5.0.
LSTMs are prone to exploding gradients during backpropagation through time.
Without this, one bad batch can corrupt all the weights. Only the decoder
is clipped — the encoder is frozen and has no gradients.

## EARLY STOPPING LOGIC — PAPER ALIGNED

After every epoch:

- If BLEU-4 improved → save checkpoint, reset patience counter to 0
- If BLEU-4 did not improve → increment patience counter
- If patience counter reaches 3 → stop training

The saved checkpoint is always the best-performing epoch, not the last.

## WHAT GETS SAVED

/content/best_model.pth — encoder + decoder weights from best BLEU-4 epoch
/content/training_history.csv — loss and BLEU metrics per epoch

## THREE PLOTS PRODUCED

```text
Plot 1: Loss curves — train CE, val CE, train attention reg, val attention reg
        Shows whether the model is learning and whether it is overfitting.
```

```text
Plot 2: Token accuracy — percentage of words predicted correctly (teacher-forced)
        Should rise during training. Train accuracy > val accuracy is normal.
```

```text
Plot 3: Validation BLEU — BLEU-1 through BLEU-4 per epoch
        The curve that drives early stopping. Should rise then plateau.
```

## PAPER REFERENCES

```text
§4.2.1: Doubly-stochastic regularisation, Equation 9
§4.3:   RMSProp optimizer, early stopping on BLEU, gradient clipping
```

## EXPECTED TRAINING TIME

Per epoch on Colab T4: 8-12 minutes
Full 20 epochs: 3-4 hours
Start before sleeping. Early stopping may halt before 20 epochs.

---

# CELL 13 — Caption Generation (Greedy Decoding)

## WHAT IS THE PROBLEM THIS CELL SOLVES?

The model has been trained. Now you want to see if it actually works —
can it look at an unseen test image and generate a meaningful caption?
This cell loads the best checkpoint and generates captions for 4 random
test images, displaying them visually.

## WHAT DOES THIS CELL ACTUALLY DO?

1. Loads best_model.pth — restores weights from the epoch with highest val BLEU-4
2. Samples 4 random images from the test set
3. For each image: runs greedy_caption(), displays the image and caption
4. Prints the generated caption, mean beta value, and first reference caption

## WHY LOAD THE CHECKPOINT INSTEAD OF USING THE CURRENT MODEL?

Training may have run for 15 epochs before early stopping. The best model
was at epoch 8. The current model state is epoch 15 which may be slightly
worse due to overfitting. The checkpoint restores the epoch 8 weights —
the actual best model.

## UNDERSTANDING THE BETA VALUES

Mean β ≈ 0.59–0.65 in the 30k output means the visual context is actively
used for about 60% of the words on average. This confirms §4.2.1 of the
paper: "the gating variable is larger when the decoder describes an object
in the image." Content words (dog, water, bench) pull beta high.
Function words (a, the, in) pull beta low.

## CAPTION QUALITY — WHAT TO EXPECT

The model handles dominant objects and simple scenes well.
Common failure modes (matching the Liu 2023 error analysis):

- Incorrect object (confuses similar objects)
- Missed object (ignores secondary subjects)
- Incorrect action (gets the verb wrong)
- Wrong count (says "a man" when there are two men)

These failures are visible in the 30k example output:
"a boy is sitting on a rock in the water" — the image shows two boys
in mud, not one boy on a rock. The model got the water/mud context right
but missed the second child and misidentified the surface.

This is exactly the type of error the Liu & Brailsford (2023) paper
(the second paper you were given) analyses and tries to correct.

## PAPER REFERENCE

§5.4 — Qualitative Analysis: the paper shows exactly this type of
visualisation in Figure 4, with generated captions and attention maps
overlaid on images.

## NO CHANGES NEEDED FOR FLICKR8K

BEST_PATH was already set to /content/best_model.pth in cell 10.
Everything else works identically.
