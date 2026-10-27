# CELL 10 — Training Setup: Hyperparameters, Model, Loss, Optimizer

> **Note.** These study notes were first written for an earlier Flickr8k / Google Colab
> version of our notebook. The code they explain is the same as in
> [`show_attend_tell_flickr30k.ipynb`](../../notebooks/show_attend_tell_flickr30k.ipynb),
> but dataset sizes, paths, the optimiser and the example outputs can differ from our final
> Flickr30k run. See the [README](../../README.md) for the actual Flickr30k results.

## WHAT IS THE PROBLEM THIS CELL SOLVES?

The model architecture is defined. Now we need to:

1. Set all the numbers that control how training behaves
2. Create the actual model instances and move them to the GPU
3. Define what we are optimising (the loss function)
4. Define how we optimise it (the optimizer)

Nothing trains here — this cell just assembles everything and gets it ready.

## WHAT DOES THIS CELL ACTUALLY DO?

Six things in sequence:

1. Sets all hyperparameters in one place
2. Saves the vocabulary to disk as vocab.json
3. Creates encoder and decoder instances on the GPU
4. Defines the cross-entropy loss function
5. Creates the RMSProp optimizer
6. Prints trainable parameter counts to verify the setup

## HYPERPARAMETERS AND THEIR PAPER REFERENCES

```text
EMBED_DIM = 256        Word embedding dimension. Called m in §3.1.2.
                       Each vocabulary word is represented as a 256-dim vector.
```

```text
DECODER_DIM = 512      LSTM hidden state size. Called n in §3.1.2.
                       The memory of the LSTM at each timestep.
```

```text
ATTENTION_DIM = 256    Size of the attention MLP hidden layer.
                       The scoring function f_att from §3.1.
```

```text
ENCODER_DIM = 512      Dimension of each annotation vector from VGG19.
                       Called D in §3.1.1. Fixed by VGG19 architecture.
```

```text
DROPOUT = 0.5          Applied to the hidden state before the output projection.
                       Paper §4.3: "In addition to dropout."
```

```text
EPOCHS = 20            Maximum training epochs. Consistent with SOFT_NUM_EPOCHS
                       set in the runtime controls cell.
```

```text
LR = 4e-4              Learning rate for RMSProp.
```

```text
GRAD_CLIP = 5.0        Gradient clipping threshold. If the gradient norm exceeds
                       5.0 it is scaled down. Prevents exploding gradients in
                       the LSTM, which is a known instability.
```

```text
LAMBDA_ATT = 1.0       Weight for the doubly-stochastic regularisation term.
                       Paper §4.2.1: "we simply fixed τ to 1."
```

```text
PATIENCE = 3           Early stopping: stop if val BLEU-4 does not improve for
                       3 consecutive epochs.
```

## WHY RMSPROP AND NOT ADAM?

This is a critical paper-specific choice. The paper states in §4.3:
"For the Flickr8k dataset, we found that RMSProp worked best, while for
Flickr30k/MS COCO dataset we found the recently proposed Adam algorithm
to be quite effective."

The Flickr30k version of this notebook used Adam — correct for Flickr30k.
This notebook uses RMSProp — correct for Flickr8k.
This is a direct, paper-justified difference between the two notebooks.

## WHY IS THE ENCODER FROZEN?

fine_tune=False means all VGG19 parameters have requires_grad=False.
The optimizer only receives parameters where requires_grad=True, so
the encoder contributes zero trainable parameters.

This matches the paper exactly: §4.3 states the Oxford VGGnet was used
"pretrained on ImageNet without finetuning."

The encoder already knows how to see — it learned from millions of
ImageNet images. Fine-tuning on 6,472 Flickr8k images would likely
cause overfitting and degrade the general visual features.

## WHY IS THE VOCABULARY SAVED TO DISK?

The model is trained using integer indices from this specific vocabulary.
When you want to use the model later — for inference, for evaluation,
or after a Colab session restart — you need the exact same word-to-index
mapping. If vocab.json is lost, the saved model checkpoint becomes useless
because every word index would be different after rebuilding.

## THE LOSS FUNCTION

nn.CrossEntropyLoss(ignore_index=vocab.stoi[PAD_TOKEN])

Cross-entropy measures how wrong the model's word predictions are.
For each position in the caption, the model outputs a probability
distribution over all 2,660 vocabulary words. Cross-entropy penalises
the model for giving low probability to the correct word.

ignore_index=0 tells PyTorch to skip pad positions when computing the
loss. Padding is not a real word — the model should not be penalised
for its predictions at those positions.

## ACTUAL OUTPUT

```text
Saved vocab to /content/vocab.json
Trainable params (decoder): 5721957
Trainable params (encoder): 0
```

## UNDERSTANDING THE OUTPUT

```text
5,721,957 decoder parameters    About 5.7 million weights to learn.
                                Covers embedding, attention, LSTM, beta gate,
                                and output projection layers. All update
                                during backpropagation.
```

```text
0 encoder parameters            VGG19 is completely frozen. This is correct.
                                The paper used VGG19 without finetuning on
                                Flickr8k. Zero is the right number.
```

## CHANGES FROM FLICKR30K VERSION

```text
Optimizer     Adam → RMSProp    Paper §4.3 explicit recommendation for Flickr8k.
EPOCHS        8 → 20            Consistent with SOFT_NUM_EPOCHS runtime control.
OUTPUT_DIR    /kaggle/working   → /content (Colab path).
```
