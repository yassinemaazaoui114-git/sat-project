# CELL 08 — Model Architecture: EncoderCNN, Attention, DecoderWithAttention

> **Note.** These study notes were first written for an earlier Flickr8k / Google Colab
> version of our notebook. The code they explain is the same as in
> [`show_attend_tell_flickr30k.ipynb`](../../notebooks/show_attend_tell_flickr30k.ipynb),
> but dataset sizes, paths, the optimiser and the example outputs can differ from our final
> Flickr30k run. See the [README](../../README.md) for the actual Flickr30k results.

## WHAT IS THE PROBLEM THIS CELL SOLVES?

Everything before this cell was preparation — loading data, building vocabulary,
creating DataLoaders. This cell defines the actual model. Three components that
connect together to form the complete SAT architecture.

The full pipeline looks like this:

```text
Image [224×224]
      ↓  EncoderCNN
196 annotation vectors [196 × 512]
      ↓  Attention  (runs at every word step)
context vector [512]  +  alpha weights [196]
      ↓  DecoderWithAttention
next word probability [vocab_size]
```

This repeats word by word until the model generates `<end>`.

## PART A — ENCODERCNN

**WHAT IT DOES:**

Loads VGG19 pretrained on ImageNet, removes the fully connected layers at the
top, and also removes the trailing MaxPool2d layers. The result is a feature
extractor that turns a 224×224 image into 196 annotation vectors of 512
dimensions each.

**WHY REMOVE THE FULLY CONNECTED LAYERS?**

The fully connected layers at the top of VGG19 compress the entire image into
one single 4096-dimensional vector. All spatial information is lost. You would
know what objects are in the image but not where they are. Attention needs to
know WHERE things are — it cannot work without spatial information.

**WHY REMOVE THE TRAILING MAXPOOL LAYERS?**

The paper states in §4.3: "we use the 14×14×512 feature map of the fourth
convolutional layer before max pooling." Each max-pooling layer halves the
spatial dimensions. Removing the last max-pool keeps the feature map at 14×14
instead of 7×7, giving the attention module 196 locations instead of 49.
More locations = finer-grained attention.

**THE FORWARD PASS STEP BY STEP:**

```text
Input image          [B, 3, 224, 224]
↓ VGG19 conv layers  [B, 512, 14, 14]   14×14 spatial map, 512 channels
↓ AdaptiveAvgPool    [B, 512, 14, 14]   ensures exactly 14×14 regardless
↓ permute(0,2,3,1)   [B, 14, 14, 512]   move channels to last dimension
↓ view(B, -1, 512)   [B, 196, 512]      flatten 14×14 to 196 locations
```

The 196 = 14×14. Each of the 196 vectors corresponds to a specific spatial
location in the original image. This spatial awareness is what attention exploits.

Paper reference: §3.1.1, §4.3, §5.4

## PART B — ATTENTION

**WHAT IT DOES:**

At each word generation step, scores all 196 image regions against the current
decoder hidden state, converts scores to weights using softmax, and returns a
context vector — a weighted summary of the image focused on the relevant region.

**THE MATH IN PLAIN ENGLISH:**

```text
Step 1 — Score every location
  enc_att(encoder_out)         each of 196 locations gets a 256-dim score
  dec_att(decoder_hidden)      current hidden state gets a 256-dim score
  tanh(location + hidden)      combine them — measures how relevant each
  full_att(combined)           compress to one number per location → [B, 196]
```

```text
Step 2 — Convert scores to weights
  softmax(scores, dim=1)       alpha [B, 196], all 196 weights sum to 1
```

```text
Step 3 — Compute context vector
  sum(encoder_out × alpha)     context [B, 512], weighted average of regions
```

```text
This is exactly the paper equations from §3.1:
  e_ti = f_att(a_i, h_{t-1})           scoring function
  alpha_ti = exp(e_ti) / sum(exp(e_tk)) softmax
  z_hat_t = sum(alpha_i × a_i)         context vector (Equation 1 & 8)
```

**THE KEY INSIGHT — WHY DOES ALPHA DEPEND ON THE HIDDEN STATE?**

The hidden state h_{t-1} contains everything the LSTM has generated so far.
After generating "a bird", the hidden state carries that information. This
steers the attention toward the sky or water region for the next word.
"Where to look next depends on what has already been said." — §3.1

Paper reference: §3.1, §4.2, Equations 1 and 8

## PART C — DECODERWITHATTENTION

**WHAT IT DOES:**

Runs a loop — one iteration per word. At each step it attends to the relevant
image region, gates how much visual information to use, updates the LSTM state,
and predicts the next word.

**THE LOOP AT EACH TIMESTEP t:**

```text
1. embed[t]                      get embedding of word at position t
2. context, alpha                run attention with current hidden state h
3. beta = sigmoid(f_beta(h))     compute gating scalar (see beta gate below)
4. context = beta × context      scale context by gate
5. [embed; context] → LSTMCell   concatenate and run one LSTM step
6. preds = fc(dropout(h))        project hidden state to vocabulary scores
```

**THE BETA GATE (§4.2.1):**

beta is a learned scalar between 0 and 1, computed from the hidden state.
It lets the decoder decide at each step how much to rely on the image vs the
language model.

- When generating "a" or "the" — grammar matters, image less important. Beta small.
- When generating "dog" or "running" — image matters. Beta large.

The paper calls this the "gating scalar β" and shows it improves caption quality.

**INITIAL STATES (§3.1):**

```text
The LSTM is NOT initialised to zeros. Instead:
  mean_enc = mean(all 196 annotation vectors)
  h_0 = tanh(Linear(mean_enc))
  c_0 = tanh(Linear(mean_enc))
This gives the LSTM a summary of the whole image before generating word 1.
```

**THE SORTED BATCH TRICK:**

Captions in a batch are sorted longest to shortest. The variable batch_t at
each timestep counts how many captions are still active. Only active captions
are processed — padding positions are skipped entirely.

**WHAT THE MODEL RETURNS:**

```text
predictions    [B, max_len, vocab_size]   word scores at each timestep
alphas         [B, max_len, 196]          attention weights at each timestep
betas          [B, max_len]               gate values at each timestep
decode_lengths [B]                        actual lengths for loss masking
```

Paper reference: §3.1, §3.1.2, §4.2, §4.2.1, Equations 1, 2, 8, 9

## FULL PAPER ALIGNMENT TABLE

```text
Code element                  Paper reference
Drop trailing MaxPool         §4.3: "feature map before max pooling"
[B, 196, 512] output          §3.1.1: "L vectors, each D-dimensional" (L=196, D=512)
Attention scoring with tanh   §3.1: f_att as multilayer perceptron
alpha = softmax(e)            §3.1: alpha_ti formula
context = sum(alpha × a)      §4.2: Equation 8 (soft attention)
beta gate                     §4.2.1: "gating scalar β = σ(f_β(h_{t-1}))"
h_0, c_0 from mean(enc_out)   §3.1: "predicted by average of annotation vectors"
dropout = 0.5                 §4.3: "In addition to dropout"
embed_dim = 256               §3.1.2: embedding dimension m
decoder_dim = 512             §3.1.2: LSTM dimensionality n
attention_dim = 256           §3.1: attention MLP hidden size
```

## EXPECTED OUTPUT

```text
EncoderCNN, Attention, DecoderWithAttention defined.
```

Just a confirmation line. The model is not trained here — this cell only
defines the architecture. Training happens in the training loop cell.
