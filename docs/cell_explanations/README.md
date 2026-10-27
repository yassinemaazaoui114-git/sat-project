# Cell-by-cell explanations

These are the study notes we wrote while building the notebook. Each note explains one cell (or a
group of cells): the problem it solves, what the code does, the matching section of the paper, and
the expected output.

> **Note.** We first wrote these notes for an earlier **Flickr8k / Google Colab** version of the
> notebook. The model code is the same as in our final
> [`show_attend_tell_flickr30k.ipynb`](../../notebooks/show_attend_tell_flickr30k.ipynb), but some
> details differ from the final Flickr30k run on Kaggle: dataset sizes (8,091 vs 31,783 images),
> vocabulary size (2,660 vs 6,928), paths (`/content` vs `/kaggle`), the optimiser (RMSProp, as the
> paper recommends for Flickr8k, vs Adam for Flickr30k) and the example outputs. The
> [main README](../../README.md) reports the actual Flickr30k results.

| Note | Topic | Notebook section |
|---|---|---|
| [Cell 01](cell_01_imports_setup.md) | Imports, random seed, GPU check | 1 |
| [Cell 02](cell_02_runtime_controls.md) | Runtime flags (soft / hard attention, debug mode, patience) | 1.5 |
| [Cell 03](cell_03_dataset_download.md) | Dataset download and path detection | 2 |
| [Cell 04](cell_04_load_captions.md) | Loading and cleaning captions | 3 |
| [Cell 04b](cell_04b_visualise_sample.md) | Visualising one image with its 5 captions | 3 |
| [Cell 05](cell_05_data_split.md) | Train / validation / test split by image | 4 |
| [Cell 06](cell_06_vocabulary.md) | Vocabulary and tokenisation | 5 |
| [Cell 07](cell_07_dataloader.md) | Dataset, DataLoader and length-based bucketing | 6 |
| [Cell 08](cell_08_model_architecture.md) | EncoderCNN, attention and the LSTM decoder with the β gate | 7 |
| [Cell 09](cell_09_encoder_sanity_check.md) | Checking the 14×14×512 VGG19 feature map | 7 |
| [Cell 10](cell_10_training_setup.md) | Hyperparameters, loss and optimiser | 8 |
| [Cells 11–13](cell_11_12_13_training_and_inference.md) | BLEU helper, training loop with early stopping, greedy caption generation | 8–9 |
