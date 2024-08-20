In this repo, the adaptation of the DivideMix code for handling text data is outlined. The two Python files are based 
on the files train_cifar.py and dataloader_cifar.py from the repo https://github.com/LiJunnan1992/DivideMix.

**Model Architecture:**

The create_model() function implements BERT instead of a CNN.

**Feature Extraction:**

In the getitem method a tokenizer is applied to each item. The word embedding happens in the BERT model (or any other
transformer).

**Batch Processing:**

The collate_batch function handles variable-length text inputs in a batch. It determines the maximum length in the 
current batch and then pads all sequences in the batch to this maximum length.

**Data Augmentation:**

Data augmentation is defined in the augment_text() function and uses WordNet.

**Linear Interpolation (MixUp):**

Adaption happens in the text_mixup() function which performs MixUp on the embedding level.