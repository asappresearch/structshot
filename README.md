# structshot

Code and data for paper ["Simple and Effective Few-Shot Named Entity Recognition with Structured Nearest Neighbor Learning"](https://arxiv.org/abs/2010.02405), Yi Yang and Arzoo Katiyar, in EMNLP 2020.

## Data

Due to license reason, we are only able to release the full CoNLL 2003 and WNUT 2017 dataset. We also release the support sets that we sampled from the CoNLL/WNUT/I2B2 dev sets to enable the reproducing of our evaluation results.

**CoNLL 2003**

The CoNLL 2003 NER train/dev/test datasets are `data/train.txt`, `data/dev.txt`, and `data/test.txt` respectively. The labels are available in `data/labels.txt`.

**WNUT 2017**

The WNUT 2017 NER dev/test datasets are `data/dev-wnut.txt` and `data/test-wnut.txt` respectively. The labels are available in `data/labels-wnut.txt`.

**Support sets for CoNLL 2003, WNUT 2017, and I2B2 2014**

The one-shot and five-shot support sets used in the paper are available in `data/support-*` folders.

## Usage

Due to data license limitation, we will show how to do five-shot transfer learning from the CoNLL 2003 dataset to the WNUT 2017 dataset, instead of transfering from the OntoNotes 5 dataset, as presented in our paper.

The first step is to install the package and `cd` into the `structshot` directory:
```
pip install -e .
cd structshot
```

### Pretrain BERT-NER model

The marjority of the code is copied from the HuggingFace [transformers](https://github.com/huggingface/transformers/tree/master/examples/token-classification) repo, which is used to pretrain a BERT-NER model:
```
# Pretrain a conventional BERT-NER model on CoNLL 2003 
bash run_pl.sh
```
In our paper, we actually merged B- and I- tags together for pretraining as well.


### Few-shot NER with NNShot

Given the pretrained model located at `output-model/checkpointepoch=2.ckpt`, we now can perform five-shot NER transfer on the WNUT test set:
```
# Five-shot NER with NNShot
bash run_pred.sh output-model/checkpointepoch=2.ckpt NNShot
```
We use the IO tagging scheme rather than the BIO tagging scheme due to its simplicity and better performance. I obtained **22.8** F1 score. 

### Few-shot NER with StructShot
Given the same pretrained model, simply run:
```
# Five-shot NER with StructShot
bash run_pred.sh output-model/checkpointepoch=2.ckpt StructShot
```
I obtained **29.5** F1 score. You can tune the parameter tau in the `run_pred.sh` script based on dev set performance.

## Notes

There are a few differences between this implementation and the one reported in the paper due to data license reason etc.:
* This implementation pretrains the BERT-NER model with the BIO tagging scheme, while in our paper we uses the IO tagging scheme.
* This implementation performs five-shot transfer learning from CoNLL 2003 to WNUT 2017, while in our paper we perform five-shot transfer learning from OntoNotes 5 to CoNLL'03/WNUT'17/I2B2'14.

If you can access OntoNotes 5 and I2B2'14, reproducing the results of the paper should be trivial.