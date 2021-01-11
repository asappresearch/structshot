"""
This script works on top of a pretrained BERT-NER model represented by a PyTorch Lightning checkpoint.
"""

import argparse
import logging
import os
import torch

import torch.nn.functional as F
import numpy as np

from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from lightning_base import add_generic_args, generic_train
from run_pl_ner import NERTransformer
from tasks import NER

from viterbi import ViterbiDecoder

logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ner_task = NER()

def get_dataloader(model, target_labels, data_dir, data_fname, batch_size):
    """
    Get dataloader for support or test set
    This method largely overlaps with run_pl_ner.py
    """
    examples = ner_task.read_examples_from_file(data_dir, data_fname)
    features = ner_task.convert_examples_to_features(
        examples,
        target_labels,
        args.max_seq_length,
        model.tokenizer,
        cls_token_at_end=bool(model.config.model_type in ["xlnet"]),
        cls_token=model.tokenizer.cls_token,
        cls_token_segment_id=2 if model.config.model_type in ["xlnet"] else 0,
        sep_token=model.tokenizer.sep_token,
        sep_token_extra=False,
        pad_on_left=bool(model.config.model_type in ["xlnet"]),
        pad_token=model.tokenizer.pad_token_id,
        pad_token_segment_id=model.tokenizer.pad_token_type_id,
        pad_token_label_id=model.pad_token_label_id,
    )

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    if features[0].token_type_ids is not None:
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    else:
        all_token_type_ids = torch.tensor([0 for f in features], dtype=torch.long)
        # HACK(we will not use this anymore soon)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    return DataLoader(
        TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_label_ids), batch_size=batch_size
    )


def get_token_encodings_and_labels(model, batch):
    """
    Get token encoding using pretrained BERT-NER model as well as groundtruth label
    """
    batch = tuple(t.to(device) for t in batch)
    label_batch = batch[3]
    with torch.no_grad():
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "output_hidden_states": True}
        if model.config.model_type != "distilbert":
            inputs["token_type_ids"] = (
                batch[2] if model.config.model_type in ["bert", "xlnet"] else None
            )  # XLM and RoBERTa don"t use token_type_ids
        outputs = model(**inputs)
        hidden_states = outputs[1][-1]  # last layer representations
    return hidden_states, label_batch


def get_support_encodings_and_labels(model, support_loader):
    """
    Get token encodings and labels for all tokens in the support set
    """
    support_encodings, support_labels = [], []
    for batch in tqdm(support_loader, desc="Support data representations"):
        encodings, labels = get_token_encodings_and_labels(model, batch)
        encodings = encodings.view(-1, encodings.shape[-1])
        labels = labels.flatten()
        # filter out PAD tokens
        idx = torch.where(labels != model.pad_token_label_id)[0]
        support_encodings.append(encodings[idx])
        support_labels.append(labels[idx])
    return torch.cat(support_encodings), torch.cat(support_labels)
    

def evaluate_few_shot(args, model):
    """
    Main method to evaluate NNShot and StructShot on a test set given a pretrained
    BERT-NER model and a support set.
    For simplicity and better performance, we will use IO encodings rather than BIO.
    """
    model.to(device)

    target_labels = ner_task.get_labels(args.target_labels)
    target_IO_labels = [label for label in target_labels if not label.startswith("B-")] 
    label_map = {i: label for i, label in enumerate(target_labels)}
    IO_label_map = {i: label for i, label in enumerate(target_IO_labels)}
    reversed_IO_map = {label: i for i, label in enumerate(target_IO_labels)}

    support_loader = get_dataloader(model, target_labels, args.data_dir, args.sup_fname, args.eval_batch_size)
    test_loader = get_dataloader(model, target_labels, args.data_dir, args.test_fname, args.eval_batch_size)
    support_encodings, support_labels = get_support_encodings_and_labels(model, support_loader)

    # merge B- and I- tags into I- tags
    support_IO_labels = []
    for label in support_labels.detach().cpu().numpy():
        label_str = label_map[label]
        if label_str.startswith("B-"):
            label_str = "I-" + label_str[2:]
        support_IO_labels.append(reversed_IO_map[label_str])
    support_IO_labels = torch.tensor(support_IO_labels).to(support_labels.device)
    
    preds = None
    emissions = None
    out_label_ids = None
    for batch in tqdm(test_loader, desc="Test data representations"):
        encodings, labels = get_token_encodings_and_labels(model, batch)
        nn_preds, nn_emissions = nn_decode(encodings, support_encodings, support_IO_labels)
        if preds is None:
            preds = nn_preds.detach().cpu().numpy()
            emissions = nn_emissions.detach().cpu().numpy()
            out_label_ids = labels.detach().cpu().numpy()
        else:
            preds = np.append(preds, nn_preds.detach().cpu().numpy(), axis=0)
            emissions = np.append(emissions, nn_emissions.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)

    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    emissions_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != model.pad_token_label_id:
                out_label_list[i].append(label_map[out_label_ids[i][j]])
                emissions_list[i].append(emissions[i][j])
                preds_list[i].append(IO_label_map[preds[i][j]])

    if args.algorithm == "StructShot":
        abstract_transitions = get_abstract_transitions(args.data_dir, args.train_fname)
        viterbi_decoder = ViterbiDecoder(len(target_IO_labels)+1, abstract_transitions, args.tau)
        preds_list = [[] for _ in range(out_label_ids.shape[0])]
        for i in range(out_label_ids.shape[0]):
            sent_scores = torch.tensor(emissions_list[i])
            sent_len, n_label = sent_scores.shape
            sent_probs = F.softmax(sent_scores, dim=1)
            start_probs = torch.zeros(sent_len) + 1e-6
            sent_probs = torch.cat((start_probs.view(sent_len, 1), sent_probs), 1)
            feats = viterbi_decoder.forward(torch.log(sent_probs).view(1, sent_len, n_label+1))
            vit_labels = viterbi_decoder.viterbi(feats)
            vit_labels = vit_labels.view(sent_len)
            vit_labels = vit_labels.detach().cpu().numpy()
            for label in vit_labels:
                preds_list[i].append(IO_label_map[label-1])

    results = {
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list),
    }
    print(results)

    # Log and save results to file
    output_test_results_file = os.path.join(args.output_dir, "test_results.txt")
    with open(output_test_results_file, "w") as writer:
        for key in results:
            writer.write("{} = {}\n".format(key, str(results[key])))

    test_file = os.path.join(args.data_dir, args.test_fname + ".txt")
    output_test_preds_file = os.path.join(args.output_dir, "test_preds.txt")
    with open(test_file, "r") as reader, open(output_test_preds_file, "w") as writer:
        ner_task.write_predictions_to_file(writer, reader, preds_list)


def nn_decode(reps, support_reps, support_tags):
    """
    NNShot: neariest neighbor decoder for few-shot NER
    """
    batch_size, sent_len, ndim = reps.shape
    scores = _euclidean_metric(reps.view(-1, ndim), support_reps, True)
    # tags = support_tags[torch.argmax(scores, 1)]
    emissions = get_nn_emissions(scores, support_tags)
    tags = torch.argmax(emissions, 1)
    return tags.view(batch_size, sent_len), emissions.view(batch_size, sent_len, -1)


def get_nn_emissions(scores, tags):
    """
    Obtain emission scores from NNShot
    """
    n, m = scores.shape
    n_tags = torch.max(tags) + 1
    emissions = -100000. * torch.ones(n, n_tags).to(scores.device)
    for t in range(n_tags):
        mask = (tags == t).float().view(1, -1)
        masked = scores * mask
        masked = torch.where(masked < 0, masked, torch.tensor(-100000.).to(scores.device))
        emissions[:, t] = torch.max(masked, dim=1)[0]
    return emissions


def _euclidean_metric(a, b, normalize=False):
    if normalize:
        a = F.normalize(a)
        b = F.normalize(b)
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b) ** 2).sum(dim=2)
    return logits


def get_abstract_transitions(data_dir, data_fname):
    """
    Compute abstract transitions on the training dataset for StructShot
    """
    examples = ner_task.read_examples_from_file(data_dir, data_fname)
    tag_lists = [example.labels for example in examples]

    s_o, s_i = 0., 0.
    o_o, o_i = 0., 0.
    i_o, i_i, x_y = 0., 0., 0.
    for tags in tag_lists:
        if tags[0] == 'O': s_o += 1
        else: s_i += 1
        for i in range(len(tags)-1):
            p, n = tags[i], tags[i+1]
            if p == 'O':
                if n == 'O': o_o += 1
                else: o_i += 1
            else:
                if n == 'O':
                    i_o += 1
                elif p != n:
                    x_y += 1
                else:
                    i_i += 1

    trans = []
    trans.append(s_o / (s_o + s_i))
    trans.append(s_i / (s_o + s_i))
    trans.append(o_o / (o_o + o_i))
    trans.append(o_i / (o_o + o_i))
    trans.append(i_o / (i_o + i_i + x_y))
    trans.append(i_i / (i_o + i_i + x_y))
    trans.append(x_y / (i_o + i_i + x_y))
    return trans

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_generic_args(parser, os.getcwd())
    parser = NERTransformer.add_model_specific_args(parser, os.getcwd())
    parser.add_argument(
            "--checkpoint",
            default=None,
            type=str,
            required=True,
            help="Path to the PTL checkpoint",
    )
    parser.add_argument(
        "--target_labels",
        default="",
        type=str,
        help="Path to a file containing all target labels. If not specified, CoNLL-2003 labels are used.",
    )
    parser.add_argument(
        "--train_fname", 
        default=None,
        type=str,
        required=True,
        help="data file name for training set",
    )
    parser.add_argument(
        "--sup_fname", 
        default=None,
        type=str,
        required=True,
        help="data file name for support set",
    )
    parser.add_argument(
        "--test_fname", 
        default=None,
        type=str,
        required=True,
        help="data file name for test set",
    )
    parser.add_argument(
        "--algorithm",
        default="NNShot",
        choices=["NNShot", "StructShot"],
        help="Few-shot NER algorithm options",
    )
    parser.add_argument(
        "--tau",
        default=0.1,
        type=float,
        help="StructShot parameter to re-normalizes the transition probabilities",
    )
    args = parser.parse_args()
    model = NERTransformer(args)
    trainer = generic_train(model, args)

    model = model.load_from_checkpoint(args.checkpoint)
    evaluate_few_shot(args, model)
