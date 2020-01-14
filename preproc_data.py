import nemo
import nemo_nlp
from pytorch_transformers import BertTokenizer
import numpy as np
import math
import json
import h5py
import os
import argparse
from nemo_nlp.data.datasets.sentence_classification import BertSentenceClassificationDataset

def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess data for sentence classification with with BERT")
    parser.add_argument(
        "--input_file", default="data/input.tsv", type=str,
        help="The input should contain the .tsv formatted with the header 'sentence\tlabel'.")
    parser.add_argument(
        "--output_dir", default="preproc_data/", type=str,
        help="The output can be used by sentence_classification.py for training and inference.")
    parser.add_argument(
        "--dataset_name", default="dataset", type=str,
        help="Used for logging.")
    parser.add_argument(
        "--max_seq_length", default=64, type=int,
        help="Padded by 0's if shorter, truncated if longer."
    )
    parser.add_argument(
        "--pretrained_bert_model", type=str,
        help="Name of the pre-trained model."
    )
    args = parser.parse_args()
    return args

def preproc_data(args):
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    tokenizer = BertTokenizer.from_pretrained(args.pretrained_bert_model)

    dataset = BertSentenceClassificationDataset(
        args.input_file,
        args.max_seq_length,
        tokenizer,
        num_samples=-1,
        shuffle=False)

    token_array = np.zeros((len(dataset), dataset.max_seq_length))
    token_types_array = np.zeros((len(dataset), dataset.max_seq_length))
    attn_mask_array = np.zeros((len(dataset), dataset.max_seq_length))
    labels_array = np.zeros((len(dataset),))
    for idx in range(len(dataset)):
        tokens, token_types, attn_mask, labels = dataset[idx]
        token_array[idx] = tokens
        token_types_array[idx] = token_types
        attn_mask_array[idx] = attn_mask
        labels_array[idx] = labels

    hdf5_filename = f'{args.dataset_name}_{args.pretrained_bert_model}_{args.max_seq_length}.hdf5'
    hdf5_path = os.path.join(args.output_dir, hdf5_filename)
    f = h5py.File(hdf5_path, mode='w')
    f.create_dataset('tokens', data=token_array)
    f.create_dataset('token_types', data=token_types_array)
    f.create_dataset('attn_mask', data=attn_mask_array)
    f.create_dataset('labels', data=labels_array)
    f.close()

if __name__ == '__main__':
    args = parse_args()
    preproc_data(args)