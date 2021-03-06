import numpy as np
import h5py
import sys
import torch
import os
import random
import logging
import random
import string
import time

import nemo
from torch.utils.data import Dataset
from nemo_nlp.data.data_layers import TextDataLayer
from nemo.backends.pytorch.nm import DataLayerNM
from nemo.core.neural_types import *
from nemo_nlp.data.datasets import utils


class PreprocBertSentenceClassificationDataset(Dataset):
    
    def __init__(self, input_file, num_samples=-1, shuffle=False):
        self.input_file = input_file
        self.num_samples = num_samples
        self.shuffle = shuffle

        f = h5py.File(input_file, 'r')
        keys = ['tokens', 'token_types', 'attn_mask', 'labels']
        self.inputs = [np.asarray(f[key], dtype=np.long) for key in keys]
        f.close()
        
        if self.shuffle:
            np.random.seed(123)
        
        if self.shuffle:
            np.random.seed(123)
            idx = np.arange(len(self))
            np.random.shuffle(idx) # shuffle idx in place
            shuffled_inputs = [arr[idx] for arr in self.inputs]
            self.inputs = shuffled_inputs
        
        if self.num_samples > 0:
            truncated_inputs = [arr[0:self.num_samples] for arr in self.inputs]
            self.inputs = truncated_inputs
            
            
    def __len__(self):
        return len(self.inputs[0])
    
    
    def __getitem__(self, index):
        return (self.inputs[0][index], self.inputs[1][index], self.inputs[2][index], self.inputs[3][index])
    
class PreprocBertSentenceClassificationDataLayer(TextDataLayer):
    
    @staticmethod
    def create_ports():
        output_ports = {
            "input_ids": NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            }),
            "input_type_ids": NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            }),
            "input_mask": NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            }),
            "labels": NeuralType({
                0: AxisType(BatchTag),
            }),
        }
        return {}, output_ports
    
    
    def __init__(self,
                 input_file,
                 num_samples=-1,
                 shuffle=False,
                 batch_size=64,
                 dataset_type=PreprocBertSentenceClassificationDataset,
                 **kwargs):
        kwargs['batch_size'] = batch_size
        dataset_params = {'input_file': input_file,
                          'num_samples': num_samples,
                          'shuffle': shuffle}
        super().__init__(dataset_type, dataset_params, **kwargs)


"""Below taken from 
https://github.com/NVIDIA/NeMo/blob/master/collections/nemo_nlp/nemo_nlp/data/datasets/sentence_classification.py
in order to keep max_seq_length
Utility functions for Token Classification NLP tasks
Some parts of this code were adapted from the HuggingFace library at
https://github.com/huggingface/pytorch-pretrained-BERT
"""


logger = logging.getLogger('log')


class BertSentenceClassificationDataset(Dataset):
    """A dataset class that converts from raw data to
    a dataset that can be used by DataLayerNM.
    Args:
        input_file (str): file to sequence + label.
            the first line is header (sentence [tab] label)
            each line should be [sentence][tab][label]
        max_seq_length (int): max sequence length minus 2 for [CLS] and [SEP]
        tokenizer (Tokenizer): such as BertTokenizer
        num_samples (int): number of samples you want to use for the dataset.
            If -1, use all dataset. Useful for testing.
        shuffle (bool): whether to shuffle your data.
    """

    def __init__(self,
                 input_file,
                 max_seq_length,
                 tokenizer,
                 num_samples=-1,
                 shuffle=True):
        with open(input_file, "r") as f:
            sent_labels, all_sent_subtokens = [], []
            sent_lengths = []
            too_long_count = 0

            lines = f.readlines()[1:]
            logger.info(f'{input_file}: {len(lines)}')

            if shuffle or num_samples > -1:
                random.seed(0)
                random.shuffle(lines)
                if num_samples > 0:
                    lines = lines[:num_samples]

            for index, line in enumerate(lines):
                if index % 20000 == 0:
                    logger.debug(f"Processing line {index}/{len(lines)}")

                sent_label = int(line.split()[-1])
                sent_labels.append(sent_label)
                sent_words = line.strip().split()[:-1]
                sent_subtokens = ['[CLS]']

                for word in sent_words:
                    word_tokens = tokenizer.tokenize(word)
                    sent_subtokens.extend(word_tokens)

                sent_subtokens.append('[SEP]')

                all_sent_subtokens.append(sent_subtokens)
                sent_lengths.append(len(sent_subtokens))

        utils.get_stats(sent_lengths)
        # Below commented because we want to keep max_seq_length
        #self.max_seq_length = min(max_seq_length, max(sent_lengths))
        self.max_seq_length = max_seq_length

        for i in range(len(all_sent_subtokens)):
            if len(all_sent_subtokens[i]) > self.max_seq_length:
                shorten_sent = all_sent_subtokens[i][-self.max_seq_length+1:]
                all_sent_subtokens[i] = ['[CLS]'] + shorten_sent
                too_long_count += 1

        logger.info(f'{too_long_count} out of {len(sent_lengths)} \
                       sentencess with more than {max_seq_length} subtokens.')

        self.convert_sequences_to_features(all_sent_subtokens,
                                           sent_labels,
                                           tokenizer,
                                           self.max_seq_length)

        self.tokenizer = tokenizer
        self.vocab_size = self.tokenizer.vocab_size

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):

        feature = self.features[idx]

        return (np.array(feature.input_ids),
                np.array(feature.segment_ids),
                np.array(feature.input_mask, dtype=np.long),
                feature.sent_label)

    def convert_sequences_to_features(self,
                                      all_sent_subtokens,
                                      sent_labels,
                                      tokenizer,
                                      max_seq_length):
        """Loads a data file into a list of `InputBatch`s.
        """

        self.features = []
        for sent_id in range(len(all_sent_subtokens)):
            sent_subtokens = all_sent_subtokens[sent_id]
            sent_label = sent_labels[sent_id]
            word_count = 0
            # input_ids = tokenizer.tokens_to_ids(sent_subtokens)
            input_ids = [tokenizer._convert_token_to_id(
                t) for t in sent_subtokens]

            # The mask has 1 for real tokens and 0 for padding tokens.
            # Only real tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
            segment_ids = [0] * max_seq_length

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length

            if sent_id == 0:
                logger.info("*** Example ***")
                logger.info("example_index: %s" % sent_id)
                logger.info("subtokens: %s" % " ".join(sent_subtokens))
                logger.info("sent_label: %s" % sent_label)
                logger.info("input_ids: %s" % utils.list2str(input_ids))
                logger.info("input_mask: %s" % utils.list2str(input_mask))

            self.features.append(InputFeatures(
                sent_id=sent_id,
                sent_label=sent_label,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids))


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 sent_id,
                 sent_label,
                 input_ids,
                 input_mask,
                 segment_ids):
        self.sent_id = sent_id
        self.sent_label = sent_label
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids