import nemo
from nemo.utils.lr_policies import get_lr_policy
import nemo_nlp
from nemo_nlp.data.datasets.utils import SentenceClassificationDataDesc
from nemo_nlp.utils.callbacks.sentence_classification import \
    eval_iter_callback, eval_epochs_done_callback

from nemo.core import NeuralModuleFactory
from nemo.backends.pytorch.common import CrossEntropyLoss

from pytorch_transformers import BertTokenizer
import torch.nn.functional as f

import math
import numpy as np
import pandas as pd
pd.options.display.max_colwidth = -1

import json
import argparse

import preproc_data_layer

def parse_args():
    parser = argparse.ArgumentParser(description="Classify sentences with BERT Fine-tuning")

    # Parsing arguments
    parser.add_argument("--train_file", default=None, type=str,
                        help="The input should contain the .tsv    \
                        formatted with the header 'sentence\tlabel'. \
                        Weights will be optimized using this data.")
    parser.add_argument("--eval_file", default=None, type=str,
                        help="The input should contain the .tsv    \
                        formatted with the header 'sentence\tlabel'. \
                        Weights not optimized, loss computed.")
    parser.add_argument("--inference_file", default=None, type=str,
                        help="The input should contain the .tsv    \
                        formatted with the header 'sentence\tlabel'. \
                        Weights not optimized, loss not computed.")
    parser.add_argument("--pretrained_bert_model", default="bert-base-uncased",
                        type=str, help="Name of the pre-trained model")
    parser.add_argument("--bert_checkpoint", default=None, type=str,
                        help="Path to bert model checkpoint")
    parser.add_argument("--bert_config", default=None, type=str,
                        help="Path to bert config file in json format")
    parser.add_argument("--tokenizer_model", default="tokenizer.model", type=str,
                        help="Path to pretrained tokenizer model, \
                        only used if --tokenizer is sentencepiece")
    parser.add_argument("--tokenizer", default="nemobert", type=str,
                        choices=["nemobert", "sentencepiece"],
                        help="tokenizer to use, \
                        only relevant when using custom pretrained checkpoint.")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        choices=range(1, 513),
                        help="The maximum total input sequence length after   \
                        tokenization.Sequences longer than this will be       \
                        truncated, sequences shorter will be padded.")
    parser.add_argument("--mlp_checkpoint", default=None, type=str,
                        help="Path to mlp model checkpoint")
    parser.add_argument("--optimizer_kind", default="adam", type=str,
                        help="Optimizer kind")
    parser.add_argument("--lr_policy", default="WarmupAnnealing", type=str)
    parser.add_argument("--lr", default=5e-5, type=float,
                        help="The initial learning rate.")
    parser.add_argument("--lr_warmup_proportion", default=0.1, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_epochs", default=3, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--batch_size", default=8, type=int,
                        help="Batch size per GPU for training, evaluation, and inference.")
    parser.add_argument("--num_gpus", default=1, type=int,
                        help="Number of GPUs")
    parser.add_argument("--amp_opt_level", default="O0", type=str,
                        choices=["O0", "O1", "O2"],
                        help="O1/O2 to enable mixed precision")
    parser.add_argument("--local_rank", type=int, default=None,
                        help="For torch distributed training: local_rank (see torch.distributed.launch)")
    parser.add_argument("--work_dir", default='work_dir/', type=str,
                        help="The output directory where the model predictions \
                        and checkpoints will be written.")
    parser.add_argument("--save_epoch_freq", default=1, type=int,
                        help="Frequency of saving checkpoint \
                        '-1' - epoch checkpoint won't be saved")
    parser.add_argument("--save_step_freq", default=-1, type=int,
                        help="Frequency of saving checkpoint \
                        '-1' - step checkpoint won't be saved")
    parser.add_argument("--num_checkpoints", default=3, type=int,
                        help="The number of checkpoints to keep. -1 to keep all")
    parser.add_argument("--loss_step_freq", default=1, type=int,
                        help="Frequency of printing loss")
    parser.add_argument("--mode", default='train', type=str,
                        choices=["train", "eval", "inference"],
                        help="Type of pipeline")
    parser.add_argument("--num_classes", default=None, type=int, required=True,
                        help="Number of classes to be classified")
    parser.add_argument("--dropout", default=0.1, type=float,
                        help="Dropout for mlp layers")
    parser.add_argument("--num_layers", default=1, type=int,
                        help="Number of layers in MLP")
    parser.add_argument("--num_samples", default=-1, type=int,
                        help="Used to reduce dataset size. -1 for all dataset")
    parser.add_argument("--preproc", default=False, action='store_true',
                        help="Use preprocessed data.")

    args = parser.parse_args()

    return args


def create_pipeline(
    nf,
    data_layer,
    bert_model,
    mlp,
    loss_fn):

    tokens, token_types, attn_mask, labels = data_layer()

    embeddings = bert_model(
        input_ids=tokens,
        token_type_ids=token_types,
        attention_mask=attn_mask)

    logits = mlp(hidden_states=embeddings)

    if loss_fn:
        loss = loss_fn(logits=logits, labels=labels)
    else:
        loss = None

    num_gpus = nf.world_size
    batch_size = data_layer.local_parameters['batch_size']
    steps_per_epoch = len(data_layer) // (batch_size * num_gpus)

    return logits, loss, steps_per_epoch, labels 


def sentence_classification(args):
    # TODO: construct name of experiment based on args
    """
    name = construct_name(
            args.exp_name,
            args.lr,
            args.batch_size,
            args.num_epochs,
            args.weight_decay,
            args.optimizer)
    work_dir = name
    if args.work_dir:
        work_dir = os.path.join(args.work_dir, name)
    """
    # Instantiate neural modules
    nf = NeuralModuleFactory(
        backend=nemo.core.Backend.PyTorch,
        local_rank=args.local_rank,
        optimization_level=args.amp_opt_level,
        log_dir=args.work_dir,
        create_tb_writer=True,
        files_to_copy=[__file__],
        add_time_to_log_dir=True)

    # Pre-trained BERT
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_bert_model)

    if args.bert_checkpoint is None:
        bert = nemo_nlp.BERT(pretrained_model_name=args.pretrained_bert_model)
        # save bert config for inference after fine-tuning
        bert_config = bert.config.to_dict()
        with open(args.work_dir + '/' + args.pretrained_bert_model + '_config.json', 'w+') as json_file:
            json.dump(bert_config, json_file)
    else:
        if args.bert_config is not None:
            with open(args.bert_config) as json_file:
                bert_config = json.load(json_file)
        bert = nemo_nlp.BERT(**bert_config)
        bert.restore_from(args.bert_checkpoint)

    # MLP
    bert_hidden_size = bert.local_parameters['hidden_size']
    mlp = nemo_nlp.SequenceClassifier(
        hidden_size=bert_hidden_size,
        num_classes=args.num_classes,
        num_layers=args.num_layers,
        log_softmax=False,
        dropout=args.dropout)

    # TODO: save mlp/all model configs (bake in to Neural Module?)

    if args.mlp_checkpoint:
        mlp.restore_from(args.mlp_checkpoint)
    
    # Loss function for classification
    loss_fn = CrossEntropyLoss()

    # Data layers, pipelines, and callbacks
    callbacks = [] # callbacks depend on files present

    if args.train_file:
        if args.preproc:
            train_data_layer = preproc_data_layer.PreprocBertSentenceClassificationDataLayer(
            input_file=args.train_file,
            shuffle=True,
            num_samples=args.num_samples, # lower for dev, -1 for all dataset
            batch_size=args.batch_size,
            num_workers=0,
            local_rank=args.local_rank)

        else:
            train_data_layer = nemo_nlp.BertSentenceClassificationDataLayer(
            input_file=args.train_file,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            shuffle=True,
            num_samples=args.num_samples, # lower for dev, -1 for all dataset
            batch_size=args.batch_size,
            num_workers=0,
            local_rank=args.local_rank)

        train_logits, train_loss, steps_per_epoch, train_labels = create_pipeline(
            nf,
            train_data_layer,
            bert,
            mlp,
            loss_fn)

        train_callback = nemo.core.SimpleLossLoggerCallback(
            tensors=[train_loss, train_logits],
            print_func=lambda x: nf.logger.info(f'Train loss: {str(np.round(x[0].item(), 3))}'),
            tb_writer=nf.tb_writer,
            get_tb_values=lambda x: [["train_loss", x[0]]],
            step_freq=steps_per_epoch)

        callbacks.append(train_callback)

        if args.num_checkpoints != 0:
            ckpt_callback = nemo.core.CheckpointCallback(
                folder=nf.checkpoint_dir,
                epoch_freq=args.save_epoch_freq,
                step_freq=args.save_step_freq,
                checkpoints_to_keep=args.num_checkpoints)
            
            callbacks.append(ckpt_callback)
        

    if args.eval_file:
        if args.preproc:
            eval_data_layer = preproc_data_layer.PreprocBertSentenceClassificationDataLayer(
            input_file=args.eval_file,
            shuffle=False,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            num_workers=0,
            local_rank=args.local_rank)
        
        else:
            eval_data_layer = nemo_nlp.BertSentenceClassificationDataLayer(
            input_file=args.eval_file,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            shuffle=False,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            num_workers=0,
            local_rank=args.local_rank)

        eval_logits, eval_loss, _, eval_labels = create_pipeline(
            nf,
            eval_data_layer,
            bert,
            mlp,
            loss_fn)
        
        eval_callback = nemo.core.EvaluatorCallback(
            eval_tensors=[eval_logits, eval_labels],
            user_iter_callback=lambda x, y: eval_iter_callback(
                x, y, eval_data_layer),
            user_epochs_done_callback=lambda x: eval_epochs_done_callback(
                x, f'{nf.work_dir}/graphs'),
            tb_writer=nf.tb_writer,
            eval_step=steps_per_epoch)

        callbacks.append(eval_callback)
    
    if args.inference_file:
        if args.preproc:
            inference_data_layer = preproc_data_layer.PreprocBertSentenceClassificationDataLayer(
            input_file=args.inference_file,
            shuffle=False,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            num_workers=0,
            local_rank=args.local_rank)
        
        else:
            inference_data_layer = nemo_nlp.BertSentenceClassificationDataLayer(
            input_file=args.inference_file,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            shuffle=False,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            num_workers=0,
            local_rank=args.local_rank)

        # TODO: Finish inference
        inference_callback = None 

    # Training, eval and inference
    if args.train_file:
        lr_policy_fn = get_lr_policy(
            args.lr_policy,
            total_steps=args.num_epochs * steps_per_epoch,
            warmup_ratio=args.lr_warmup_proportion)

        nf.train(
            tensors_to_optimize=[train_loss],
            callbacks=callbacks,
            lr_policy=lr_policy_fn,
            optimizer=args.optimizer_kind,
            optimization_params={'num_epochs': args.num_epochs, 'lr': args.lr})





if __name__ == '__main__':
    args = parse_args()
    sentence_classification(args)