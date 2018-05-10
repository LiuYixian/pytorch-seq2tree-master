import os
import argparse
import logging

import torch
from torch.optim.lr_scheduler import StepLR

import sys

sys.path=['/home/liuyx/PycharmProjects/pytorch-seq2seq-master/examples']+sys.path
sys.path=['/home/liuyx/PycharmProjects/pytorch-seq2seq-master']+sys.path
for a in sys.path:
    print(a)
# import seq2seq
import torchtext
from seq2seq.trainer import SupervisedTrainer
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq, DecoderTree, Seq2tree
from seq2seq.loss import Perplexity, NLLLoss
from seq2seq.optim import Optimizer
from seq2seq.dataset import SourceField, TargetField, CompField, NTField, TreeField, PosField
from seq2seq.evaluator import Predictor
from seq2seq.util.checkpoint import Checkpoint
from nltk.tree import Tree
from torch import optim
import pickle
global m
m=1
try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3

# Sample usage:
#     # training
#     python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH
#     # resuming from the latest checkpoint of the experiment
#      python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH --resume
#      # resuming from a specific checkpoint
#      python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH --load_checkpoint $CHECKPOINT_DIR

parser = argparse.ArgumentParser()
parser.add_argument('--train_path', action='store', dest='train_path',
                    help='Path to train data')
parser.add_argument('--dev_path', action='store', dest='dev_path',
                    help='Path to dev data')
parser.add_argument('--expt_dir', action='store', dest='expt_dir', default='./experiment',
                    help='Path to experiment directory. If load_checkpoint is True, then path to checkpoint directory has to be provided')
parser.add_argument('--load_checkpoint', action='store', dest='load_checkpoint',
                    help='The name of the checkpoint to load, usually an encoded time string')
parser.add_argument('--resume', action='store_true', dest='resume',
                    default=False,
                    help='Indicates if training has to be resumed from the latest checkpoint')
parser.add_argument('--GPU', default=-1, dest='GPU')
parser.add_argument('--log-level', dest='log_level',
                    default='info',
                    help='Logging level.')

opt = parser.parse_args()

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, opt.log_level.upper()))
logging.info(opt)
if int(opt.GPU)>=0:
    torch.cuda.set_device(int(opt.GPU))
if opt.load_checkpoint is not None:
    logging.info("loading checkpoint from {}".format(os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)))
    checkpoint_path = os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)
    checkpoint = Checkpoint.load(checkpoint_path)
    seq2tree = checkpoint.model
    input_vocab = checkpoint.input_vocab
    output_vocab = checkpoint.output_vocab
else:
    # Prepare dataset
    src = SourceField()
    nt = NTField()
    pos = PosField()
    tgt_tree = TreeField()
    comp= CompField()
    max_len = 50
    def len_filter(example):
        return len(example.src) <= max_len
    train = torchtext.data.TabularDataset(
        path=opt.train_path, format='tsv',
        fields=[('src', src), ('nt', nt), ('pos', pos), ('tree', tgt_tree) ],
        filter_pred=len_filter
    )
    dev = torchtext.data.TabularDataset(
        path=opt.dev_path, format='tsv',
        fields=[('src', src), ('nt', nt), ('pos', pos), ('tree', tgt_tree) ],
        filter_pred=len_filter
    )
    src.build_vocab(train, max_size=50000)
    comp.build_vocab(train, max_size=50000)
    nt.build_vocab(train, max_size=50000)
    pos.build_vocab(train, max_size=50000)
    # src_tree.build_vocab(train, max_size=50000)
    pos_in_nt = set()
    for Pos in pos.vocab.stoi:
        if nt.vocab.stoi[Pos] > 1:
            pos_in_nt.add(nt.vocab.stoi[Pos])
    hidden_size = 300
    input_vocab = src.vocab
    nt_vocab = nt.vocab
    def tree_to_id(tree):
        tree.set_label(nt_vocab.stoi[tree.label()])
        if len(tree) == 1 and str(tree[0])[0] is not '(':
            tree[0] = input_vocab.stoi[tree[0]]
            return
        else:
            for subtree in tree:
                tree_to_id(subtree)
            tree.append(Tree(nt_vocab.stoi['<eos>'],[]))
            return

    for ex in train.examples:
        tree = Tree.fromstring(ex.tree)
        tree_to_id(tree)
        ex.tree = str(tree)
    for ex in dev.examples:
        tree = Tree.fromstring(ex.tree)
        tree_to_id(tree)
        ex.tree = str(tree)

    # input_vocab.load_vectors([])
    #
    input_vocab.load_vectors(['glove.840B.300d'])

    #
    input_vocab.vectors[input_vocab.stoi['<unk>']] = torch.Tensor(hidden_size).uniform_(-0.8,0.8)#<unk>
    # input_vocab.vectors[input_vocab.stoi['<pad>']] = torch.Tensor(hidden_size).uniform_(-0.8,0.8)#<pad>
    # output_vocab.vectors[output_vocab.stoi['<pad>']] = torch.Tensor(hidden_size).uniform_(-0.8,0.8)#<unk>
    # output_vocab.vectors[output_vocab.stoi['<sos>']] = torch.Tensor(hidden_size).uniform_(-0.8,0.8)#<sos>
    # output_vocab.vectors[output_vocab.stoi['<eos>']] = torch.Tensor(hidden_size).uniform_(-0.8,0.8)#<eos>

    # NOTE: If the source field name and the target field name
    # are different from 'src' and 'tgt' respectively, they have
    # to be set explicitly before any training or inference
    # seq2seq.src_field_name = 'src'
    # seq2seq.tgt_field_name = 'tgt'

    # Prepare loss

    # loss = NLLLoss(weight, pad)#Perplexity(weight, pad)
    loss = NLLLoss()

    if torch.cuda.is_available():
        loss.cuda()
    loss.reset()
    seq2tree = None

    if not opt.resume:
        # Initialize model

        bidirectional = True
        encoder = EncoderRNN(len(src.vocab), max_len, hidden_size,
                             bidirectional=bidirectional, variable_lengths=True)
        decoder = DecoderTree(len(src.vocab), len(nt.vocab),max_len, hidden_size * 2 if bidirectional else hidden_size,
                             dropout_p=0.2, use_attention=True, bidirectional=bidirectional, pos_in_nt = pos_in_nt)

        seq2tree = Seq2tree(encoder, decoder)
        if torch.cuda.is_available():
            seq2tree.cuda()

        for param in seq2tree.parameters():
            param.data.uniform_(-0.08, 0.08)
            # encoder.embedding.weight.data.set_(input_vocab.vectors)
            # encoder.embedding.weight.data.set_(output_vocab.vectors)

        # Optimizer and learning rate scheduler can be customized by
        # explicitly constructing the objects and pass to the trainer.
        #
        # optimizer = Optimizer(torch.optim.Adam(seq2seq.parameters()), max_grad_norm=5)
        # scheduler = StepLR(optimizer.optimizer, 1)
        # optimizer.set_scheduler(scheduler)

    optimizer = Optimizer(optim.Adam(seq2tree.parameters(), lr=1e-4), max_grad_norm=5)
    # train
    t = SupervisedTrainer(loss=loss, batch_size=1,
                          checkpoint_every=50,
                          print_every=10, expt_dir=opt.expt_dir)

    seq2tree = t.train(seq2tree, train,
                      num_epochs=20, dev_data=dev,
                      optimizer=optimizer,
                      teacher_forcing_ratio=0.5,
                      resume=opt.resume)

predictor = Predictor(seq2tree, input_vocab, input_vocab)

while True:
    seq_str = raw_input("Type in a source sequence:")
    seq = seq_str.strip().split()
    print(predictor.predict(seq))
