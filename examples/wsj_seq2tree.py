import os
import argparse
import logging
import torch
from torch.optim.lr_scheduler import StepLR
import sys
sys.path=['/home/liuyx/PycharmProjects/pytorch-seq2tree-master/examples']+sys.path
sys.path=['/home/liuyx/PycharmProjects/pytorch-seq2tree-master']+sys.path
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
from seq2seq.evaluator import Evaluator
from torch import optim
import pickle


def type_in():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-path', action='store', dest='train_path',
                        help='Path to train data')
    parser.add_argument('--dev-path', action='store', dest='dev_path',
                        help='Path to dev data')
    parser.add_argument('--expt-dir', action='store', dest='expt_dir', default='./experiment',
                        help='Path to experiment directory. If load_checkpoint is True, then path to checkpoint directory has to be provided')
    parser.add_argument('--load-checkpoint', action='store', dest='load_checkpoint',
                        help='The name of the checkpoint to load, usually an encoded time string')
    parser.add_argument('--resume', action='store_true', dest='resume',
                        default=False,
                        help='Indicates if training has to be resumed from the latest checkpoint')
    parser.add_argument('--GPU', default=-1, dest='GPU', type=int)
    parser.add_argument('--log-level', dest='log_level',
                        default='info',
                        help='Logging level.')
    parser.add_argument('--epoch', default=10, dest='epoch', type=int)
    parser.add_argument('--max-len', default=20, dest='max_len', type=int)
    parser.add_argument('--hidden-size', default=32, dest='hidden_size', type=int)
    parser.add_argument('--word-embedding-size', default=100, dest='word_embedding_size', type=int)
    parser.add_argument('--nt-embedding-size', default=32, dest='nt_embedding_size', type=int)
    parser.add_argument('--word-embedding', default=None, dest='word_embedding')
    parser.add_argument('--batch-size', default=1, dest='batch_size', type=int)
    parser.add_argument('--checkpoint-every', default=10000 , dest='checkpoint_every', type=int)
    parser.add_argument('--print-every', default=100, dest='print_every', type=int)
    parser.add_argument('--bidirectional-encoder', default=True, dest='bidirectional_encoder')
    parser.add_argument('--teacher-forcing-ratio', default=0.5, dest='teacher_forcing_ratio', type=float)
    parser.add_argument('--lr', default=1e-4, dest='lr', type=float)
    parser.add_argument('--drop-out', default=0.2, dest = 'drop_out', type=float)
    opt = parser.parse_args()
    return opt

def train(opt):
    LOG_FORMAT = '%(asctime)s %(levelname)-8s %(message)s'
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

    else:
        # Prepare dataset
        src = SourceField()
        nt = NTField()
        pos = PosField()
        tgt_tree = TreeField()
        comp= CompField()
        max_len = opt.max_len
        def len_filter(example):
            return len(example.src) <= max_len
        train = torchtext.data.TabularDataset(
            path=opt.train_path, format='tsv',
            fields=[('src', src), ('nt', nt), ('pos', pos), ('tree', tgt_tree)],
            filter_pred=len_filter
        )
        dev = torchtext.data.TabularDataset(
            path=opt.dev_path, format='tsv',
            fields=[('src', src), ('nt', nt), ('pos', pos), ('tree', tgt_tree)],
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
        hidden_size = opt.hidden_size
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
                return tree

        # train.examples = [str(tree_to_id(ex.tree)) for ex in train.examples]
        # dev.examples = [str(tree_to_id(ex.tree)) for ex in dev.examples]
        for ex in train.examples:
            ex.tree = str(tree_to_id(Tree.fromstring(ex.tree)))
        for ex in dev.examples:
            ex.tree = str(tree_to_id(Tree.fromstring(ex.tree)))
        # train.examples = [tree_to_id(Tree.fromstring(ex.tree)) for ex in train.examples]
        # dev.examples = [str(tree_to_id(Tree.fromstring(ex.tree))) for ex in dev.examples]
        if opt.word_embedding is not None:
            input_vocab.load_vectors([opt.word_embedding])

        loss = NLLLoss()
        if torch.cuda.is_available():
            loss.cuda()
        loss.reset()
        seq2tree = None
        optimizer = None
        if not opt.resume:
            # Initialize model
            bidirectional = opt.bidirectional_encoder
            encoder = EncoderRNN(len(src.vocab), opt.word_embedding_size, max_len, hidden_size,
                                 bidirectional=bidirectional, variable_lengths=True)
            decoder = DecoderTree(len(src.vocab), opt.word_embedding_size, opt.nt_embedding_size, len(nt.vocab),
                                  max_len, hidden_size * 2 if bidirectional else hidden_size,
                                  sos_id=nt_vocab.stoi['<sos>'], eos_id=nt_vocab.stoi['<eos>'],
                                 dropout_p=0.2, use_attention=True, bidirectional=bidirectional,
                                  pos_in_nt = pos_in_nt)

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

            optimizer = Optimizer(optim.Adam(seq2tree.parameters(), lr=opt.lr), max_grad_norm=5)
        # train
        t = SupervisedTrainer(loss=loss, batch_size=opt.batch_size,
                              checkpoint_every=opt.checkpoint_every,
                              print_every=10, expt_dir=opt.expt_dir, lr=opt.lr)

        seq2tree = t.train(seq2tree, train,
                          num_epochs=opt.epoch, dev_data=dev,
                          optimizer=optimizer,
                          teacher_forcing_ratio=0,
                          resume=opt.resume)

    predictor = Predictor(seq2tree, input_vocab, nt_vocab)
    return predictor, dev, train
def predict(predictor):
    while True:
        tree_or_sentence = input("Type in a tree or a sentence?")
        if tree_or_sentence != 'tree':
            seq_str = input("Type in a source sequence:")
            if seq_str == ':end':
                break
            seq = seq_str.strip().split()
            print(predictor.predict(seq))
        else:
            tree_str = input("Type in a source sequence:")
            try:
                tree = Tree.fromstring(tree_str)
                seq = tree.leaves()
                print(predictor.predict(seq,tree))
            except:
                print('The input is not a valid tree.')


if __name__=='__main__':
    opt = type_in()
    predictor, dev,train = train(opt)
    evaluator = Evaluator(loss=NLLLoss(), batch_size=1)
    # dev_loss, accuracy, tree_acc = evaluator.evaluate(predictor.model, dev)
    # print(accuracy)
    # dev_loss, accuracy, tree_acc = evaluator.evaluate(predictor.model, train)
    # print(accuracy)
    predict(predictor)
