import torch
from torch.autograd import Variable
import seq2seq
from nltk.tree import Tree
from copy import deepcopy
class Predictor(object):

    def __init__(self, model, src_vocab, nt_vocab):
        """
        Predictor class to evaluate for a given model.
        Args:
            model (seq2seq.models): trained model. This can be loaded from a checkpoint
                using `seq2seq.util.checkpoint.load`
            src_vocab (seq2seq.dataset.vocabulary.Vocabulary): source sequence vocabulary
            tgt_vocab (seq2seq.dataset.vocabulary.Vocabulary): target sequence vocabulary
        """
        if torch.cuda.is_available():
            self.model = model.cuda()
        else:
            self.model = model.cpu()
        self.model.eval()
        self.src_vocab = src_vocab
        self.nt_vocab = nt_vocab

    def tree_to_id(self, tree):
        for sub_tree in tree.subtrees():
            if isinstance(sub_tree.label(), str):
                continue
            if int(sub_tree.label()) > 2:
                sub_tree.set_label(self.nt_vocab.itos[int(sub_tree.label())])
                if not isinstance(sub_tree[0], Tree):
                    sub_tree[0] = self.src_vocab.itos[int(sub_tree[0])]
        return tree
    def compare(self, tree, tgt_tree):
        right = 0
        total = 0
        if tree.label() == tgt_tree.label():
            right += 1
        total += 1
        for id, tgt_sub_tree in enumerate(tgt_tree):
            if id > len(tree) - 1 or isinstance(tgt_sub_tree, str) or isinstance(tree[id], str):
                break
            sub_tree = tree[id]
            sub_right, sub_total = self.compare(sub_tree, tgt_sub_tree)
            right += sub_right
            total += sub_total
        return right, total

    def predict(self, src_seq, tgt_tree = None):
        """ Make prediction given `src_seq` as input.

        Args:
            src_seq (list): list of tokens in source language

        Returns:
            tgt_seq (list): list of tokens in target language as predicted
            by the pre-trained model
        """
        src_id_seq = Variable(torch.LongTensor([self.src_vocab.stoi[tok] for tok in src_seq]),
                              volatile=True).view(1, -1)
        if torch.cuda.is_available():
            src_id_seq = src_id_seq.cuda()
        if type(self.model) is not seq2seq.models.seq2seq.Seq2seq:
            tree, loss = self.model(src_id_seq, [len(src_seq)])
            # tree0 = deepcopy(tree)
            try:
                tree = self.tree_to_id(tree)
                return tree
            except:
                return ['a','b']
        else:
            softmax_list, _, other = self.model(src_id_seq, [len(src_seq)])
            length = other['length'][0]

            src_id_seq = [other['sequence'][di][0].data[0] for di in range(length)]
            src_seq = [self.src_vocab.itos[tok] for tok in src_id_seq]
            return src_seq
