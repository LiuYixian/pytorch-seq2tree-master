from __future__ import print_function, division

import torch
import torchtext

import seq2seq
from seq2seq.loss import NLLLoss

class Evaluator(object):
    """ Class to evaluate models with given datasets.

    Args:
        loss (seq2seq.loss, optional): loss for evaluator (default: seq2seq.loss.NLLLoss)
        batch_size (int, optional): batch size for evaluator (default: 64)
    """

    def __init__(self, loss=NLLLoss(), batch_size=64):
        self.loss = loss
        self.batch_size = batch_size

    def tree_acc(self, target, pre):
        target = [int(a) for a in target.leaves()]
        pre = [int(a) for a in pre.leaves()]
        right = 0
        for id,p in enumerate(target):
            if p==pre[id]:
                right+=1

        return right/len(target)

    def compare(self, tree, tgt_tree):
        right = 0
        total = 0
        if int(tree.label()) == int(tgt_tree.label()):
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

    def evaluate(self, model, data):
        """ Evaluate a model on given dataset and return performance.

        Args:
            model (seq2seq.models): model to evaluate
            data (seq2seq.dataset.dataset.Dataset): dataset to evaluate against

        Returns:
            loss (float): loss of the given model on the given dataset
        """
        model.eval()

        loss = self.loss
        loss.reset()
        match = 0
        total = 0

        device = None if torch.cuda.is_available() else -1
        batch_iterator = torchtext.data.BucketIterator(
            dataset=data, batch_size=self.batch_size,
            sort=True, sort_key=lambda x: len(x.src),
            device=device, train=False)
        src_vocab = data.fields[seq2seq.src_field_name].vocab
        pad = src_vocab.stoi[data.fields[seq2seq.src_field_name].pad_token]

        for batch in batch_iterator:
            input_variables, input_lengths  = getattr(batch, seq2seq.src_field_name)
            target_variables = getattr(batch, seq2seq.src_field_name)
            if type(model) is seq2seq.models.seq2seq.Seq2seq:
                decoder_outputs, decoder_hidden, other = model(input_variables, input_lengths.tolist(), target_variables)
                # Evaluation
                seqlist = other['sequence']
                for step, step_output in enumerate(decoder_outputs):
                    target = target_variables[:, step + 1]
                    loss.eval_batch(step_output.view(target_variables.size(0), -1), target)

                    non_padding = target.ne(pad)
                    correct = seqlist[step].view(-1).eq(target).masked_select(non_padding).sum().data[0]
                    match += correct
                    total += non_padding.sum().data[0]
            else:
                trees = getattr(batch, seq2seq.tree_field_name)
                tree, loss = model(input_variables, input_lengths.tolist(),
                                                               target_variables, trees = trees, loss = loss)
                right, total = self.compare(tree, trees)
                tree_acc = right / max(len([a for a in trees.subtrees()]), len([a for a in tree.subtrees()]))
                target = [int(a) for a in trees.leaves()]
                pre = [int(a) for a in tree.leaves()]
                for id, p in enumerate(target):
                    try:
                        if p == pre[id]:
                            match += 1
                    except:
                        x=1
                total += len(target)
                # return loss.get_loss(), self.tree_acc(trees, tree)



        if total == 0:
            accuracy = float('nan')
        else:
            accuracy = match / total

        return loss.get_loss(), accuracy, tree_acc
