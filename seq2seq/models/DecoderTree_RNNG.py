import random
#lyx
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from .attention import Attention
from .baseRNN import BaseRNN
from nltk.tree import Tree
from .Tree_with_para import Tree_with_para

MAX_LAYERS = 10
MAX_NT = 1000
if torch.cuda.is_available():
    import torch.cuda as device
else:
    import torch as device


class DecoderTree_RNNG(BaseRNN):
    r"""
    Provides functionality for decoding in a seq2seq framework, with an option for attention.

    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): a maximum allowed length for the sequence to be processed
        hidden_size (int): the number of features in the hidden state `h`
        sos_id (int): index of the start of sentence symbol
        eos_id (int): index of the end of sentence symbol
        n_layers (int, optional): number of recurrent layers (default: 1)
        rnn_cell (str, optional): type of RNN cell (default: gru)
        bidirectional (bool, optional): if the encoder is bidirectional (default False)
        input_dropout_p (float, optional): dropout probability for the input sequence (default: 0)
        dropout_p (float, optional): dropout probability for the output sequence (default: 0)
        use_attention(bool, optional): flag indication whether to use attention mechanism or not (default: false)

    Attributes:
        KEY_ATTN_SCORE (str): key used to indicate attention weights in `ret_dict`
        KEY_LENGTH (str): key used to indicate a list representing lengths of output sequences in `ret_dict`
        KEY_SEQUENCE (str): key used to indicate a list of sequences in `ret_dict`

    Inputs: inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio
        - **inputs** (batch, seq_len, input_size): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs.  It is used for teacher forcing when provided. (default `None`)
        - **encoder_hidden** (num_layers * num_directions, batch_size, hidden_size): tensor containing the features in the
          hidden state `h` of encoder. Used as the initial hidden state of the decoder. (default `None`)
        - **encoder_outputs** (batch, seq_len, hidden_size): tensor with containing the outputs of the encoder.
          Used for attention mechanism (default is `None`).
        - **function** (torch.nn.Module): A function used to generate symbols from RNN hidden state
          (default is `torch.nn.functional.log_softmax`).
        - **teacher_forcing_ratio** (float): The probability that teacher forcing will be used. A random number is
          drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
          teacher forcing would be used (default is 0).

    Outputs: decoder_outputs, decoder_hidden, ret_dict
        - **decoder_outputs** (seq_len, batch, vocab_size): list of tensors with size (batch_size, vocab_size) containing
          the outputs of the decoding function.
        - **decoder_hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the last hidden
          state of the decoder.
        - **ret_dict**: dictionary containing additional information as follows {*KEY_LENGTH* : list of integers
          representing lengths of output sequences, *KEY_SEQUENCE* : list of sequences, where each sequence is a list of
          predicted token IDs }.
    """

    KEY_ATTN_SCORE = 'attention_score'
    KEY_LENGTH = 'length'
    KEY_SEQUENCE = 'sequence'

    def __init__(self, vocab_size, nt_size, max_len, hidden_size,
            sos_id, eos_id,
            n_layers=1, rnn_cell='gru', bidirectional=False,
            input_dropout_p=0, dropout_p=0, use_attention=False, pos_in_nt = None):
        super(DecoderTree, self).__init__(vocab_size, max_len, hidden_size, input_dropout_p, dropout_p, n_layers,
                                          rnn_cell)

        self.bidirectional_encoder = bidirectional
        self.rnn = self.rnn_cell(hidden_size, hidden_size, n_layers, batch_first=True, dropout=dropout_p)
        self.layer_rnn = self.rnn_cell(hidden_size, hidden_size, n_layers, batch_first = True, dropout = dropout_p)
        self.layer_bi_rnn = self.rnn_cell(hidden_size, hidden_size, n_layers, batch_first=True, bidirectional=True, dropout=dropout_p)
        self.depth_rnn = self.rnn_cell(hidden_size*2, hidden_size*2, n_layers, batch_first = True, dropout = dropout_p)


        self.output_size = vocab_size
        self.nt_output_size = nt_size
        self.max_length = max_len
        self.use_attention = use_attention
        self.eos_id = eos_id
        self.sos_id = sos_id
        self.pos_in_nt = pos_in_nt
        self.init_input = None
        self.embedding_NT = nn.Embedding(self.nt_output_size, self.hidden_size)
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.feature2attention = nn.Linear(self.hidden_size, self.hidden_size)
        if use_attention:
            self.attention = Attention(self.hidden_size)
        self.NT_out = nn.Linear(self.hidden_size, self.nt_output_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward_step(self, input_var, hidden, encoder_outputs, function):
        batch_size = input_var.size(0)
        output_size = input_var.size(1)
        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)

        output, hidden = self.rnn(embedded, hidden)

        attn = None
        if self.use_attention:
            output, attn = self.attention(output, encoder_outputs)

        predicted_softmax = function(self.out(output.contiguous().view(-1, self.hidden_size))).view(batch_size, output_size, -1)
        return predicted_softmax, hidden, attn

    def generate_sub_tree(self, tgt_sub_tree, sub_root, encoder_outputs, top_structure_info, left_structure_info, loss):
        # left_structure_info = Variable(torch.zeros(1, 1, self.hidden_size))
        sampling = True
        if int(tgt_sub_tree.label()) in self.pos_in_nt:
            this_feature = self.feature2attention(sub_root.left_structure_info)
            if self.use_attention:
                this_feature, attn = self.attention(this_feature, encoder_outputs)
            this_word = function(self.out(this_feature).view(1, -1)).topk(1)[1].view(1, -1)
            tgt_word = Variable(torch.LongTensor([int(tgt_sub_tree[0])])).view(1, -1)
            next_word = this_word if sampling else tgt_word
            left_structure_info, _ = self.layer_rnn(left_structure_info,
                                                    self.embedding_NT(next_word))
            sub_root.append(next_word.data[0][0])
            loss.eval_batch(function(self.out(this_feature).view(1, -1)), tgt_word[0])
            
        for sub_tree in tgt_sub_tree:




    def forward(self, inputs=None, encoder_hidden=None, encoder_outputs=None,
                    function=F.log_softmax, teacher_forcing_ratio=0, trees = None, loss= None):

        decoder_hidden = self._init_state(encoder_hidden)

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        use_teacher_forcing = False
        # Manual unrolling is used to support random teacher forcing.
        # If teacher_forcing_ratio is True or False instead of a probability, the unrolling can be done in graph


        sampling = True
        top_structure_info = Variable(torch.zeros(1, 1, self.hidden_size * 2))
        left_structure_info = Variable(torch.zeros(1, 1, self.hidden_size))
        semantic_info = decoder_hidden
        this_feature = self.feature2attention(
            torch.cat((top_structure_info, left_structure_info, semantic_info), 2))
        if self.use_attention:
            this_feature, attn = self.attention(this_feature, encoder_outputs)
        this_NT = self.NT_out(this_feature).topk(1)[1].view(1, -1)
        tgt_NT = Variable(torch.LongTensor([int(trees.label())])).view(1, -1)
        next_NT = this_NT if sampling else tgt_NT
        loss.eval_batch(function(self.NT_out(this_feature).view(1, -1)), tgt_NT[0])
        pre_tree = Tree_with_para(next_NT.data[0][0], [],
                                  depth_feature=torch.cat([self.embedding_NT(next_NT), self.embedding_NT(next_NT)],
                                                          1).view(1, 1, -1),
                                  semantic_featrue=semantic_info, att=attn)
        tgt_tree = Tree_with_para(trees.label(), [],
                                  depth_feature=torch.cat([self.embedding_NT(tgt_NT), self.embedding_NT(tgt_NT)],
                                                          1).view(1, 1, -1),
                                  semantic_featrue=semantic_info, att=attn)
        sub_trees = [trees]  # sub_tree in obj layer
        pre_sub_trees = [pre_tree]  # sub_tree_with para in obj layer

        for i in range(trees.height() - 1):
            next_sub_trees = []  # sub_tree in next layer
            next_pre_trees = []  # sub_tree_with para in next layer
            left_structure_info = Variable(torch.zeros(1, 1, self.hidden_size))

            for id, obj_tree in enumerate(sub_trees):
                pre_obj_tree = pre_sub_trees[id]
                top_structure_info = pre_obj_tree.depth_feature
                semantic_info = pre_obj_tree.semantic_feature

                if int(obj_tree.label()) in self.pos_in_nt:
                    # generate from Pos_tag
                    this_feature = self.feature2attention(torch.cat((top_structure_info,
                                                                     left_structure_info, semantic_info), 2))
                    if self.use_attention:
                        this_feature, attn = self.attention(this_feature, encoder_outputs)
                    this_word = function(self.out(this_feature).view(1, -1)).topk(1)[1].view(1, -1)
                    tgt_word = Variable(torch.LongTensor([int(obj_tree[0])])).view(1, -1)
                    next_word = this_word if sampling else tgt_word
                    left_structure_info, _ = self.layer_rnn(left_structure_info,
                                                            self.embedding(next_word))
                    pre_obj_tree.append(next_word.data[0][0])
                    loss.eval_batch(function(self.out(this_feature).view(1, -1)), tgt_word[0])
                else:
                    for sub_id in range(len(obj_tree)):
                        # generate from nonterminal
                        this_feature = self.feature2attention(torch.cat((top_structure_info,
                                                                         left_structure_info, semantic_info), 2))
                        if self.use_attention:
                            this_feature, attn = self.attention(this_feature, encoder_outputs)
                        this_NT = function(self.NT_out(this_feature).view(1, -1)).topk(1)[1].view(1, -1)
                        target_NT = Variable(torch.LongTensor([int(obj_tree[sub_id].label())])).view(1, -1)
                        next_NT = this_NT if sampling else tgt_NT
                        loss.eval_batch(function(self.NT_out(this_feature).view(1, -1)), target_NT[0])

                        left_structure_info, _ = self.layer_rnn(left_structure_info,
                                                                self.embedding_NT(next_NT))
                        new_tree = Tree_with_para(next_NT.data[0][0], [], layer_feature=left_structure_info,
                                                  semantic_featrue=this_feature, parent=pre_obj_tree, att=attn)
                        pre_obj_tree.append(new_tree)
                        next_pre_trees.append(new_tree)
                        next_sub_trees.append(obj_tree[sub_id])
                        if next_NT.data[0, 0] == 2:
                            break
            if len(next_pre_trees) > 0:
                layer_feature = torch.cat([sub_tree.layer_feature for sub_tree in next_pre_trees], 1)
                layer_bi_feature = self.layer_bi_rnn(layer_feature)[0]
                for j, tree in enumerate(next_pre_trees):
                    tree.bi_layer_feature = layer_bi_feature[:, j, :].unsqueeze(1)
                    tree.depth_feature = self.depth_rnn(tree.parent.depth_feature, tree.bi_layer_feature)[0]
                sub_trees = next_sub_trees
                pre_sub_trees = next_pre_trees

        # schedule sampling

        ret_dict = pre_tree
        return pre_tree, loss

    def evaluate(self, inputs=None, encoder_hidden=None, encoder_outputs=None,
                    function=F.log_softmax, teacher_forcing_ratio=0, trees = None, loss= None):

        decoder_hidden = self._init_state(encoder_hidden)
        sampling = True

        top_structure_info = Variable(torch.zeros(1, 1, self.hidden_size * 2))
        left_structure_info = Variable(torch.zeros(1, 1, self.hidden_size))
        semantic_info = decoder_hidden
        this_feature = self.feature2attention(
            torch.cat((top_structure_info, left_structure_info, semantic_info), 2))
        if self.use_attention:
            this_feature, attn = self.attention(this_feature, encoder_outputs)
        this_NT = self.NT_out(this_feature).topk(1)[1].view(1, -1)
        next_NT = this_NT
        pre_tree = Tree_with_para(next_NT.data[0][0], [],
                                  depth_feature=torch.cat([self.embedding_NT(next_NT), self.embedding_NT(next_NT)],
                                                          1).view(1, 1, -1),
                                  semantic_featrue=semantic_info, att=attn)
        pre_sub_trees = [pre_tree]  # sub_tree_with para in obj layer

        for i in range(30):
            next_pre_trees = []  # sub_tree_with para in next layer
            left_structure_info = Variable(torch.zeros(1, 1, self.hidden_size))

            for id, pre_obj_tree in enumerate(pre_sub_trees):
                top_structure_info = pre_obj_tree.depth_feature
                semantic_info = pre_obj_tree.semantic_feature
                if pre_obj_tree.label()==2:
                    continue
                if int(pre_obj_tree.label()) in self.pos_in_nt:
                    # generate from Pos_tag
                    this_feature = self.feature2attention(torch.cat((top_structure_info,
                                                                     left_structure_info, semantic_info), 2))
                    if self.use_attention:
                        this_feature, attn = self.attention(this_feature, encoder_outputs)
                    this_word = function(self.out(this_feature).view(1, -1)).topk(1)[1].view(1, -1)
                    next_word = this_word
                    left_structure_info, _ = self.layer_rnn(left_structure_info,
                                                            self.embedding_NT(next_NT))
                    pre_obj_tree.append(next_word.data[0][0])
                else:
                    for sub_id in range(10):
                        # generate from nonterminal
                        this_feature = self.feature2attention(torch.cat((top_structure_info,
                                                                         left_structure_info, semantic_info), 2))
                        if self.use_attention:
                            this_feature, attn = self.attention(this_feature, encoder_outputs)
                        this_NT = function(self.NT_out(this_feature).view(1, -1)).topk(1)[1].view(1, -1)
                        next_NT = this_NT

                        left_structure_info, _ = self.layer_rnn(left_structure_info,
                                                                self.embedding_NT(next_NT))
                        new_tree = Tree_with_para(next_NT.data[0][0], [], layer_feature=left_structure_info,
                                                  semantic_featrue=this_feature, parent=pre_obj_tree, att=attn)
                        pre_obj_tree.append(new_tree)
                        next_pre_trees.append(new_tree)
                        if next_NT.data[0, 0] == 2:
                            break
            if len(next_pre_trees) > 0:
                layer_feature = torch.cat([sub_tree.layer_feature for sub_tree in next_pre_trees], 1)
                layer_bi_feature = self.layer_bi_rnn(layer_feature)[0]
                for j, tree in enumerate(next_pre_trees):
                    tree.bi_layer_feature = layer_bi_feature[:, j, :].unsqueeze(1)
                    tree.depth_feature = self.depth_rnn(tree.parent.depth_feature, tree.bi_layer_feature)[0]
            pre_sub_trees = next_pre_trees
            # else:
            #     break
        # schedule sampling

        ret_dict = pre_tree
        return pre_tree, loss

    def _init_state(self, encoder_hidden):
        """ Initialize the encoder hidden state. """
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden)
        return encoder_hidden

    def _cat_directions(self, h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
        """
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def _validate_args(self, inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio):
        if self.use_attention:
            if encoder_outputs is None:
                raise ValueError("Argument encoder_outputs cannot be None when attention is used.")

        # inference batch size
        if inputs is None and encoder_hidden is None:
            batch_size = 1
        else:
            if inputs is not None:
                batch_size = inputs.size(0)
            else:
                if self.rnn_cell is nn.LSTM:
                    batch_size = encoder_hidden[0].size(1)
                elif self.rnn_cell is nn.GRU:
                    batch_size = encoder_hidden.size(1)

        # set default input and max decoding length
        if inputs is None:
            if teacher_forcing_ratio > 0:
                raise ValueError("Teacher forcing has to be disabled (set 0) when no inputs is provided.")
            inputs = Variable(torch.LongTensor([self.sos_id] * batch_size),
                                    volatile=True).view(batch_size, 1)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            max_length = self.max_length
        else:
            max_length = inputs.size(1) - 1 # minus the start of sequence symbol

        return inputs, batch_size, max_length
