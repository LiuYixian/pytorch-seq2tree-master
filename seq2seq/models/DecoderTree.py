import random

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


class DecoderTree(BaseRNN):
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

    def __init__(self, vocab_size, word_embedding_size, nt_embedding_size, nt_size, max_len, hidden_size,
            sos_id=None, eos_id=None,
            n_layers=1, rnn_cell='gru', bidirectional=False,
            input_dropout_p=0, dropout_p=0, use_attention=False, pos_in_nt = None):
        super(DecoderTree, self).__init__(vocab_size, max_len, hidden_size, input_dropout_p, dropout_p, n_layers,
                                          rnn_cell)

        self.bidirectional_encoder = bidirectional
        self.word_embedding_size = word_embedding_size
        self.nt_embedding_size = nt_embedding_size
        self.output_size = vocab_size
        self.nt_output_size = nt_size
        self.max_length = max_len
        self.use_attention = use_attention
        self.eos_id = eos_id
        self.sos_id = sos_id
        self.pos_in_nt = pos_in_nt
        self.init_input = None
        self.layer_rnn = self.rnn_cell(self.nt_embedding_size, hidden_size, n_layers, batch_first = True, dropout = dropout_p)
        self.layer_bi_rnn = self.rnn_cell(self.nt_embedding_size, hidden_size, n_layers, batch_first=True, bidirectional=True, dropout=dropout_p)
        self.depth_rnn = self.rnn_cell(hidden_size*2, hidden_size*2, n_layers, batch_first = True, dropout = dropout_p)
        self.update_vec = nn.Linear(self.hidden_size+ self.nt_embedding_size, self.hidden_size )
        self.embedding_NT = nn.Embedding(self.nt_output_size, self.nt_embedding_size)
        self.embedding = nn.Embedding(self.output_size, self.word_embedding_size)
        self.feature2attention = nn.Linear(self.hidden_size*4, self.hidden_size)
        if use_attention:
            self.attention = Attention(self.hidden_size)
        self.NT_out = nn.Linear(self.hidden_size, self.nt_output_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, inputs=None, encoder_hidden=None, encoder_outputs=None,
                    function=F.log_softmax, teacher_forcing_ratio=0, trees = None, loss= None):

        decoder_hidden = self._init_state(encoder_hidden)
        sampling = True if random.random() > teacher_forcing_ratio else False
        top_structure_info = Variable(torch.zeros(1, 1, self.hidden_size * 2))
        left_structure_info = Variable(torch.zeros(1, 1, self.hidden_size))
        if torch.cuda.is_available():
            top_structure_info = top_structure_info.cuda()
            left_structure_info = left_structure_info.cuda()
        semantic_info = decoder_hidden
        this_feature = self.feature2attention(
            torch.cat((top_structure_info, left_structure_info, semantic_info), 2))
        if self.use_attention:
            this_feature, attn = self.attention(this_feature, encoder_outputs)
        this_NT = self.NT_out(this_feature).topk(1)[1].view(1, -1)
        next_NT = this_NT
        if self.training:
            tgt_NT = Variable(torch.LongTensor([int(trees.label())])).view(1, -1)
            if torch.cuda.is_available():
                tgt_NT = tgt_NT.cuda()
            next_NT = this_NT if random.random() > teacher_forcing_ratio else tgt_NT
            loss.eval_batch(function(self.NT_out(this_feature).view(1, -1)), tgt_NT[0])
        pre_tree = Tree_with_para(next_NT.data[0][0], [],
                                  depth_feature=torch.cat([self.embedding_NT(next_NT), self.embedding_NT(next_NT),self.embedding_NT(next_NT), self.embedding_NT(next_NT)],
                                                          1).view(1, 1, -1),
                                  semantic_feature=semantic_info, att=attn)
        sub_trees = [trees]  # sub_tree in o bj layer
        pre_sub_trees = [pre_tree]  # sub_tree_with para in obj layer

        for i in range(10):#range(trees.height() - 1):
            # generate every layer
            next_sub_trees = []  # sub_tree in next layer
            next_pre_trees = []  # sub_tree_with para in next layer
            left_structure_info = Variable(torch.zeros(1, 1, self.hidden_size))
            if torch.cuda.is_available():
                left_structure_info = left_structure_info.cuda()
            for id, pre_obj_tree in enumerate(pre_sub_trees):
            # for id, obj_tree in enumerate(sub_trees):
            #     pre_obj_tree = pre_sub_trees[id]
                if self.training:
                    if id > len(sub_trees)-1: break
                    obj_tree = sub_trees[id]
                top_structure_info = pre_obj_tree.depth_feature
                semantic_info = pre_obj_tree.semantic_feature

                if int(pre_obj_tree.label()) in self.pos_in_nt:
                    if self.training and int(obj_tree.label()) not in self.pos_in_nt:
                        continue
                    # generate from Pos_tag
                    # this_feature = self.feature2attention(torch.cat((top_structure_info,
                    #                                                  left_structure_info, semantic_info), 2))
                    this_feature = pre_obj_tree.semantic_feature
                    if self.use_attention:
                        this_feature, attn = self.attention(this_feature, encoder_outputs)
                    this_word = function(self.out(this_feature).view(1, -1)).topk(1)[1].view(1, -1)
                    next_word = this_word
                    if self.training:
                        tgt_word = Variable(torch.LongTensor([int(obj_tree[0])])).view(1, -1)
                        if torch.cuda.is_available():
                            tgt_word = tgt_word.cuda()
                        next_word = this_word if random.random() > teacher_forcing_ratio else tgt_word
                        loss.eval_batch(function(self.out(this_feature).view(1, -1)), tgt_word[0])
                    # left_structure_info, _ = self.layer_rnn(left_structure_info,
                    #                                         self.embedding(next_word))
                    pre_obj_tree.append(next_word.data[0][0])

                else:
                    if self.training and int(obj_tree.label()) in self.pos_in_nt:
                        continue
                    # for sub_id in range(len(obj_tree)):
                    for sub_id in range(10):

                        # generate from nonterminal
                        this_feature = self.feature2attention(torch.cat((top_structure_info,
                                                                         left_structure_info, semantic_info), 2))
                        if self.use_attention:
                            this_feature, attn = self.attention(this_feature, encoder_outputs)
                        this_NT = function(self.NT_out(this_feature).view(1, -1)).topk(1)[1].view(1, -1)
                        next_NT = this_NT
                        if self.training:
                            if sub_id > len(obj_tree) - 1:
                                break # next sub_tree in this layer
                            tgt_NT = Variable(torch.LongTensor([int(obj_tree[sub_id].label())])).view(1, -1)
                            if torch.cuda.is_available():
                                tgt_NT = tgt_NT.cuda()
                            next_NT = this_NT if random.random() > teacher_forcing_ratio else tgt_NT
                            loss.eval_batch(function(self.NT_out(this_feature).view(1, -1)), tgt_NT[0])


                        left_structure_info, _ = self.layer_rnn(self.embedding_NT(next_NT),left_structure_info)
                        semantic = self.update_vec(torch.cat([this_feature, self.embedding_NT(next_NT)],2))
                        new_tree = Tree_with_para(next_NT.data[0][0], [], layer_feature=left_structure_info,
                                                  semantic_feature=semantic, parent=pre_obj_tree, att=attn)
                        pre_obj_tree.append(new_tree) # grow new sub_tree
                        if next_NT.data[0, 0] == 2:
                            break
                        if self.training:
                            next_sub_trees.append(obj_tree[sub_id])
                        next_pre_trees.append(new_tree) # sub_trees in next layer

            if len(next_pre_trees) > 0 and (not self.training or len(next_sub_trees)>0): #
                # if training both pre_tree and tgt_tree have next layer
                # if eval, pre_tree has next layer

                if torch.cuda.is_available():
                    layer_feature = torch.cat(
                        [self.embedding_NT(Variable(torch.LongTensor([sub_tree.label()]).cuda())).view(1, 1, -1) for sub_tree
                         in next_pre_trees], 1)
                else:
                    layer_feature = torch.cat(
                        [self.embedding_NT(Variable(torch.LongTensor([sub_tree.label()]))).view(1, 1, -1) for sub_tree
                         in next_pre_trees], 1)

                layer_bi_feature = self.layer_bi_rnn(layer_feature)[0]
                for j, tree in enumerate(next_pre_trees):
                    tree.bi_layer_feature = layer_bi_feature[:, j, :].unsqueeze(1)
                    tree.depth_feature = self.depth_rnn(tree.parent.depth_feature, tree.bi_layer_feature)[0]
                sub_trees = next_sub_trees
                pre_sub_trees = next_pre_trees
            else:# end grow
                break
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
