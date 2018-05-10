import torch.nn as nn
import torch.nn.functional as F

class Seq2tree(nn.Module):



    def __init__(self, encoder, decoder, decode_function=F.log_softmax):
        super(Seq2tree, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.decode_function = decode_function

    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def forward(self, input_variable, input_lengths=None, target_variable=None,
                target_comp = None, teacher_forcing_ratio=0, trees = None, loss = None):
        if trees is None:
            x=1
        encoder_outputs, encoder_hidden = self.encoder(input_variable, input_lengths)
        tree, loss = self.decoder(inputs=target_variable,
                              encoder_hidden=encoder_hidden,
                              encoder_outputs=encoder_outputs,
                              function=self.decode_function,
                              teacher_forcing_ratio=teacher_forcing_ratio, trees = trees, loss = loss)
        return tree, loss

    def evaluate(self, input_variable, input_lengths=None, target_variable=None,
                target_comp = None, teacher_forcing_ratio=0, trees = None, loss = None):
        if trees is None:
            x=1
        encoder_outputs, encoder_hidden = self.encoder(input_variable, input_lengths)
        tree, loss = self.decoder.evaluate(inputs=target_variable,
                              encoder_hidden=encoder_hidden,
                              encoder_outputs=encoder_outputs,
                              function=self.decode_function,
                              teacher_forcing_ratio=teacher_forcing_ratio, trees = trees, loss = loss)
        return tree, loss