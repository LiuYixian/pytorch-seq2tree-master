import logging
from nltk.tree import Tree
import torchtext

class SourceField(torchtext.data.Field):
    """ Wrapper class of torchtext.data.Field that forces batch_first and include_lengths to be True. """

    def __init__(self, **kwargs):
        logger = logging.getLogger(__name__)

        if kwargs.get('batch_first') is False:
            logger.warning("Option batch_first has to be set to use pytorch-seq2seq.  Changed to True.")
        kwargs['batch_first'] = True
        if kwargs.get('include_lengths') is False:
            logger.warning("Option include_lengths has to be set to use pytorch-seq2seq.  Changed to True.")
        kwargs['include_lengths'] = True
        kwargs['pad_token'] = None
        kwargs['unk_token'] = '<unk>'
        kwargs['init_token'] = '<sos>'
        kwargs['eos_token'] = None

        super(SourceField, self).__init__(**kwargs)

class CompField(torchtext.data.Field):
    """ Wrapper class of torchtext.data.Field that forces batch_first and include_lengths to be True. """

    def __init__(self, **kwargs):
        logger = logging.getLogger(__name__)

        if kwargs.get('batch_first') is False:
            logger.warning("Option batch_first has to be set to use pytorch-seq2seq.  Changed to True.")
        kwargs['batch_first'] = True
        if kwargs.get('include_lengths') is False:
            logger.warning("Option include_lengths has to be set to use pytorch-seq2seq.  Changed to True.")
        kwargs['include_lengths'] = True

        super(CompField, self).__init__(**kwargs)

class TreeField(torchtext.data.Field):
    """ Wrapper class of torchtext.data.Field that forces batch_first and include_lengths to be True. """

    def __init__(self, **kwargs):
        logger = logging.getLogger(__name__)

        kwargs['batch_first'] = False
        kwargs['include_lengths'] = False

        kwargs['sequential'] = False
        kwargs['use_vocab'] = False
        kwargs['tokenize'] = (lambda s: Tree.fromstring(s))
        kwargs['tensor_type'] = Tree.fromstring


        # sequential = True, use_vocab = True, init_token = None,
        # eos_token = None, fix_length = None, tensor_type = torch.LongTensor,
        # preprocessing = None, postprocessing = None, lower = False,
        # tokenize = (lambda s: s.split()), include_lengths = False,
        # batch_first = False, pad_token = "<pad>", unk_token = "<unk>"

        super(TreeField, self).__init__(**kwargs)

    def build_vocab(self, *args, **kwargs):
        #[a.label() for a in tree.subtrees()]
        super(TreeField, self).build_vocab(*args, **kwargs)
        a=1

class PosField(torchtext.data.Field):
    """ Wrapper class of torchtext.data.Field that forces batch_first and include_lengths to be True. """


    def __init__(self, **kwargs):
        logger = logging.getLogger(__name__)

        if kwargs.get('batch_first') is False:
            logger.warning("Option batch_first has to be set to use pytorch-seq2seq.  Changed to True.")
        kwargs['batch_first'] = True
        if kwargs.get('include_lengths') is False:
            logger.warning("Option include_lengths has to be set to use pytorch-seq2seq.  Changed to True.")
        kwargs['include_lengths'] = True
        kwargs['pad_token'] = None
        kwargs['unk_token'] = None
        kwargs['init_token'] = None
        kwargs['eos_token'] = None

        super(PosField, self).__init__(**kwargs)



class NTField(torchtext.data.Field):
    """ Wrapper class of torchtext.data.Field that forces batch_first and include_lengths to be True. """

    # SYM_SOS = '<sos>'
    # SYM_EOS = '<eos>'

    def __init__(self, **kwargs):
        logger = logging.getLogger(__name__)

        if kwargs.get('batch_first') is False:
            logger.warning("Option batch_first has to be set to use pytorch-seq2seq.  Changed to True.")
        kwargs['batch_first'] = True
        if kwargs.get('include_lengths') is False:
            logger.warning("Option include_lengths has to be set to use pytorch-seq2seq.  Changed to True.")
        kwargs['include_lengths'] = True
        kwargs['pad_token'] = None
        kwargs['unk_token'] = '<unk>'
        kwargs['init_token'] = '<sos>'
        kwargs['eos_token'] = '<eos>'
        super(NTField, self).__init__(**kwargs)


class TargetField(torchtext.data.Field):
    """ Wrapper class of torchtext.data.Field that forces batch_first to be True and prepend <sos> and append <eos> to sequences in preprocessing step.

    Attributes:
        sos_id: index of the start of sentence symbol
        eos_id: index of the end of sentence symbol
    """

    SYM_SOS = '<sos>'
    SYM_EOS = '<eos>'

    def __init__(self, **kwargs):
        logger = logging.getLogger(__name__)

        if kwargs.get('batch_first') == False:
            logger.warning("Option batch_first has to be set to use pytorch-seq2seq.  Changed to True.")
        kwargs['batch_first'] = True
        # if kwargs.get('preprocessing') is None:
        #     kwargs['preprocessing'] = lambda seq: [self.SYM_SOS] + seq + [self.SYM_EOS]
        # else:
        #     func = kwargs['preprocessing']
        #     kwargs['preprocessing'] = lambda seq: [self.SYM_SOS] + func(seq) + [self.SYM_EOS]

        kwargs['include_lengths'] = True
        kwargs['pad_token'] = None
        kwargs['unk_token'] = '<unk>'
        kwargs['init_token'] = None
        kwargs['eos_token'] = None
        super(TargetField, self).__init__(**kwargs)

