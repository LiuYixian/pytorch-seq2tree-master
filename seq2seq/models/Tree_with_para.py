from nltk.tree import Tree
# import torch
# from torch.autograd.variable import Variable


class Tree_with_para(Tree):

    def __init__(self, label, sub_tree, layer_feature=None, bi_layer_feature=None, depth_feature=None,
                 semantic_feature=None, parent=None, att = None):
        super(Tree_with_para, self).__init__(label, sub_tree)
        self.layer_feature = layer_feature
        self.bi_layer_feature = bi_layer_feature
        self.depth_feature = depth_feature
        self.semantic_feature = semantic_feature
        self.parent = parent
        self.att = att
        self.target_word = None