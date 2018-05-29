from nltk.tree import Tree
# import torch
# from torch.autograd.variable import Variable


class Tree_with_para(Tree):
    pos_in_nt = None
    eos = None
    sos = None
    def __init__(self, label, sub_tree, layer_feature=None, bi_layer_feature=None, depth_feature=None,
                 semantic_feature=None, parent=None, att = None, target_label = None, pre_label = None):
        super(Tree_with_para, self).__init__(label, sub_tree)
        self.layer_feature = layer_feature
        self.bi_layer_feature = bi_layer_feature
        self.depth_feature = depth_feature
        self.semantic_feature = semantic_feature
        self.parent = parent
        self.att = att
        self.target_word = None
        self.pre_label = None
        self.target_label = target_label
        self.pre_label = pre_label
        self.is_leaf = False
        if target_label is not None:
            self.end_subtree = (target_label == self.eos)
            self.is_leaf = (target_label in self.pos_in_nt)
        else:
            self.end_subtree = (pre_label == self.eos)
            self.is_leaf = (pre_label in self.pos_in_nt)

    def add_child(self, child):
        self.append(child)

    def get_parent(self):
        return self.parent

    def update_node(self, target_word):
        if self.is_leaf:
            self.target_word = target_word
        else:
            raise "This node is not a leaf node"





if __name__ == "__main__":
    a = Tree_with_para('s',[])
    a.add_child(Tree_with_para('s1',[]))
    a.add_child(Tree_with_para('s2',[]))
    x=1