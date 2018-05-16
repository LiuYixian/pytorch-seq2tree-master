from nltk.tree import Tree
import numpy as np
from copy import deepcopy
import random
nonterminals = ['S', 'NPa', 'NPp', 'ViADV', 'VPa', 'VsTS', 'Vi', 'ADV', 'Vs', 'TS', 'T', 'S', 'Vt', 'VtNP', 'PNPp', 'P']
NT2id = dict(zip(nonterminals,range(len(nonterminals))))
id2NT = dict(zip(range(len(nonterminals)),nonterminals))

words = 'ugly Mary the_city the_shop' \
    ' from John appears misses looks seems' \
    ' says sees is with tells beautiful hopes' \
    ' in knows the_car that near thinks the_man' \
    ' likes the_house the_child small large'.split()
word2id = dict(zip(words,range(len(words))))
id2word = dict(zip(range(len(words)),words))
# construct nonterminal rule
with open('emile.txt', 'r') as f:
    head = None
    child = []
    NT_rule = dict()
    T_rule = dict()
    for line in f.readlines():
        if '.' not in line:
            if '1' in line:
                if '->' in line:
                    if head is not None:
                        NT_rule[head] = child

                    head = NT2id[line.replace('(', '').replace(')', '').split()[0]]
                    child = []
                    child.append([NT2id[line.replace('(','').replace(')','').split()[3]],
                                  NT2id[line.replace('(','').replace(')','').split()[4]]])
                else:
                    child.append([NT2id[line.replace('(', '').replace(')', '').split()[1]],
                                  NT2id[line.replace('(', '').replace(')', '').split()[2]]])
        else:
            if '1' in line:
                if '->' in line:
                    if head is not None:
                        if type(child[0])==list:
                            NT_rule[head] = child
                        else:
                            T_rule[head] = child

                    head = NT2id[line.replace('(', '').replace(')', '').split()[0]]
                    child = []
                    child.append(word2id[line.replace('(', '').replace(')', '').split()[4]])
                else:
                    child.append(word2id[line.replace('(', '').replace(')', '').split()[2]])
    T_rule[head] = child

## generation
root=Tree('S',[])
test_tree = Tree('S',[Tree('NPa',[]), Tree('ViADV',[])])
trees = [root]
def rewrite_based_on_a_tree(tree):
    new_trees = []
    for sub_tree in tree.subtrees():
        if len(sub_tree)==0:
            this_label = sub_tree.label()
            if NT2id[this_label] in NT_rule.keys():
                childs = NT_rule[NT2id[this_label]]
                if this_label == 'S':# and len(tree) > 0:
                    childs = childs[0:-1]
                for child in childs:
                    new_tree = deepcopy(tree)
                    target_sub_tree = [a for a in new_tree.subtrees() if len(a)==0][0]
                    for id in child:
                        target_sub_tree.append(Tree(id2NT[id], []))
                    new_trees.append(new_tree)

            else:
                childs = T_rule[NT2id[this_label]]
                for child in childs:
                    new_tree = deepcopy(tree)
                    target_sub_tree = [a for a in new_tree.subtrees() if len(a) == 0][0]
                    target_sub_tree.append(id2word[child])
                    new_trees.append(new_tree)
            return new_trees
    return False


b = rewrite_based_on_a_tree(test_tree)

def ge_all_sub_from_this_tree(root_tree):
    trees = [root_tree]
    while 1:
        make_new = False
        for id, tree in enumerate(trees):
            result = rewrite_based_on_a_tree(tree)
            if result is not False:
                trees.pop(id)
                trees = trees + result
                # trees.sort()
                make_new = True
                break
        if make_new is False:
            break
    return trees



trees = ge_all_sub_from_this_tree(root)
random.shuffle(trees)
train = trees[0:1500]
test = trees[1500:2000]
valid = trees[2000:]
# sub = ge_all_sub_from_this_tree(root)
# sub2 = ge_all_sub_from_this_tree(Tree('P',[]))
#
#
#
#
#
# def generate_S():
#     ini=[]
#     ini.append(NT2id['S'])
#     targ = ''
#     while len(ini)>0:
#         NT = ini.pop(0)
#         if NT == ')':
#             targ += ')'
#             continue
#         targ += ' ('
#         targ += id2NT[NT]
#         targ += ' '
#         if NT in NT_rule.keys():
#             child = NT_rule[NT]
#         else:
#             child = T_rule[NT]
#         index = int(len(child) * np.random.rand())
#         if type(child[index]) == list:
#             ini = child[index] + ini
#             ini.append(')')
#         else:
#             targ += id2word[child[0]]
#             targ += ')'
#     return targ
#
# a= generate_S()

with open('toy_train.txt', 'w', encoding='utf-8') as f:
    for sentence in train:
        text = sentence.leaves()
        text = ' '.join(text)
        NT = [a.label() for a in sentence.subtrees()]
        NT = ' '.join(NT)
        Pos = [a.label() for a in sentence.subtrees() if len(a) == 1 and isinstance(a[0], str)]
        Pos = ' '.join(Pos)
        tree_str = str(sentence).replace('\n', '')
        f.write(text)
        f.write('\t')
        f.write(NT)
        f.write('\t')
        f.write(Pos)
        f.write('\t')
        f.write(tree_str)
        f.write('\n')

with open('toy_test.txt', 'w', encoding='utf-8') as f:
    for sentence in test:
        text = sentence.leaves()
        text = ' '.join(text)
        NT = [a.label() for a in sentence.subtrees()]
        NT = ' '.join(NT)
        Pos = [a.label() for a in sentence.subtrees() if len(a) == 1 and isinstance(a[0], str)]
        Pos = ' '.join(Pos)
        tree_str = str(sentence).replace('\n', '')
        f.write(text)
        f.write('\t')
        f.write(NT)
        f.write('\t')
        f.write(Pos)
        f.write('\t')
        f.write(tree_str)
        f.write('\n')

with open('toy_dev.txt', 'w', encoding='utf-8') as f:
    for sentence in valid:
        text = sentence.leaves()
        text = ' '.join(text)
        NT = [a.label() for a in sentence.subtrees()]
        NT = ' '.join(NT)
        Pos = [a.label() for a in sentence.subtrees() if len(a) == 1 and isinstance(a[0], str)]
        Pos = ' '.join(Pos)
        tree_str = str(sentence).replace('\n', '')
        f.write(text)
        f.write('\t')
        f.write(NT)
        f.write('\t')
        f.write(Pos)
        f.write('\t')
        f.write(tree_str)
        f.write('\n')
# with open('toy_100.txt', 'w') as f:
#     for i in range(100):
#         sentence = generate_S()+'\r'
#         f.write(sentence)
#
# with open('toy_50.txt', 'w') as f:
#     for i in range(50):
#         sentence = generate_S()+'\r'
#         f.write(sentence)