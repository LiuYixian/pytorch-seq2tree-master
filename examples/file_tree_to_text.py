import nltk
from nltk.tree import Tree
path = '../wsj_dataset/WSJ_s23_tree'
t_path = 'WSJ_s23_post.txt'

def to_binary(node):
    if isinstance(node[0], str):
        return node[0]
    if len(node) == 1:
        return to_binary(node[0])
    if len(node) > 2:
        new_tree = Tree('',[node[0]])
        for i in range(len(node) - 1):
            new_tree = Tree('', [new_tree, node[i + 1]])
        return to_binary(new_tree)
    return Tree('', [to_binary(node[0]), to_binary(node[1])])
def strtree_to_comid(text):
    word_id = 0
    id = []
    com_num = 0
    text = text.split()
    for x in text:
        if x[-1] == ')':
            word_id += 1
            m = x
            while m[-1] == ')':
                id.append(word_id - com_num - 2)
                com_num += 1
                m=m[:-1]

        elif x != '(':
            word_id +=1
    return id


with open(path, 'r') as reader:
    text = []
    comids = []
    trees = []
    NTs = []
    Poss = []
    for line in reader:
        tree = Tree.fromstring(line)
        words = tree.leaves()
        NT = [a.label() for a in tree.subtrees()]
        Pos = [a.label() for a in tree.subtrees() if len(a)==1 and isinstance(a[0], str)]
        comid = strtree_to_comid(str(to_binary(tree)).replace('\n', ''))
        text.append(words)
        NTs.append(NT)
        Poss.append(Pos)
        comids.append(' '.join([str(a) for a in comid]))
        trees.append(line)
x=1

with open(t_path, 'w', encoding='utf-8') as reader:
    for i in range(len(text)):
        reader.write(' '.join(text[i]))
        reader.write('\t')
        reader.write(' '.join(NTs[i]))
        reader.write('\t')
        reader.write(' '.join(Poss[i]))
        reader.write('\t')
        reader.write(trees[i])