#!/usr/bin/python
# -*- coding: utf-8
from igraph import *
from math import *
import numpy as np
from functools32 import lru_cache


onto = Graph.Read_Ncol('ontologie.txt')
for n in onto.vs() :
    print(n)

#----- Context Function -----#

"""
   :param a: Index of centrality (mean) -- Int
   :param b: Exposant -- Float
   :param c: Dispersion (sd) -- Float
   :param x: Variable -- Float
"""

def id(a, x) :
    return 1

def local(a, x) :
    if(x == a) :
        return 1
    else :
        return 0

def gaussian(mu, x, sigma=1.0) :
    return exp(-1.0/2.0 * ((x-mu)/(sigma*1.0))**2)

def psi(a, x, b=2, c=1.0) :
    return 1.0/(b**(abs((x-a)/c)))

#----------#

def context_vector(f,k,S,op) :
    """
    :param f: Context function -- (Int, Int) -> Float
    :param k: Index in sequence  -- Int
    :param S: Sequence -- String
    :param op: Edit Operation
    :return: Context vector -- Float []
    """
    if(op == "mod") :
        return map(lambda x : f(k, x), range(len(S)))
    else :
        return map(lambda x : f(k, x+1), range(k-1))+\
               [1.0,1.0]+\
               map(lambda x : f(k, x), range(k+1, len(S)))

def sim(x,y) :
    return 1 - float(abs(ord(x) - ord(y))) / (ord('z') - ord('a'))

def cost_edit(S, x, f, k, op, G = onto, alpha = 0.25) :
    """
    :param S: Sequence -- String
    :param x: Symbol to edit -- Char
    :param f: Context function
    :param k: Index in sequence  -- Int
    :param op: Edit Operation -- Boolean
    :param alpha: Parameter -- Float [0,1]
    :return: Agregation of scores vector -- Float
    """
    sim_v = map(lambda s : sim_Wu_Palmer(s, x, G), S) # Vecteur de similarité
    ctx_v = context_vector(f, k, S, op) # Vecteur contextuel
    simWeigth_v = []
    for i in range(len(sim_v)) :
        simWeigth_v += [sim_v[i] * ctx_v[i]]
    if (op == "mod"): # Si alpha != 0, fonction delta
        delta_local_cost = 1 - sim_Wu_Palmer(S[k], x, G) # 0 if S[k] == x else 1
    else:
        delta_local_cost = 1
    return (alpha * delta_local_cost) + (1 - alpha) * (1 - max(simWeigth_v))


#--------------#

"""
def ced(S, _S, i, j, f = phi) :
    if (i == 0 and j == 0) :
        return 0
    if (j == 0) :
        return ced(S, _S, i - 1, j) + cost_edit(_S, S[i - 1], f, 0, False, max) #Op_ins_del(S2, S1[i - 1], i-1)
    if (i == 0) :
        return ced(S, _S, i, j - 1) + cost_edit(S, _S[j - 1], f, 0, False, max) #Op_ins_del(S1, S2[j - 1], j-2)

    return \
        min(ced(S, _S, i - 1, j - 1) + cost_edit(_S, S[i - 1], f, j - 1, True, max), #1 - sim(S1[i - 1], S2[j - 1]),  Mod
        min(ced(S, _S, i - 1, j) + cost_edit(_S, S[i - 1], f, j - 2, False, max),  # ,  Sup
                ced(S, _S, i, j - 1) + cost_edit(S, _S[j - 1], f, i - 2, False, max)  #Op_ins_del(S1, S2[j - 1], j-2)  Add
            ))
"""

def cedDP(S, _S, f = gaussian) :
    D = np.zeros((len(S)+1, len(_S)+1))
    for i in range (len(S)+1) :
        for j in range (len(_S)+1) :
            if (i==0 and j == 0) :
                D[i,j] = 0
            else :
                if (i==0) :
                    D[i, j] = D[i, j-1] + cost_edit(S, _S[j - 1], f, 0, "del")
                if (j==0) :
                    D[i, j] = D[i-1, j] +cost_edit(_S, S[i - 1], f, 0, "add")
                if(i != 0 and j != 0) :
                    cost_mod = cost_edit(_S, S[i - 1], f, j - 1, "mod")
                    cost_add = cost_edit(S, _S[j - 1], f, i - 2, "del")
                    cost_del = cost_edit(_S, S[i - 1], f, j - 2, "add")
                    """
                    print(i,j)
                    print("mod : "),
                    print(cost_mod)
                    print("add : "),
                    print(cost_add)
                    print("sup : "),
                    print(cost_del)"""
                    D[i,j ]	=	min(D[i-1,	j-1] + cost_mod,
                                D[i-1,	j] + cost_del ,
                                D[i,	j-1] + cost_add )
    #print D
    return D[len(S),len(_S)]

#--- Calcul des similarités ---#


def LCA(G, nodeA, nodeB) :
    """
    Lowest Common Ancestor
    :param G: Ontologie
    :param nodeA
    :param nodeB
    :return:
    """
    if(nodeA == nodeB) :
        return [[nodeA]]
    if (nodeA == G.vs()[0] or nodeB == G.vs()[0]):
        return [[G.vs()[0]]]
    else :
        nodeAParents = set([nodeA.index])
        nodeBParents = set([])
        # All Predecessors of A
        new_nodeParents = set(G.predecessors(nodeA))
        while(len(new_nodeParents) != 0) :
            nodeAParents |= new_nodeParents
            temp = set([])
            for n in new_nodeParents :
                temp |= set(G.predecessors(n))
            new_nodeParents = temp
        # First Ancestor of B and A
        new_nodeParents = set([nodeB.index])
        while(len(new_nodeParents & nodeAParents) < 1) :
            nodeBParents |= new_nodeParents
            temp = set([])
            for n in new_nodeParents:
                temp |= set(G.predecessors(n))
            new_nodeParents = temp
        return list(new_nodeParents & nodeAParents)

def max_prof(G) :
    max_shortest_path = 0
    for i in range(0, len(G.vs)) :
        for j in range (i+1, len(G.vs)) :
            max_shortest_path = max(max_shortest_path,
                                G.as_undirected().shortest_paths_dijkstra(source=G.vs[i], target=G.vs[j],
                                                                              weights=None))
    return max_shortest_path

##### Similarity measure #####

def sim_Wu_Palmer(x, y, G) :
    prof_comm = G.as_undirected().shortest_paths_dijkstra(source=G.vs[0], target=LCA(G, x, y)[0], weights=None)[0][0]
    prof_x = G.as_undirected().shortest_paths_dijkstra(source=G.vs[0], target=x, weights=None)[0][0]
    prof_y = G.as_undirected().shortest_paths_dijkstra(source=G.vs[0], target=y, weights=None)[0][0]
    if (x == y):
        return 1
    if(prof_comm == 0) :
        return 0
    else :
        return (2.0 * int(prof_comm)) / (prof_x + prof_y)


def sim_leacock_chodrow(x, y, G) :
    prof_max = max_prof(G)[0][0]
    dist_xy = G.as_undirected().shortest_paths_dijkstra(source=x, target=y, weights=None)[0][0]
    return -log((prof_max + dist_xy)/(2.0*prof_max), 2)

onto.edge_arrow_size = 0
onto.edge_arrow_width = 0
onto.vs["label"] = onto.vs["name"]
onto.vs['color'] = ["white"]
Graph.write_svg(onto, 'onto.svg', layout=onto.layout('rt',root=[0]), width= 500, heigth = 300)
#layout = onto.layout("kk")
plot(onto, layout=onto.layout('rt',root=[0]), bbox=(500,350))

sim_matrix = np.ones((len(onto.vs), len(onto.vs)))

for i in range(0, len(onto.vs)):
    for j in range(0, len(onto.vs)):
        sim_matrix[i,j] = round(sim_Wu_Palmer(onto.vs()[i], onto.vs()[j], onto), 2)

#print (sim_matrix)

S1 = [onto.vs()[17], onto.vs()[14]]
S2 = [onto.vs()[13], onto.vs()[14]]
S3 = [onto.vs()[15], onto.vs()[7]]
S4 = [onto.vs()[15], onto.vs()[8]]
S5 = [onto.vs()[15], onto.vs()[9]]
S6 = [onto.vs()[7], onto.vs()[16], onto.vs()[10]]
S7 = [onto.vs()[7], onto.vs()[13], onto.vs()[10]]
S8 = [onto.vs()[7], onto.vs()[16], onto.vs()[7], onto.vs()[17], onto.vs()[10]]
S9 = [onto.vs()[7], onto.vs()[13], onto.vs()[7], onto.vs()[17], onto.vs()[10]]
S10 = [onto.vs()[8]]
S11 = [onto.vs()[8], onto.vs()[11], onto.vs()[8]]
S12 = [onto.vs()[7], onto.vs()[17], onto.vs()[7], onto.vs()[13], onto.vs()[10]]
S13 = [onto.vs()[10], onto.vs()[16], onto.vs()[7], onto.vs()[17], onto.vs()[7]]
S14 = [onto.vs()[8], onto.vs()[12], onto.vs()[8]]
S15 = [onto.vs()[7], onto.vs()[9], onto.vs()[14], onto.vs()[8], onto.vs()[12]]



Sequences = [S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12, S13, S14, S15]


print(cost_edit(Sequences[12], onto.vs()[7], id, 0, "add"))
print(cedDP(Sequences[0], Sequences[12]))
print(cedDP(Sequences[12], Sequences[0]))


CED_matrix = np.zeros((len(Sequences), len(Sequences)))
for i in range(len(Sequences)) :
    for j in range(len(Sequences)):
        S = Sequences[i]
        _S = Sequences[j]
        #CED_matrix[i,j] = round(2.0*max(cedDP(S, _S), cedDP(_S, S)) / (len(S) + len(_S)), 2)


np.savetxt("dist_editDist_weight.csv", CED_matrix, delimiter=",")
#print(CED_matrix)
#print(distance/25)
#print(dtw_matrix)