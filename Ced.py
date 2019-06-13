#!/usr/bin/python
# -*- coding: utf-8
import numpy as np
from Operator import *
from Ontology import *
from Contextual_function import *

def cedDP(S1, S2, f, g) :
    D = np.zeros((len(S1) + 1, len(S2) + 1))
    for i in range(len(S1) + 1):
        for j in range(len(S2) + 1):
            if (i == 0 and j == 0):
                D[i, j] = 0
            else:
                if (i == 0):
                    e = Operator("del", S1, 0, S2, j-1)
                    D[i, j] = D[i, j - 1] + e.cost_edit(f, g.sim_Wu_Palmer)
                if (j == 0):
                    e = Operator("add", S2, 0, S1, i - 1)
                    D[i, j] = D[i - 1, j] + e.cost_edit(f, g.sim_Wu_Palmer)
                if (i != 0 and j != 0):

                    e_mod = Operator("mod", S2, j - 1, S1, i - 1)
                    e_del = Operator("del", S2, j - 2, S1, i - 1)
                    e_add = Operator("add", S1, i - 2, S2, j - 1)
                    cost_mod = e_mod.cost_edit(f, g.sim_Wu_Palmer)
                    cost_del = e_del.cost_edit(f, g.sim_Wu_Palmer)
                    cost_add = e_add.cost_edit(f, g.sim_Wu_Palmer)

                    D[i, j] = min(D[i - 1, j - 1] + cost_mod,
                                  D[i - 1, j] + cost_del,
                                  D[i, j - 1] + cost_add)
    return D[len(S1), len(S2)]

onto = Ontology('ontologie.txt')

S1 = [onto.graph.vs()[17], onto.graph.vs()[14]]
S2 = [onto.graph.vs()[13], onto.graph.vs()[14]]
S3 = [onto.graph.vs()[15], onto.graph.vs()[7]]
S4 = [onto.graph.vs()[15], onto.graph.vs()[8]]
S5 = [onto.graph.vs()[15], onto.graph.vs()[9]]
S6 = [onto.graph.vs()[7], onto.graph.vs()[16], onto.graph.vs()[10]]
S7 = [onto.graph.vs()[7], onto.graph.vs()[13], onto.graph.vs()[10]]
S8 = [onto.graph.vs()[7], onto.graph.vs()[16], onto.graph.vs()[7], onto.graph.vs()[17], onto.graph.vs()[10]]
S9 = [onto.graph.vs()[7], onto.graph.vs()[13], onto.graph.vs()[7], onto.graph.vs()[17], onto.graph.vs()[10]]
S10 = [onto.graph.vs()[8]]
S11 = [onto.graph.vs()[8], onto.graph.vs()[11], onto.graph.vs()[8]]
S12 = [onto.graph.vs()[7], onto.graph.vs()[17], onto.graph.vs()[7], onto.graph.vs()[13], onto.graph.vs()[10]]
S13 = [onto.graph.vs()[10], onto.graph.vs()[16], onto.graph.vs()[7], onto.graph.vs()[17], onto.graph.vs()[7]]
S14 = [onto.graph.vs()[8], onto.graph.vs()[12], onto.graph.vs()[8]]
S15 = [onto.graph.vs()[7], onto.graph.vs()[9], onto.graph.vs()[14], onto.graph.vs()[8], onto.graph.vs()[12]]

Sequences = [S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12, S13, S14, S15]

CED_matrix = np.zeros((len(Sequences), len(Sequences)))
f = Contextual_function().gaussian

for i in range(len(Sequences)) :
    for j in range(len(Sequences)):
        S1 = Sequences[i]
        S2 = Sequences[j]
        CED_matrix[i,j] = max(cedDP(S1, S2, f, onto), cedDP(S1, S2, f, onto))
print(CED_matrix)