#!/usr/bin/python
# -*- coding: utf-8
import numpy as np
from Operator import *
from Ontology import *
from Contextual_function import *

def one_sided_CED(S1, S2, f, sim) :
    D = np.zeros((len(S1) + 1, len(S2) + 1))
    for i in range(len(S1) + 1):
        for j in range(len(S2) + 1):
            if (i == 0 and j == 0):
                D[i, j] = 0
            else:
                if (i == 0):
                    e = Operator(Edit_operator.ADD, S1, S2[j - 1], 0)
                    D[i, j] = D[i, j - 1] + e.cost_edit(f, sim)
                if (j == 0):
                    e = Operator(Edit_operator.DEL, S1, S1[i - 1], i - 1)
                    D[i, j] = D[i - 1, j] + e.cost_edit(f, sim)
                if (i != 0 and j != 0):

                    cost_mod = Operator(Edit_operator.MOD, S1, S2[j - 1], i - 1).cost_edit(f, sim)
                    cost_del = Operator(Edit_operator.DEL, S1, S1[i - 1], i - 1).cost_edit(f, sim)
                    cost_add = Operator(Edit_operator.ADD, S1, S2[j - 1], i - 2).cost_edit(f, sim)

                    D[i, j] = min(D[i - 1, j - 1] + cost_mod,
                                  D[i - 1, j] + cost_del,
                                  D[i, j - 1] + cost_add)
    #print D
    return D[len(S1), len(S2)]


#----- Main -----#

onto = Ontology('ontologie.txt')
#onto.print_ontology()

"""
0,{Activity}
1,{Move Activity}
2,{Shopping}
3,{Cultural Activity}
4,{Sport Activity}
5,{Human powered transports}
6,{Motor Vehicles}
7,{Walk}
8,{Cycle}
9,{Drive car}
10,{Take bus}
11,{Bakery}
12,{Other kind of shoppinf}
13,{Cinema}
14,{Danse}
15,{Librairy}
16,{Swim}
17,{Play Football}
"""

S1 = [onto.graph.vs()[8]]
S2 = [onto.graph.vs()[8], onto.graph.vs()[11], onto.graph.vs()[8]]
S3 = [onto.graph.vs()[7], onto.graph.vs()[12], onto.graph.vs()[7]]
S4 = [onto.graph.vs()[17], onto.graph.vs()[7], onto.graph.vs()[10], onto.graph.vs()[13]]
S5 = [onto.graph.vs()[10], onto.graph.vs()[13], onto.graph.vs()[7], onto.graph.vs()[17]]
S6 = [onto.graph.vs()[7], onto.graph.vs()[16], onto.graph.vs()[10], onto.graph.vs()[13]]
S7 = [onto.graph.vs()[7], onto.graph.vs()[15], onto.graph.vs()[7], onto.graph.vs()[13]]
S8 = [onto.graph.vs()[14], onto.graph.vs()[7], onto.graph.vs()[15]]
S9 = [onto.graph.vs()[8], onto.graph.vs()[16], onto.graph.vs()[8], onto.graph.vs()[15]]
S10 = [onto.graph.vs()[15], onto.graph.vs()[9], onto.graph.vs()[10], onto.graph.vs()[12]]

Sequences = [S1, S2, S3, S4, S5, S6, S7, S8, S9, S10]

dist_matrix = np.zeros((len(Sequences), len(Sequences)))
f = Contextual_function().gaussian
# Do not forget to custom parameters of CED in : 
# alpha : Operator.py
# f     : Contextual_function.py

for i in range(len(Sequences)) :
    for j in range(len(Sequences)):
        S_1 = Sequences[i]
        S_2 = Sequences[j]
        dist_matrix[i,j] = max(one_sided_CED(S_1, S_2, f, onto.sim_wu_Palmer), one_sided_CED(S_2, S_1, f, onto.sim_wu_Palmer))

print(dist_matrix)
file_name = "your_file_name_here" # 

np.savetxt(file_name, dist_matrix, delimiter=",")
