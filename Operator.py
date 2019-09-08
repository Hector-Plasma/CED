#!/usr/bin/python
# -*- coding: utf-8
from enum import Enum

class Edit_operator(Enum):
    ADD = 1
    MOD = 2
    DEL = 3

class Operator :

    def __init__(self, op, S, x, k) :
        """
        :param op: Operator performed
        :param S: On Sequence
        :param x: Symbol
        :param k: At Index
        """
        self.op = op
        self.S = S
        self.x = x
        self.k = k

    def context_vector(self, f):
        if (self.op == Edit_operator.MOD):
            return map(lambda x: f(self.k, x), range(len(self.S)))
        if (self.op == Edit_operator.ADD):
            if(self.k == 0) :
                return [1.0] + \
                   map(lambda x: f(self.k, x), range(1, len(self.S)))
            else :
                return map(lambda x: f(self.k, x + 1), range(self.k - 1)) + \
                   [1.0, 1.0] + \
                   map(lambda x: f(self.k, x), range(self.k + 1, len(self.S)))
        else :
            if (self.k == 0):
                return [0, 1.0] + \
                       map(lambda x: f(self.k, x - 1), range(2, len(self.S)))
            else :
                return map(lambda x: f(self.k, x + 1), range(self.k - 1)) + \
                   [1.0, 0, 1.0] + \
                   map(lambda x: f(self.k, x-1), range(self.k + 2, len(self.S)))

    def cost_edit(self, f, sim, alpha=1):
        sim_v = map(lambda s: sim(s, self.x), self.S)
        ctx_v = self.context_vector(f)
        simWeigth_v = []
        for i in range(len(sim_v)):
            simWeigth_v += [sim_v[i] * ctx_v[i]]
        delta_local_cost = 1 if self.op != Edit_operator.MOD else 1 - sim(self.S[self.k], self.x)
        return (alpha * delta_local_cost) + (1 - alpha) * (1 - max(simWeigth_v))
