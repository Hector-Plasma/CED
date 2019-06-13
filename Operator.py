#!/usr/bin/python
# -*- coding: utf-8
class Operator :
    def __init__(self, op, S1, i, S2, j) :
        self.op = op
        self.S1 = S1
        self.i = i
        self.S2 = S2
        self.j = j

    def context_vector(self, f):
        if (self.op == "mod"):
            return map(lambda x: f(self.i, x), range(len(self.S1)))
        else:
            return map(lambda x: f(self.i, x + 1), range(self.i - 1)) + \
                   [1.0, 1.0] + \
                   map(lambda x: f(self.i, x), range(self.i + 1, len(self.S1)))

    def cost_edit(self, f, sim, alpha=0):
        sim_v = map(lambda s: sim(s, self.S2[self.j]), self.S1)  
        ctx_v = self.context_vector(f)
        simWeigth_v = []
        for i in range(len(sim_v)):
            simWeigth_v += [sim_v[i] * ctx_v[i]]
        delta_local_cost = 1 if self.op != "mod" else 1 - sim(self.S1[self.i], self.S2[self.j])
        return (alpha * delta_local_cost) + (1 - alpha) * (1 - max(simWeigth_v))
