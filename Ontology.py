#!/usr/bin/python
# -*- coding: utf-8
from igraph import *
from math import *

class Ontology :
    def __init__(self, file_name) :
        self.graph = Graph.Read_Ncol(file_name)

    def print_ontology(self) :
        for n in self.graph.vs():
            print(n)
        plot(self.graph, layout=self.graph.layout('rt', root=[0]), bbox=(500, 350))


    def LCA(self, x, y):
        """
        :param x
        :param y
        :return: Lowest Common Ancestor of x and y
        """
        if (x == y):
            return [[x]]
        if (x == self.graph.vs()[0] or y == self.graph.vs()[0]):
            return [[self.graph.vs()[0]]]
        else:
            AParents = set([x.index])
            BParents = set([])
            # All Predecessors of x
            new_Parents = set(self.graph.predecessors(x))
            while (len(new_Parents) != 0):
                AParents |= new_Parents
                temp = set([])
                for n in new_Parents:
                    temp |= set(self.graph.predecessors(n))
                new_Parents = temp
            # First Ancestor of y and x
            new_Parents = set([y.index])
            while (len(new_Parents & AParents) < 1):
                BParents |= new_Parents
                temp = set([])
                for n in new_Parents:
                    temp |= set(self.graph.predecessors(n))
                new_Parents = temp
            return list(new_Parents & AParents)

    def max_prof(self):
        max_shortest_path = 0
        for i in range(0, len(self.graph.vs)):
            for j in range(i + 1, len(self.graph.vs)):
                max_shortest_path = max(max_shortest_path,
                                        self.graph.as_undirected().shortest_paths_dijkstra(source=self.graph.vs[i],
                                                                                           target=self.graph.vs[j],
                                                                                           weights=None))
        return max_shortest_path

    def sim_Wu_Palmer(self, x, y):
        d_comm = self.graph.as_undirected().shortest_paths_dijkstra(
            source=self.graph.vs[0],
            target=self.LCA(x, y)[0],
            weights=None)[0][0]
        d_x = self.graph.as_undirected().shortest_paths_dijkstra(source=self.graph.vs[0], target=x, weights=None)[0][0]
        d_y = self.graph.as_undirected().shortest_paths_dijkstra(source=self.graph.vs[0], target=y, weights=None)[0][0]
        if (x == y):
            return 1
        if (d_comm == 0):
            return 0
        else:
            return (2.0 * int(d_comm)) / (d_x + d_y)

    def sim_leacock_chodrow(self, x, y):
        prof_max = self.max_prof(self)[0][0]
        d_xy = self.graph.as_undirected().shortest_paths_dijkstra(source=x, target=y, weights=None)[0][0]
        return -log((prof_max + d_xy) / (2.0 * prof_max), 2)
