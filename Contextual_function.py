#!/usr/bin/python
# -*- coding: utf-8
from math import *

class Contextual_function :
    def __init__(self) :
        pass

    def unit(self):
        return 1

    def gaussian(self, x, k, a = 1.0) :
        return exp(-1.0/2 * (x-k)**2/a)

    def local(self, x, k,) :
        return 1 if (x == k) else 0

    def elipse(self, x, k, a = 1.0) :
        return sqrt(1 - (x - k)**2/a)

    def phi(self, x, k, a = 1.0, b = 1.0, n = 2) :
        return 1.0 / ((abs(a * x - k)**n)/b +1)