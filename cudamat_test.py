import numpy as np
import gnumpy as g
from bkputils import *
import scipy.special

g.max_memory_usage = 285 * (1024 * 1024) # no idea where this comes from, but it makes gnumpy not crash (empiric value)

w = 10000
h = 10000

write("making random matrices")
m1 = np.random.rand(w, h)
m2 = np.random.rand(h, w)
writeDone()

write("numpy multiply")
n = np.dot(m1, m2)
p = scipy.special.expit(n)
writeDone()

write("gnumpy setup")
a = g.garray(m1)
b = g.garray(m2)
writeDone()

write("gnumpy multiply")
c = g.dot(a, b)
c = g.logistic(c)
writeDone()