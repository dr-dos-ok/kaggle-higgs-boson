import math, time
import scipy.optimize as optimize
import numpy as np
import pandas as pd
from scipy.special import expit
from bkputils import *

import matplotlib.pyplot as plt

GLOBAL_START = time.time()

def gauss(x):
	global STD
	global MEAN

	pt1 = 1.0 / (STD * np.sqrt(2.0 * math.pi))
	pt2 = np.exp(-(x-MEAN)**2/(2.0*STD*STD))

	return pt1 * pt2

def eval_net(xs, weights):
	global STD
	global MEAN

	numerator1, someconst = weights

	wa = 2.0 / STD
	wb = -wa
	wc = numerator1 / STD
	wd = wc

	shift = (2.0 * MEAN) / STD
	ba = someconst - shift
	bb = someconst + shift
	bc = -wc

	la = expit((wa*xs) + ba)
	lb = expit((wb*xs) + bb)
	foo = wc*la + wd*lb + bc
	lc = expit(foo)

	return lc

NUM_SAMPLES = 40.0
def net_err(weights):
	step = (STD*8.0) / (NUM_SAMPLES - 1.0)
	xs = np.arange(MEAN - (STD*4.0), MEAN + (STD*4.0) + step, step, dtype=np.float64)

	net_output = eval_net(xs, weights)

	diff = expit(gauss(xs)) - net_output
	err = np.sum(diff*diff)

	return err

MEAN = -100.0
STD = 1.0

std_weights = np.array([0.54651296, 1.85061951])

print net_err(std_weights)
res = optimize.minimize(net_err, std_weights, method="Nelder-Mead")
print res.fun
print std_weights
print res.x

writeDone(time.time() - GLOBAL_START)