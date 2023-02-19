from scipy.spatial import ConvexHull, convex_hull_plot_2d
import matplotlib.pyplot as plt
import numpy as np
import math
import itertools
import time
from multiprocessing import Pool
from functools import partial
import sys
from scipy.optimize import minimize
from polytopes import *
from polygon import *
from projection import *
from os import listdir
from os.path import isfile, join
import itertools
import mpmath


def rotate(ps, angle, i, j):
  assert(i != j)
  sa = math.sin(angle)
  ca = math.cos(angle)

  # R = [ca -sa]
  #     [sa  ca]

  for p in ps:
    pi = p[i]
    pj = p[j]
    p[i] = ca * pi - sa * pj
    p[j] = sa * pi + ca * pj

class State:
  def __init__(self, n):
    self.combinations = list(itertools.combinations(range(n), 2))
    self.cached_x = None
    self.cached_p = None

  def combination(self, i):
    return self.combinations[i]

  def get_p(self, x, p):
    if any(self.cached_x != x):
      y = [list(z) for z in p]
      angles = x[:-1]
      scaling = x[-1]
      assert(len(angles) == len(self.combinations))
      for angle, (i,j) in zip(angles, self.combinations):
        rotate(y, angle, i, j)
      self.cached_p = [[zi * scaling for zi in z] for z in y]
      self.cached_x = x
    return self.cached_p

def constraints_fun(x, p, state, i, j, k):
  p = state.get_p(x, p)

  if k == 0:
    return p[i][j] + 1 # p[i][j] >= -1
  else:
    return 1 - p[i][j] # p[i][j] <= 1

def get_constraint(p, state, i, j, k):
  return lambda x: constraints_fun(x, p, state, i, j, k)

def objective(x):
  return -x[-1]

def get_p(m, n):
  p = [list(x) for x in itertools.product([-1,1], repeat=m)]
  for pt in p:
    pt += [0] * (n-m)
  return p

def get_constraints(p, state, m, n):
  constraints = []
  for i in range(len(p)):
    for j in range(n):
      for k in range(2):
        constr = {'type': 'ineq', 'fun': get_constraint(p, state, i, j, k)}
        constraints.append(constr)
  return constraints

def optimize(m, n, seed=1338, x0=None):
  assert(n > m)
  
  p = get_p(m, n)
 
  n_angles = math.comb(n, 2);

  method = 'SLSQP'
  jac = None if method == 'Nelder-Mead' else '3-point'

  state = State(n)

  constraints = get_constraints(p, state, m, n)

  if not x0:
    rng = np.random.default_rng(seed)
    x0 = list(rng.random(n_angles))
    x0.append(1)


  res = minimize(objective, x0, method=method, jac=jac, constraints=constraints)
  return res


def run(m, n, k=10):
  best_r = None
  for s in range(k):
    r = optimize(m, n, s + 1338)
    if r.success:
      if not best_r:
        best_r = r
      elif r.fun < best_r.fun:
        best_r = r

  
  # mpmath.identify(-best_r.x)
  return best_r
