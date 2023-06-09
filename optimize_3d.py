from scipy.spatial import ConvexHull, convex_hull_plot_2d
import matplotlib.pyplot as plt
import numpy as np
import math
import itertools
import time
from multiprocessing import Pool
from functools import partial
from scipy.optimize import minimize
from polytopes import *
from polygon import *
from projection import *

def constraints_fun(x, q_in, ps_in):
  assert(len(x) == 6)
  theta_q, phi_q, theta_p, phi_p, alpha, t = x
  q = project_to_plane(q_in, theta_q, phi_q)
  ps = rotate(project_to_plane(ps_in, theta_p, phi_p), alpha)

  constraints = []
  r = []
  for p in ps:
    n = len(q)
    rp = []
    rn = []
    for i in range(n):
      next_i = (i + 1) % n
      q1 = q[next_i]
      q0 = q[i]
      det = (q1[0] - p[0]) * (q0[1] - p[1]) - (q0[0] - p[0]) * (q1[1] - p[1])
      constraints.append(det)
      rp.append(det >= 0.0)
      rn.append(det <= 0.0)
    r.append(all(rp) or all(rn))
  return constraints

def constraints_fun_ij(x, qs, ps, i, j):
  assert(len(x) == 6)
  (theta_q, phi_q, theta_p, phi_p, alpha, t) = x
  next_i = (i + 1) % len(qs)
  q0, q1 = project_to_plane([qs[i], qs[next_i]], theta_q, phi_q)
  p = rotate(project_to_plane([ps[j]], theta_p, phi_p), alpha)[0]
  det = (q1[0] - p[0]) * (q0[1] - p[1]) - (q0[0] - p[0]) * (q1[1] - p[1])
  #print('C: %s, %s: %s' % (i,j,det))
  return t - det

def containment_determinant(x, qs, ps, i, j):
  (theta_q, phi_q, theta_p, phi_p, alpha, t) = x
  return constraints_fun_ij(x, qs, ps, i, j)

def obj(x):
  return sum(y**2 for y in x)

def obj_grad(x):
  return [2 * y for y in x]

def constr_test(x):
  return x[2] - 1

def objective(x):
  return x[-1]

def objective_grad(x):
  return [0,0,0,0,0,1]

def get_fun(q, ps, i, j):
  return lambda x: constraints_fun_ij(x, q, ps, i, j)

def optimize(q, ps, seed=1338, x0=None): # q is a silhouette of p
# variables:
# q: theta_q, phi_q
# p: theta_p, phi_p, alpha

# minimize t
# t, alpha, theta_q, phi_q, theta_p, phi_p
# st R * M * p is inside M * q

  jac = '3-point'

  constraints = []
  datas = []
  cons = []
  ii = 0
  for j in range(len(ps)):
    for i in range(len(q)):
      datas.append((i,j))
      cons.append(get_fun(q, ps, i, j))
      constr = {'type': 'ineq', 'fun': cons[-1]}
      constraints.append(constr)
      ii += 1

  #x0 = [0.5, 0.5, 0.5, 0.5, 0.5, 10]

  rng = np.random.default_rng(seed)
  if not x0:
    x0 = list(rng.random(6))
    x0[-1] = 10

  res = minimize(objective, x0, method='SLSQP', jac=objective_grad, constraints=constraints)
  #constraints = [{'type':'ineq', 'fun': constr_test}]
  #res = minimize(obj, x0, method='SLSQP', jac=obj_grad, constraints=constraints)
  #for c in constraints:
  #  print(str(c['fun'](r.x)))
  return res, constraints

def get_silhouette(polyhedron, theta, phi):
  points_2d = project_to_plane(polyhedron, theta, phi)
  hull = ConvexHull(points_2d)
  vertices = list(hull._vertices)
  min_vertex = min(vertices)
  min_index = vertices.index(min_vertex)
  vertices = vertices[min_index:] + vertices[:min_index]
  return tuple(vertices)


def get_silhouettes(polyhedron):
  n = 100
  silhouettes = []
  for i in range(n):
    theta = (2 * math.pi * i) / n
    for j in range(n):
      phi_bar = -1 + 2 * j / (n - 1.0)
      phi = math.acos(phi_bar)
      silhouettes.append(get_silhouette(polyhedron, theta, phi))
  return sorted(list(set(silhouettes)))


def run_improve():
  p = dodecahedron()
  theta_q = 4.91885536
  phi_q = math.acos(-0.4651241815)
  theta_p = 0.85533109966
  phi_p = math.acos(-0.51181376779)
  s = get_silhouette(p, theta_q, phi_q)
  q = [p[i] for i in s]
  r, constraints = optimize(q, p, 1338, x0=[theta_q, phi_q, theta_p, phi_p, 0, 10])
  theta_q, phi_q, theta_p, phi_p, alpha, t = r.x
  contains = test_containment(p, [theta_q, math.cos(phi_q)], [theta_p, math.cos(phi_p)])
  

def run_silhouette():
  p = cube()
  silhouette_q = [2, 3, 7, 5, 4, 0]
  silhouette_p = [6, 4, 0, 2]
  q = [p[i] for i in silhouette_q]
  ps = [p[i] for i in silhouette_p]
  r, constraints = optimize(q, ps, 1338)
  theta_q, phi_q, theta_p, phi_p, alpha, t = r.x
  contains, largest_scaling, alpha, trans = test_containment(p, [theta_q, math.cos(phi_q)], [theta_p, math.cos(phi_p)])
  print(str(alpha))
  print(str(trans))

  for j in range(len(ps)):
    for i in range(len(q)):
      print(str(containment_determinant(r.x, q, ps, i, j)))
   

def run():
  q = [(-1,-1,1), (-1,1,-1), (-1,1,1), (1,-1,-1),(1,-1,1),(1,1,-1)]
  p = cube()

  p = dodecahedron()
  p = cube()
  p = pentagonal_icositetrahedron()
  silhouettes = get_silhouettes(p)
  any_contains = False
  best_r = []
  best_scaling = 1
  for rni in range(1):
    ii = 0
    for s in silhouettes:
      q = [p[i] for i in s]
      r, constraints = optimize(q, p, 1338 + rni)
      theta_q, phi_q, theta_p, phi_p, alpha, t = r.x

      print(str(rni) + ", " + str(ii) + ' of ' + str(len(silhouettes)) + (" " + str(best_r.x) + " " + str(best_scaling) if any_contains else ' '), end='')
      contains, largest_scaling, alpha, trans = test_containment(p, [theta_q, math.cos(phi_q)], [theta_p, math.cos(phi_p)])
      any_contains = any_contains or contains
      if largest_scaling > best_scaling:
        best_scaling = largest_scaling
        best_r = r
      ii += 1
  return r, constraints
