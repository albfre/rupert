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
  
class Result:
  def __init__(self):
    self.x = [0,0,0,0]

def objective(x, qs, ps):
  assert(len(x) == 4)
  (theta_q, phi_q, theta_p, phi_p) = x
  phi_bar_q = math.cos(phi_q)
  phi_bar_p = math.cos(phi_p)
  points_q = project_to_plane(qs, theta_q, phi_q)
  points_p = project_to_plane(ps, theta_p, phi_p)

  q = Polygon(points_q, theta_q, phi_q, phi_bar_q)
  p = Polygon(points_p, theta_p, phi_p, phi_bar_p)
  s = q.compute_largest_scaling(p)
  return -s**2

def get_obj(q, p):
  return lambda x: objective(x, q, p)

def get_shifted_objective(obj, x, i, h):
  x_shift = list(x)
  x_shift[i] = x[i] + h
  fp = obj(x_shift)
  x_shift[i] = x[i] - h
  fm = obj(x_shift)
  return fp, fm

def get_gradient(obj, x):
  h = 1e-8
  f = obj(x)
  grad = [0]*len(x)

  for i in range(len(x)):
    fp, fm = get_shifted_objective(obj, x, i, h)
    grad[i] = (fp - fm) / (2 * h)
  return grad

def optimize_gradient_descent(obj, x):
  improves = True
  f = obj(x)
  while improves:
    improves = False

    grad = get_gradient(obj, x)
    delta = 1
    while delta > 1e-8:
      new_x = [y - delta * z for y, z in zip(x, grad)]
      new_f = obj(new_x) 
      if new_f < f - 1e-8:
        f = new_f
        x = new_x
        improves = True
        #print('New obj: %s, delta: %s' % (new_obj, delta))
        break
      delta /= 2
  r = Result()
  r.x = x
  return r

def optimize(obj, x0):
# variables:
# q: theta_q, phi_q
# p: theta_p, phi_p

# maximize scaling
# theta, phi

  method = 'Nelder-Mead' # SLSQP
  method = 'L-BFGS-B' # SLSQP # bfgs: triakis 200s, pentagonal
  method = 'SLSQP' # SLSQP # bfgs: triakis 200s, pentagonal
  jac = None if method == 'Nelder-Mead' else '3-point'

  constraints = []
  cons = []

  res = minimize(obj, x0, method=method, jac=jac, constraints=constraints)
  return res

def optimize_inner(q, p, gradient_descent, x0):
  try:
    obj = get_obj(q, p)
    if gradient_descent:
      r = optimize_gradient_descent(obj, x0)
    else:
      r = optimize(obj, x0)
    return r
  except KeyboardInterrupt:
    sys.exit()
  except:
    print('Exception, continuing')
    return Result()

def find_trajectory(p):
  qa = [4.627912, math.acos(0.0715399)]
  pa =[5.8376, math.acos(-0.112877)]

  angles = []
  angles.append((qa[0], qa[1], pa[0], pa[1]))
  p_orig = list(p)

  contains = True
  expansion = 0.13
  expansion_increment = 0.01
  while contains:
    print('Expansion: ' + str(expansion))
    print('Angles: ' +str(angles))
    p = expand(p_orig, expansion)
    q = p
    r = optimize_inner(q, p, False, angles[0])
    theta_q, phi_q, theta_p, phi_p = r.x
    contains, largest_scaling, alpha, trans = test_containment(p, [theta_q, math.cos(phi_q)], [theta_p, math.cos(phi_p)])
    if contains:
      angles = [(theta_q, phi_q, theta_p, phi_p)]
    else:
      expansion -= expansion_increment
      expansion_increment /= 2
      contains = True

    expansion += expansion_increment

def run(p=None, n=1, gradient_descent=False, early_return=False):
  if p is None:
    p = triakis_tetrahedron()

  contains = False
  r = None
  best_r = None
  best_scaling = 0.0
  t = time.time()
  q = p

  angles1 = []
  for i in range(n):
    theta = (2 * math.pi * i) / n
    for j in range(n):
      phi_bar = -1 + (2 * j / (n - 1.0) if n > 0 else 0)
      phi = math.acos(phi_bar)
      angles1.append((theta, phi))

  angles = []
  for angle1 in angles1:
    for angle2 in angles1:
      angles.append((angle1[0], angle1[1], angle2[0], angle2[1]))

  i = 0
  n_tests = len(angles)
  n_per_chunk = 16
  n_chunks = round(n_tests / n_per_chunk + 1)
  largest_scaling = 0.0

  with Pool() as pool:
    for chunk_i in range(n_chunks):
      begin_i = min(chunk_i * n_per_chunk, n_tests)
      end_i = min((chunk_i + 1) * n_per_chunk, n_tests)
      print(str(begin_i) + ' of ' + str(len(angles)) + '. ')
      if begin_i == end_i: continue

      chunk_angles = angles[begin_i:end_i]
      result = pool.map(partial(optimize_inner, q, p, gradient_descent), chunk_angles)

      for r in result:
        theta_q, phi_q, theta_p, phi_p = r.x
        try:
          contains, largest_scaling, alpha, trans = test_containment(p, [theta_q, math.cos(phi_q)], [theta_p, math.cos(phi_p)])
          if contains:
            print('contains')
            if largest_scaling > best_scaling:
              best_scaling = largest_scaling
              best_r = r
              if early_return:
                return True, best_r
        except:
          print('Except')
      print('Best scaling: %s, %s' % (best_scaling, best_r.x if best_r else None))

  d = time.time() - t
  print('Time: %s' % d)
  if best_r:
    theta_q, phi_q, theta_p, phi_p = best_r.x
    contains, largest_scaling, alpha, trans = test_containment(p, [theta_q, math.cos(phi_q)], [theta_p, math.cos(phi_p)])
  return largest_scaling > 1.0, best_r
  
def run_johnson():
  d = 'Johnson/'

  left = ['Gyrate Rhombicosidodecahedron', 'Parabigyrate Rhombicosidodecahedron', 'Metabigyrate Rhombicosidodecahedron', 'Trigyrate Rhombicosidodecahedron', 'Paragyrate Diminished Rhombicosidodecahedron']

  files = [f for f in listdir(d) if isfile(join(d, f)) and len(f) > 4 and f[-4:] == '.txt']
  #files = [f for f in files if any(l in f for l in left)]
  for f in files:
    name, p = read_file(d + f)
    if name in left:
      print(name)
      contains, r = run_improve(p, 5, True)
      if contains:
        print(name + ' contains ' + str(r))

def run_catalan():
  d = 'Catalan/'
  files = [f for f in listdir(d) if isfile(join(d, f)) and len(f) > 4 and f[-4:] == '.txt']

  for f in files:
    name, p = read_file(d + f)
    contains, r = run_improve(p, 5, True)
    if contains:
      print(name + ' contains ' + str(r))
