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

def objective(x, qs, ps):
  assert(len(x) == 4)
  (theta_q, phi_q, theta_p, phi_p) = x
  phi_bar_q = math.cos(phi_q)
  phi_bar_p = math.cos(phi_p)
  points_q = project_to_plane(qs, theta_q, phi_q)
  points_p = project_to_plane(ps, theta_p, phi_p)

  p = Polygon(points_p, theta_p, phi_p, phi_bar_p)
  q = Polygon(points_q, theta_q, phi_q, phi_bar_q)
  s = q.compute_largest_scaling(p)
  #print(str(s))
  return -s**2

def get_obj(qs, ps):
  return lambda x: objective(x, qs, ps)

def optimize(q, ps, seed=1338, x0=None): # q is a silhouette of p
# variables:
# q: theta_q, phi_q
# p: theta_p, phi_p

# maximize scaling
# theta, phi

  jac = '3-point'

  constraints = []
  cons = []

  rng = np.random.default_rng(seed)
  if not x0:
    x0 = list(rng.random(4))

  res = minimize(get_obj(q, ps), x0, method='SLSQP', jac=jac, constraints=constraints)
  return res

def test_dual():
  c = snub_cube()
  c = read_file('Catalan/07LpentagonalIcositetrahedron.txt')[1]

  #c = cube()

  qa = [-0.464478427013, 0.66954456377]
  pa = [1.570798385, 0]

  qa = [4.62855249619, 0.078693308675666]
  pa =[5.817518733566667, -0.102768517482666]

  dual = dual_polytope(c)
  

  test_containment(dual, pa, qa) # 1.000244 optimize





  #  test_containment(, [1.39437813, 0.037334784167499], [1.6361904311, -0.37044476646]) # 1.000244 optimize
class Result:
  def __init__(self):
    self.x = [0,0,0,0]

def optimize_inner(q, p, seed, x0):
  try:
    r = optimize(q, p, 1338, x0)
    return r
  except KeyboardInterrupt:
    sys.exit()
  except:
    print('Exception, continuing')
    return Result()


def run_improve(p=None, n=1):
  if p is None:
    p = dual_polytope(rhombicosidodecahedron())
    p = dual_polytope(snub_cube())

  theta_q = 4.91885536
  phi_q = math.acos(-0.4651241815)
  theta_p = 0.85533109966
  phi_p = math.acos(-0.51181376779)
  rng = np.random.default_rng()
  test_containment(p, [theta_q, math.cos(phi_q)], [theta_p, math.cos(phi_p)])

  contains = False
  r = None
  best_r = None
  best_scaling = 0.0
  t = time.time()
  q = p

  if False:
    angles1 = []
    n_theta = n
    for i in range(n_theta):
      theta = (2 * math.pi * i) / n_theta
      for j in range(n):
        phi_bar = -1 + 2 * j / (n - 1.0)
        phi = math.acos(phi_bar)
        angles1.append((theta, phi))

    angles = []
    for angle1 in angles1:
      for angle2 in angles1:
        angles.append((angle1[0], angle1[1], angle2[0], angle2[1]))
  else:
    angles = []
    for i in range(n):
      theta_q, phi_q, theta_p, phi_p = list(rng.random(4))
      theta_q = 2 * math.pi * theta_q
      theta_p = 2 * math.pi * theta_p
      phi_q = math.acos(2 * (phi_q - 0.5))
      phi_p = math.acos(2 * (phi_p - 0.5))
      angles.append((theta_q, phi_q, theta_p, phi_p))

  i = 0
  n_tests = len(angles)
  n_per_chunk = 16
  n_chunks = round(n_tests / n_per_chunk + 1)
  if True:
    with Pool() as pool:
      for chunk_i in range(n_chunks):
        begin_i = min(chunk_i * n_per_chunk, n_tests)
        end_i = min((chunk_i + 1) * n_per_chunk, n_tests)
        #if end_i < 412624: continue # snub_cube

        print(str(begin_i) + ' of ' + str(len(angles)) + '. ')

        if begin_i == end_i: continue
        chunk_angles = angles[begin_i:end_i]
        result = pool.map(partial(optimize_inner, q, p, 1338), chunk_angles)

        for r in result:
          theta_q, phi_q, theta_p, phi_p = r.x
          contains, largest_scaling = test_containment(p, [theta_q, math.cos(phi_q)], [theta_p, math.cos(phi_p)])
          if contains:
            print('contains')
            if largest_scaling > best_scaling:
              best_scaling = largest_scaling
              best_r = r
        print('Best scaling: %s' % best_scaling)


  else:
    for theta_q, phi_q, theta_p, phi_p in angles:
      print(str(i) + ' of ' + str(len(angles)) + '. ')
      i += 1
      #if i < 412234: continue

      try:
        r = optimize(q, p, 1338, x0=[theta_q, phi_q, theta_p, phi_p])
        theta_q, phi_q, theta_p, phi_p = r.x

        contains, largest_scaling = test_containment(p, [theta_q, math.cos(phi_q)], [theta_p, math.cos(phi_p)])
        if contains:
          print('contains')
          if largest_scaling > best_scaling:
            best_scaling = largest_scaling
            best_r = r

        print('Best scaling: %s' % best_scaling)
      except KeyboardInterrupt:
        sys.exit()
      except:
        print('Exception, continuing')

  d = time.time() - t
  print('Time: %s' % d)
  return largest_scaling > 1.0, best_r
  

def run():
  q = [(-1,-1,1), (-1,1,-1), (-1,1,1), (1,-1,-1),(1,-1,1),(1,1,-1)]
  p = cube()

  p = snub_cube()
  p = rhombicosidodecahedron()
  #p = dual_polytope(rhombicosidodecahedron())
  silhouettes = get_silhouettes(p, n=400)
  any_contains = False
  t = time.time()
  for rni in range(10):
    ii = 0
    for s in silhouettes:
      q = [p[i] for i in s]
      try:
        r =  optimize(q, p, 1338 + rni)
        theta_q, phi_q, theta_p, phi_p = r.x

        print(str(rni) + ", " + str(ii) + ' of ' + str(len(silhouettes)) + (' * ' if any_contains else ' '), end='')
        contains = test_containment(p, [theta_q, math.cos(phi_q)], [theta_p, math.cos(phi_p)])
        any_contains = any_contains or contains
      except KeyboardInterrupt:
        sys.exit()
      except:
        print('Exception, continuing')
      ii += 1
  d = time.time() - t
  print('Time: %s' % d)
  return r

def run_johnson():
  d = 'Johnson/'

  left = ['Gyroelongated Pentagonal Rotunda', 'Gyroelongated Square Bicupola', 'Gyroelongated Pentagonal Cupolarotunda', 'Triaugmented Truncated Dodecahedron']

  left = ['Gyrate Rhombicosidodecahedron', 'Parabigyrate Rhombicosidodecahedron', 'Metabigyrate Rhombicosidodecahedron', 'Trigyrate Rhombicosidodecahedron', 'Diminished Rhombicosidodecahedron', 'Paragyrate Diminished Rhombicosidodecahedron'] # no found after 100 runs

  files = [f for f in listdir(d) if isfile(join(d, f)) and len(f) > 4 and f[-4:] == '.txt']
  #files = [f for f in files if any(l in f for l in left)]
  for f in files:
    name, p = read_file(d + f)
    if name in left:
      print(name)
      contains, r = run_improve(p, 100)
      if contains:
        print(name + ' contains ' + str(r))

def run_catalan():
  d = 'Catalan/'
  files = [f for f in listdir(d) if isfile(join(d, f)) and len(f) > 4 and f[-4:] == '.txt']

  for f in files:
    name, p = read_file(d + f)
    contains, r = run_improve(p, 100)
    if contains:
      print(name + ' contains ' + str(r))