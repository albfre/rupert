from scipy.spatial import ConvexHull, convex_hull_plot_2d
import math
import time
from multiprocessing import Pool
from functools import partial
import sys
from scipy.optimize import minimize
from polyhedra import *
from polygon import *
from projection import *
  
class Result:
  def __init__(self):
    self.x = [0]*4

def objective(x, qs, ps):
  assert(len(x) == 4)
  (theta_q, phi_q, theta_p, phi_p) = x
  points_q = project_to_plane(qs, theta_q, phi_q)
  points_p = project_to_plane(ps, theta_p, phi_p)

  q = Polygon(points_q, theta_q, phi_q)
  p = Polygon(points_p, theta_p, phi_p)
  s = q.compute_largest_scaling(p)
  return -s**2

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
        break
      delta /= 2
  r = Result()
  r.x = x
  return r

def optimize(obj, x0):
  method = 'Nelder-Mead' # SLSQP
  jac = None if method == 'Nelder-Mead' else '3-point'
  return minimize(obj, x0, method=method, jac=jac, constraints=[], tol=1e-4)

def optimize_wrapper(q, p, gradient_descent, x0):
  try:
    obj = lambda x: objective(x, q, p)
    if gradient_descent:
      return optimize_gradient_descent(obj, x0)
    else:
      return optimize(obj, x0)
  except KeyboardInterrupt:
    sys.exit()
  except:
    print('Exception, continuing')
    return Result()

def run(p=None, n=1, gradient_descent=False, early_return=False):
  if p is None:
    p = triakis_tetrahedron()

  contains = False
  r = None
  best_r = None
  best_scaling = 0.0
  t = time.time()
  q = p

  thetas = [2 * math.pi * i / n for i in range(n)]
  phis = [math.acos(-1 + (2 * i / (n - 1) if n > 1 else 0)) for i in range(n)]
  angles = [(theta, phi) for theta in thetas for phi in phis]
  angles = [qa + pa for qa in angles for pa in angles]

  n_starting_points = len(angles)
  n_per_chunk = 16
  n_chunks = round(n_starting_points / n_per_chunk + 1)
  largest_scaling = 0.0

  with Pool() as pool:
    for chunk_i in range(n_chunks):
      begin_i = min(chunk_i * n_per_chunk, n_starting_points)
      end_i = min((chunk_i + 1) * n_per_chunk, n_starting_points)
      print(str(begin_i) + ' of ' + str(len(angles)) + '. ')
      if begin_i == end_i: continue

      chunk_angles = angles[begin_i:end_i]
      result = pool.map(partial(optimize_wrapper, q, p, gradient_descent), chunk_angles)

      for r in result:
        theta_q, phi_q, theta_p, phi_p = r.x
        try:
          contains, largest_scaling, alpha, trans = test_containment(p, [theta_q, phi_q], [theta_p, phi_p])
          if contains and largest_scaling > best_scaling:
            best_scaling = largest_scaling
            best_r = r
            if early_return:
              return True, best_r
        except:
          print('Except')
      print('Best scaling: %s, %s' % (best_scaling, best_r.x if best_r else None))

  print('Time: %s' % (time.time() - t,))
  if best_r:
    theta_q, phi_q, theta_p, phi_p = best_r.x
    contains, largest_scaling, alpha, trans = test_containment(p, [theta_q, phi_q], [theta_p, phi_p])
  return largest_scaling > 1.0, best_r
