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
from pyscipopt import Model, sqrt

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

def sin_transformed(angle):
  return 2 * angle / (1 + angle * angle)

def cos_transformed(angle):
  return (1 - angle * angle) / (1 + angle * angle)

def project_to_plane_transformed(points, theta, phi):
  assert(all(len(p) == 3 for p in points))
  st = sin_transformed(theta)
  ct = cos_transformed(theta)
  sp = sin_transformed(phi)
  cp = cos_transformed(phi)
  return [[-st * p[0] + ct * p[1], -ct * cp * p[0] - st * cp * p[1] + sp * p[2]] for p in points]

def rotate_transformed(points, alpha):
  ca = cos_transformed(alpha)
  sa = sin_transformed(alpha)
  return [[ca * p[0] - sa * p[1], sa * p[0] + ca * p[1]] for p in points]

def project_to_plane_transformed2(points, st, ct, sp, cp):
  return [[-st * p[0] + ct * p[1], -ct * cp * p[0] - st * cp * p[1] + sp * p[2]] for p in points]

def rotate_transformed2(points, sa, ca):
  return [[ca * p[0] - sa * p[1], sa * p[0] + ca * p[1]] for p in points]

def constraints_fun_ij(x, qs, ps, i, j):
  assert(len(x) == 6)
  (theta_q, phi_q, theta_p, phi_p, alpha, t) = x
  next_i = (i + 1) % len(qs)
  q0, q1 = project_to_plane_transformed([qs[i], qs[next_i]], theta_q, phi_q)
  p = rotate_transformed(project_to_plane_transformed([ps[j]], theta_p, phi_p), alpha)[0]
  det = (q1[0] - p[0]) * (q0[1] - p[1]) - (q0[0] - p[0]) * (q1[1] - p[1])
  #print('C: %s, %s: %s' % (i,j,det))
  return t - det

def containment_determinant(x, qs, ps, i, j):
  (st_q, ct_q, st_p, ct_p, sp_q, cp_q, sp_p, cp_p, sa, ca, t) = x

  next_i = (i + 1) % len(qs)
  q0, q1 = project_to_plane_transformed2([qs[i], qs[next_i]], st_q, ct_q, sp_q, cp_q)
  p = rotate_transformed2(project_to_plane_transformed2([ps[j]], st_p, ct_p, sp_p, cp_p), sa, ca)[0]
  det = (q1[0] - p[0]) * (q0[1] - p[1]) - (q0[0] - p[0]) * (q1[1] - p[1])
  return det

def containment_determinant3(x, qs, ps, i, j, signs):
  (theta_q, theta_p, phi_q, phi_p, alpha, tx, ty, t) = x

  st_q = theta_q
  ct_q = sqrt(1 - theta_q * theta_q)

  sp_q = phi_q
  cp_q = sqrt(1 - phi_q * phi_q)

  st_p = theta_p
  ct_p = -sqrt(1 - theta_p * theta_p)

  sp_p = phi_p
  cp_p = -sqrt(1 - phi_p * phi_p)

  sa = alpha
  ca = -sqrt(1 - alpha * alpha)

  next_i = (i + 1) % len(qs)
  q0, q1 = project_to_plane_transformed2([qs[i], qs[next_i]], st_q, ct_q, sp_q, cp_q)
  p = rotate_transformed2(project_to_plane_transformed2([ps[j]], st_p, ct_p, sp_p, cp_p), sa, ca)[0]
  p[0] += tx
  p[1] += ty
  det = (q1[0] - p[0]) * (q0[1] - p[1]) - (q0[0] - p[0]) * (q1[1] - p[1])
  return det

def containment_determinant_angles(x, qs, ps, i, j):
  (theta_q, phi_q, theta_p, phi_p, alpha) = x

  next_i = (i + 1) % len(qs)
  q0, q1 = project_to_plane([qs[i], qs[next_i]], theta_q, phi_q)
  p = rotate(project_to_plane([ps[j]], theta_p, phi_p), alpha)[0]
  det = (q1[0] - p[0]) * (q0[1] - p[1]) - (q0[0] - p[0]) * (q1[1] - p[1])
  return det

def objective(x):
  return x[-1]

def get_angle3(s):
  angle1 = math.asin(s)
  return angle1

def get_angle(s, c):
  angle1 = math.asin(s)
  angle2 = math.pi - angle1

  diff1 = abs(math.cos(angle1) - c)
  diff2 = abs(math.cos(angle2) - c)

  return angle1 if diff1 < diff2 else angle2

def optimize3(q, ps, x0=None): # q is a silhouette of p
# variables:
# q: theta_q, phi_q
# p: theta_p, phi_p, alpha

# minimize t
# t, alpha, theta_q, phi_q, theta_p, phi_p
# st R * M * p is inside M * q


# replace sin(a), cos(a) by 2*t/(1+t^2), (1-t^2)/(1+t^2)

  model = Model('Example')
  lb = -1
  ub = 1
  theta_q = model.addVar('st_q', vtype='C', lb=lb, ub=ub)
  theta_p = model.addVar('st_p', vtype='C', lb=lb, ub=ub)
  phi_q = model.addVar('sp_q', vtype='C', lb=lb, ub=ub)
  phi_p = model.addVar('sp_p', vtype='C', lb=lb, ub=ub)
  alpha = model.addVar('salpha', vtype='C', lb=lb, ub=ub)
  tx = model.addVar('tx', vtype='C', lb=-0.01, ub=0.01)
  ty = model.addVar('ty', vtype='C', lb=-0.01, ub=0.01)
  t = model.addVar('t', vtype='C', lb=-2, ub=2)

  #theta_q = model.addVar('st_q', vtype='C', lb=0.98, ub=0.99)
  #theta_p = model.addVar('st_p', vtype='C', lb=0.99, ub=1)
  #phi_q = model.addVar('sp_q', vtype='C', lb=0.99, ub=1)
  #phi_p = model.addVar('sp_p', vtype='C', lb=0.92, ub=0.94)
  #alpha = model.addVar('salpha', vtype='C', lb=-1, ub=-0.98)
  #tx = model.addVar('tx', vtype='C', lb=-0.01, ub=0.01)
  #ty = model.addVar('ty', vtype='C', lb=-0.01, ub=0.01)
  #t = model.addVar('t', vtype='C', lb=-2, ub=2)

  constraints = []
  datas = []
  cons = []
  x = (theta_q, theta_p, phi_q, phi_p, alpha, tx, ty, t)
  for j in range(len(ps)):
    for i in range(len(q)):
      model.addCons(t - containment_determinant3(x, q, ps, i, j, 0) >= 0)
      
  model.setObjective(t)
  model.optimize()
  sol = model.getBestSol()

  theta_q = sol[theta_q]
  theta_p = sol[theta_p]
  phi_q = sol[phi_q]
  phi_p = sol[phi_p]
  alpha = sol[alpha]
  tx = sol[tx]
  ty = sol[ty]
  t = sol[t]
  x = (theta_q, theta_p, phi_q, phi_p, alpha, tx, ty, t)
  theta_q = get_angle3(theta_q)
  phi_q = get_angle3(phi_q)
  theta_p = get_angle3(theta_p)
  phi_p = get_angle3(phi_p)

  contains = test_containment(ps, [theta_q, math.cos(phi_q)], [theta_p, math.cos(phi_p)])
  return x

def optimize2(q, ps, x0=None): # q is a silhouette of p
# variables:
# q: theta_q, phi_q
# p: theta_p, phi_p, alpha

# minimize t
# t, alpha, theta_q, phi_q, theta_p, phi_p
# st R * M * p is inside M * q


# replace sin(a), cos(a) by 2*t/(1+t^2), (1-t^2)/(1+t^2)

  model = Model('Example')
  lb = -1
  ub = 1
  st_q = model.addVar('st_q', vtype='C', lb=lb, ub=ub)
  ct_q = model.addVar('ct_q', vtype='C', lb=lb, ub=ub)
  st_p = model.addVar('st_p', vtype='C', lb=lb, ub=ub)
  ct_p = model.addVar('ct_p', vtype='C', lb=lb, ub=ub)
  sp_q = model.addVar('sp_q', vtype='C', lb=lb, ub=ub)
  cp_q = model.addVar('cp_q', vtype='C', lb=lb, ub=ub)
  sp_p = model.addVar('sp_p', vtype='C', lb=lb, ub=ub)
  cp_p = model.addVar('cp_p', vtype='C', lb=lb, ub=ub)
  salpha = model.addVar('salpha', vtype='C', lb=lb, ub=ub)
  calpha = model.addVar('calpha', vtype='C', lb=lb, ub=ub)
  t = model.addVar('t', vtype='C', lb=-2, ub=2)

  '''
  st_q = model.addVar('st_q', vtype='C', lb=math.sin(0.5299069675048) - 1e-5, ub=math.sin(0.5299069675048) + 1e-5)
  ct_q = model.addVar('ct_q', vtype='C', lb=math.cos(0.5299069675048) - 1e-5, ub=math.cos(0.5299069675048) + 1e-5)
  st_p = model.addVar('st_p', vtype='C', lb=math.sin(0.47397226040856) - 1e-5, ub=math.sin(0.47397226040856) + 1e-5)
  ct_p = model.addVar('ct_p', vtype='C', lb=math.cos(0.47397226040856) - 1e-5, ub=math.cos(0.47397226040856) + 1e-5)
  sp_q = model.addVar('sp_q', vtype='C', lb=math.sin(math.acos(-0.653285231486307)) - 1e-5, ub=math.sin(math.acos(-0.653285231486307)) + 1e-5)
  cp_q = model.addVar('cp_q', vtype='C', lb=math.cos(math.acos(-0.653285231486307)) - 1e-5, ub=math.cos(math.acos(-0.653285231486307)) + 1e-5)
  sp_p = model.addVar('sp_p', vtype='C', lb=math.sin(math.acos(1.0)) - 1e-5, ub=math.sin(math.acos(1.0)) + 1e-5)
  cp_p = model.addVar('cp_p', vtype='C', lb=math.cos(math.acos(1.0)) - 1e-5, ub=math.cos(math.acos(1.0)) + 1e-5)
  salpha = model.addVar('salpha', vtype='C', lb=math.sin(0.83946456025974) - 1e-5, ub=math.sin(0.83946456025974) + 1e-5)
  calpha = model.addVar('calpha', vtype='C', lb=math.cos(0.83946456025974) - 1e-5, ub=math.cos(0.83946456025974) + 1e-5)
  t = 1.0
  '''

  model.addCons(st_q * st_q + ct_q * ct_q == 1)
  model.addCons(st_p * st_p + ct_p * ct_p == 1)
  model.addCons(sp_q * sp_q + cp_q * cp_q == 1)
  model.addCons(sp_p * sp_p + cp_p * cp_p == 1)
  model.addCons(salpha * salpha + calpha * calpha == 1)

  constraints = []
  datas = []
  cons = []
  x = (st_q, ct_q, st_p, ct_p, sp_q, cp_q, sp_p, cp_p, salpha, calpha, t)
  for j in range(len(ps)):
    for i in range(len(q)):
      model.addCons(t - containment_determinant(x, q, ps, i, j) >= 0)
      
  model.setObjective(t)
  model.optimize()
  sol = model.getBestSol()

  st_q = sol[st_q]
  ct_q = sol[ct_q]
  st_p = sol[st_p]
  ct_p = sol[ct_p]
  sp_q = sol[sp_q]
  cp_q = sol[cp_q]
  sp_p = sol[sp_p]
  cp_p = sol[cp_p]
  salpha = sol[salpha]
  calpha = sol[calpha]
  t = sol[t]
  x = (st_q, ct_q, st_p, ct_p, sp_q, cp_q, sp_p, cp_p, salpha, calpha, t)
  theta_q = get_angle(st_q, ct_q)
  phi_q = get_angle(sp_q, cp_q)
  theta_p = get_angle(st_p, ct_p)
  phi_p = get_angle(sp_p, cp_p)

  contains = test_containment(ps, [theta_q, math.cos(phi_q)], [theta_p, math.cos(phi_p)])
  return x

def optimize(q, ps, x0=None): # q is a silhouette of p
# variables:
# q: theta_q, phi_q
# p: theta_p, phi_p, alpha

# minimize t
# t, alpha, theta_q, phi_q, theta_p, phi_p
# st R * M * p is inside M * q


# replace sin(a), cos(a) by 2*t/(1+t^2), (1-t^2)/(1+t^2)

  model = Model('Example')
  lb = -100
  ub = 100
  theta_q = model.addVar('theta_q', vtype='C', lb=lb, ub=ub)
  theta_p = model.addVar('theta_p', vtype='C', lb=lb, ub=ub)
  phi_q = model.addVar('phi_q', vtype='C', lb=lb, ub=ub)
  phi_p = model.addVar('phi_p', vtype='C', lb=lb, ub=ub)
  alpha = model.addVar('alpha', vtype='C', lb=lb, ub=ub)
  t = model.addVar('t', vtype='C', lb=-2, ub=10)

  constraints = []
  datas = []
  cons = []
  x = (theta_q, phi_q, theta_p, phi_p, alpha, t)
  for j in range(len(ps)):
    for i in range(len(q)):
      model.addCons(constraints_fun_ij(x, q, ps, i, j) >= 0)
      
  model.setObjective(t)
  model.optimize()
  sol = model.getBestSol()

  def get_vals(y):
    return tuple(math.asin(sin_transformed(x)) for x in y)
    
  (theta_q, phi_q, theta_p, phi_p) = get_vals((sol[theta_q], sol[phi_q], sol[theta_p], sol[phi_p]))

  contains = test_containment(ps, [theta_q, math.cos(phi_q)], [theta_p, math.cos(phi_p)])
  return contains

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

def test():
  model = Model('Example')
  model.hideOutput()
  x = model.addVar('x', vtype='C', lb=2.5, ub=None)
  y = model.addVar('y', vtype='I', lb=1.5, ub=None)
  model.setObjective(x + y)
  model.addCons(2*x - y*y >= 0)
  model.addCons(x >= 1)
  model.optimize()
  sol = model.getBestSol()
  print('x: {}'.format(sol[x]))
  print('y: {}'.format(sol[y]))

def get_t(angle):
  st = math.sin(angle)
  ct = math.cos(angle)
  t1 = 1 / st + math.sqrt(1/st**2 - 1)
  t2 = 1 / st - math.sqrt(1/st**2 - 1)
  #t3 = math.sqrt((1 - ct) / (1 + ct))
  #t4 = -math.sqrt((1 - ct) / (1 + ct))

  diff1 = abs(sin_transformed(t1) - st) + abs(cos_transformed(t1) - ct)
  diff2 = abs(sin_transformed(t2) - st) + abs(cos_transformed(t2) - ct)
  return t1 if diff1 < diff2 else t2

  #print(str((st, ct)))
  #print(str((t1, t2, t3, t4)))
  #print(str((sin_transformed(t1), sin_transformed(t2), cos_transformed(t3), cos_transformed(t4))))
  return t

def test_cube():
  x0 = (get_t(4.918788-math.pi), get_t(2.0545287), get_t(0.8553414), get_t(2.108091)) # dodecahedron

  p = cube()

  contains, scaling, alpha, trans = test_containment(cube(), [0.52990696750, -0.65328523], [0.47397226, 1.0])
  print(str(alpha))
  print(str(trans))

  #x0 = (get_t(1.142397), get_t(math.acos(-0.6)), get_t(2.8559933), get_t(math.acos(-1.0))) #cube

  st_q = math.sin(0.5299069675048)
  ct_q = math.cos(0.5299069675048)
  st_p = math.sin(0.47397226040856)
  ct_p = math.cos(0.47397226040856)
  sp_q = math.sin(math.acos(-0.653285231486307))
  cp_q = math.cos(math.acos(-0.653285231486307))
  sp_p = math.sin(math.acos(1.0))
  cp_p = math.cos(math.acos(1.0))
  salpha = math.sin(0.83946456025974)
  calpha = math.cos(0.83946456025974)
  t = 1.0
  x = (st_q, ct_q, st_p, ct_p, sp_q, cp_q, sp_p, cp_p, salpha, calpha, t)

  #print(str([sin_transformed(x) for x in x0]))
  print(str(x))

  print(str(st_q**2 + ct_q**2))
  print(str(st_p**2 + ct_p**2))
  print(str(sp_q**2 + cp_q**2))
  print(str(sp_p**2 + cp_p**2))
  print(str(salpha**2 + calpha**2))

  silhouette_q = [2, 3, 7, 5, 4, 0]
  silhouette_p = [6, 4, 0, 2]

  q = [p[i] for i in silhouette_q]
  ps = [p[i] for i in silhouette_p]

  x_angles = [0.5299069675048, math.acos(-0.65328523148), 0.4739722604, math.acos(1.0), 0.8394645602]
  for j in range(len(ps)):
    for i in range(len(q)):
      print(str(containment_determinant(x, q, ps, i, j)))
      #print(str(containment_determinant_angles(x_angles, q, ps, i, j)))

def test_x(x):
  p = cube()
  (st_q, ct_q, st_p, ct_p, sp_q, cp_q, sp_p, cp_p, salpha, calpha, t) = x

  theta_q = get_angle(st_q, ct_q)
  phi_q = get_angle(sp_q, cp_q)
  theta_p = get_angle(st_p, ct_p)
  phi_p = get_angle(sp_p, cp_p)

  print(str((theta_q, phi_q, theta_p, phi_p)))

  contains = test_containment(p, [theta_q, math.cos(phi_q)], [theta_p, math.cos(phi_p)])

  ps = p
  q = p

  silhouette_q = [2, 3, 7, 5, 4, 0]
  silhouette_p = [6, 4, 0, 2]

  q = [p[i] for i in silhouette_q]
  ps = [p[i] for i in silhouette_p]

  for j in range(len(ps)):
    for i in range(len(q)):
      print(str(containment_determinant(x, q, ps, i, j)))

def run():
  q = [(-1,-1,1), (-1,1,-1), (-1,1,1), (1,-1,-1),(1,-1,1),(1,1,-1)]

  p = snub_cube()
  p = cube()
  p = pentagonal_icositetrahedron()

  silhouettes = get_silhouettes(p)
  silhouettes = [[27, 5, 28, 13, 3, 12, 29, 4, 26, 11, 2, 10]]

  # pentagonal icositetrahedron
  silhouette_q = [2, 3, 7, 5, 4, 0]
  silhouette_p = [6, 4, 0, 2]

  # cube
  #silhouette_q = [6,7,3,1,0,4]

  q = [p[i] for i in silhouette_q]
  #q = p
  ps = p

  x = optimize2(q, ps)


  if False:
    any_contains = False
    for rni in range(1):
      for ii, s in enumerate(silhouettes):
        print('Running silhouette %s of %s' % (ii, len(silhouettes)))
        q = [p[i] for i in s]
        optimize(q, p, 1338 + rni)
        #theta_q, phi_q, theta_p, phi_p, alpha, t = r.x

        #print(str(rni) + ", " + str(ii) + ' of ' + str(len(silhouettes)) + (' * ' if any_contains else ' '), end='')
        #contains = test_containment(p, [theta_q, math.cos(phi_q)], [theta_p, math.cos(phi_p)])
        #any_contains = any_contains or contains
        ii += 1
  return x
