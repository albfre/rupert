import math
import time
from multiprocessing import Pool
from functools import partial
from polyhedra import *
from polygon import *
from projection import *

def brute_force_inner(q, p):
  best_q, best_p = None, None
  max_scaling = 0.0
  contains, largest_scaling, test = q.contains(p)
  if contains:
    if largest_scaling > max_scaling:
      max_scaling = largest_scaling
      best_q, best_p = q, p
  return max_scaling, best_q, best_p, test, test_unique

def bruteforce(q_polygons, p_polygons):
  max_scaling = 1.0
  best_q, best_p = None, None

  def update_best(max_scaling_r, best_q_r, best_p_r):
    if max_scaling_r > max_scaling:
      max_scaling= max_scaling_r
      best_q, best_p = best_q_r, best_p_r
      print('(t,p) = (%s, %s) contains (t,p) = (%s, %s) with scaling=%s' % (best_q.theta, best_q.phi_bar, best_p.theta, best_p.phi_bar, max_scaling))

  if True:
    with Pool() as pool:
      for qi, q in enumerate(q_polygons):
        print('Testing polygon for q number ' + str(qi + 1) + ' of ' + str(len(q_polygons)) + ', ' + str(max_scaling), end='\r')
        results = pool.map(partial(brute_force_inner, q), p_polygons)
        max_scaling_inner = 0.0
        for max_scaling_r, best_q_r, best_p_r, test_r, test_unique_r in results:
          update_best(max_scaling_r, best_q_r, best_p_r)
  else:
    for qi, q in enumerate(q_polygons):
      print('Testing polygon for q number ' + str(qi + 1) + ' of ' + str(len(q_polygons)) + ', ' + str(max_scaling), end='\r')
      for pj, p in enumerate(p_polygons):
        max_scaling_r, best_q_r, best_p_r, test_r, test_unique_r = brute_force_inner(q, p)
        update_best(max_scaling_r, best_q_r, best_p_r)

  print('')

  if max_scaling > 1.0 and best_q and best_p:
    q, p = best_q, best_p
    print('(t,p) = (%s, %s) contains (t,p) = (%s, %s) with scaling=%s' % (q.theta, q.phi_bar, p.theta, p.phi_bar, max_scaling))

  return best_q, best_p, max_scaling

def search_sphere(polyhedron, n, max_factor = 1.0):
  polygons = []
  n_theta = round(max_factor * n)
  for i in range(n_theta):
    print('Creating polytope for theta number %s of %s' % (i + 1, n_theta), end='\r')
    theta = (max_factor * 2 * math.pi * i) / n_theta
    for j in range(n):
      phi_bar = -1 + 2 * j / (n - 1.0)
      phi = math.acos(phi_bar)
      points_2d = project_to_plane(polyhedron, theta, phi)
      polygons.append(Polygon(points_2d, theta, phi, phi_bar))
  p_polygons = polygons
  q_polygons = polygons
  print('')
  return bruteforce(q_polygons, p_polygons)

def run():
  c = random_polytope(20)
  c = icosahedron()
  c = truncated_tetrahedron()
  c = truncated_cube()
  c = tetrahedron()
  c = truncated_octahedron()
  c = snub_cube()
  c = rhombicosidodecahedron()
  c = truncated_icosahedron()
  c = truncated_dodecahedron()
  c = cube()
  c = dodecahedron()

  c = icosidodecahedron()
  c = truncated_icosidodecahedron()
  c = truncated_cuboctahedron()
  c = truncated_icosidodecahedron()

  c = snub_cube() # no found for n = 101
  c = rhombicosidodecahedron()
  #c = snub_dodecahedron()

  c = rhombicuboctahedron()
  c = read_file('Catalan/01TriakisTetrahedron.txt')[1]
  c = read_file('Catalan/07LpentagonalIcositetrahedron.txt')[1] # no found for 31
  c = read_file('Catalan/11DeltoidalHexecontahedron.txt')[1] # no found for 31

  qa = [1.39437813, math.cos(0.037334784167499)]
  pa = [1.6361904311, math.cos(-0.37044476646)]
  qa = [4.6290895915288, 0.078712875009]
  pa = [5.8170826169, -0.102959535816]
  n = 11
  t = time.time()
  qa = [2.40587132115149, 0.649014357100288]
  pa =[5.071119649324, -8.998662569e-5]
  q, p, max_scaling = search_sphere(cube(), n)

  c = read_file('Catalan/07LpentagonalIcositetrahedron.txt')[1]
  c = truncated_tetrahedron()

  if True and q and p:
    grid = 1
    n = 11
    best_q, best_p = q, p
    while grid > 1e-8:
      grid *= 0.1
      improvement = True
      print('grid: %s' % grid)
      while improvement:
        [q, p, max_scaling2] = search_around_point(c, n, [best_q.theta, best_q.phi_bar], [best_p.theta, best_p.phi_bar], grid)
        improvement = max_scaling2 > max_scaling
        if improvement:
          max_scaling = max_scaling2
          best_q, best_p = q, p

  d = time.time() - t
  print('Time: %s' % d)
