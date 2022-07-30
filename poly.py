from scipy.spatial import ConvexHull, convex_hull_plot_2d
import matplotlib.pyplot as plt
import numpy as np
import math
import itertools
import time
from multiprocessing import Pool
from functools import partial
from polytopes import *
from polygon import *
from projection import *

def brute_force_inner(q, p):
  best_q, best_p = None, None
  test = [0, 0, 0, 0]
  test_unique = [0, 0, 0, 0]
  max_scaling = 0.0
  contains, largest_scaling, test = q.contains(p)
  if contains:
    if largest_scaling > max_scaling:
      max_scaling = largest_scaling
      best_q, best_p = q, p
  else:
    for i in range(4):
      test_unique[i] += 1 if test[i] and all(j == i or not test[j] for j in range(4)) else 0
      test[i] += 1 if test[i] else 0
  return max_scaling, best_q, best_p, test, test_unique

def bruteforce(q_polygons, p_polygons):
  max_scaling = 1.0
  best_q, best_p = None, None
  test = [0, 0, 0, 0]
  test_unique = [0, 0, 0, 0]
  n_tests = 0
  if True:
    with Pool() as pool:
      for qi, q in enumerate(q_polygons):
        print('Testing polygon for q number ' + str(qi + 1) + ' of ' + str(len(q_polygons)) + ', ' + str(max_scaling), end='\r')
        n_tests += 1
        result = pool.map(partial(brute_force_inner, q), p_polygons)
        max_scaling_inner = 0.0
        for max_scaling_r, best_q_r, best_p_r, test_r, test_unique_r in result:
          if max_scaling_r > max_scaling:
            max_scaling= max_scaling_r
            best_q, best_p = best_q_r, best_p_r
            print('(t,p) = (%s, %s) contains (t,p) = (%s, %s) with scaling=%s' % (best_q.theta, best_q.phi_bar, best_p.theta, best_p.phi_bar, max_scaling))
          for i in range(len(test)):
            test[i] += test_r[i]
            test_unique[i] += test_unique_r[i]
  else:
    for qi, q in enumerate(q_polygons):
      print('Testing polygon for q number ' + str(qi + 1) + ' of ' + str(len(q_polygons)) + ', ' + str(max_scaling), end='\r')
      for pj, p in enumerate(p_polygons):
     #   if pj % 10 == 0:
     #     print('Testing polygon for q number ' + str(qi + 1) + ', ' + str(pj + 1) + ' of ' + str(len(q_polygons)) + ', ' + str(max_scaling), end='\r')
        n_tests += 1
        contains, largest_scaling, test = q.contains(p)
        if contains:
          if largest_scaling > max_scaling:
            max_scaling = largest_scaling
            best_q, best_p = q, p
            print('(t,p) = (%s, %s) contains (t,p) = (%s, %s) with scaling=%s' % (q.theta, q.phi_bar, p.theta, p.phi_bar, max_scaling))
        else:
          for i in range(4):
            test_unique[i] += 1 if test[i] and all(j == i or not test[j] for j in range(4)) else 0
            test[i] += 1 if test[i] else 0
  print('')

  if max_scaling > 1.0 and best_q and best_p:
    q, p = best_q, best_p
    print('(t,p) = (%s, %s) contains (t,p) = (%s, %s) with scaling=%s' % (q.theta, q.phi_bar, p.theta, p.phi_bar, max_scaling))

  print('tests: ' + str(test) + " of: " + str(n_tests))
  print('test unique: ' + str(test_unique))
  return best_q, best_p, max_scaling


def search_sphere(polyhedron, n, max_factor = 1.0):
  polygons = []
  n_theta = round(max_factor * n)
  for i in range(n_theta):
    print('Creating polytope for theta number %s of %s' % (i + 1, n_theta), end='\r')
    theta = (max_factor * 2 * math.pi * i) / n_theta
    for j in range(n):
      phi_bar = -1 + 2 * j / (n - 1.0)
    #for j in range(n_half):
    #  phi_bar = -1 + 2 * j / (n - 1.0)
      phi = math.acos(phi_bar)
      points_2d = project_to_plane(polyhedron, theta, phi)
      polygons.append(Polygon(points_2d, theta, phi, phi_bar))
  p_polygons = polygons
  q_polygons = polygons
  print('')
  return bruteforce(q_polygons, p_polygons)

def search_around_point(polyhedron, n, q_angles, p_angles, grid_size):
  p_polygons = []
  q_polygons = []
  n_half = round(n / 2.0)
  for angles, polygons in [(q_angles, q_polygons), (p_angles, p_polygons)]:
    for i in range(n_half):
      theta = min(max(angles[0] - grid_size + 2.0 * grid_size * i / n_half, 0), 2 * math.pi)
      for j in range(n_half):
        phi_bar = min(max(angles[1] - grid_size + 2.0 * grid_size * j / n_half, -1.0), 1.0)
        phi = math.acos(phi_bar)
        points_2d = project_to_plane(polyhedron, theta, phi)
        polygons.append(Polygon(points_2d, theta, phi, phi_bar))
      theta = (2 * math.pi * i) / n
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
  c = read_file('Catalan/07LpentagonalIcositetrahedron.txt')[1]
  qa = [1.39437813, math.cos(0.037334784167499)]
  pa = [1.6361904311, math.cos(-0.37044476646)]
  qa = [4.6290895915288, 0.078712875009]
  pa = [5.8170826169, -0.102959535816]
  c = snub_cube()
  n = 11
  t = time.time()
  pa = [4.62855249619, 0.078693308675666]
  qa = [5.817518733566667, -0.102768517482666]
  #q, p, max_scaling = search_sphere(c, n)
  #q, p, max_scaling = search_around_point(c, 51, qa, pa, 1e-3)
  #q, p, max_scaling = search_around_point(truncated_icosahedron, n, [0.0023, -0.2542333], [0.32158333, -0.5797303], 1e-2)

  test_containment(snub_cube(), [0.055726718395558206, -0.4999974246334165], [3.090528105136888, 0.7408556305019884]) # 1.001084489 optimize

  if False:
    test_containment(tetrahedron(), [0.7801554885282173, -0.5793576087575756], [1.5707963257948965, 0.5773572977575758]) # 1.014610373
    test_containment(dodecahedron(), [4.91885536, -0.4651241815], [0.85533109966, -0.51181376779]) # 1.010822219108
    test_containment(dodecahedron(), [4.918788, math.cos(2.0545287)], [0.8553414, math.cos(2.108091)]) # 1.010818 from paper
    test_containment(icosahedron(), [0.73701686138, -0.64815309888], [0, -0.352054176666]) # 1.010822219108 #1.01082280359
    test_containment(truncated_tetrahedron(),[0.3424291073589, 0.843452253487], [3.3333333e-10, 0.707106781]) # 1.014255728
    test_containment(cuboctahedron(),[0.785398161, -0.57830026], [0.61548018768, 1.3864235e-17]) # 1.0146117
    test_containment(cuboctahedron(),[0.78524668297487, 0.577357842844], [0, 0.81649658429]) # 1.01461186 optimized
    test_containment(truncated_cube(),[0.785398165, -0.372140328948], [0.76768607073, -1.0]) # 1.0306624
    test_containment(truncated_octahedron(),[2.815909397, 0.30473783], [4.712388979, 0.70710678]) # 1.014611
    test_containment(rhombicuboctahedron(),[0.6217515917199, -0.143038708666], [1.05994735488, -1.0]) # 1.0128198
    test_containment(truncated_cuboctahedron(),[1.1981691280, 0.110749025], [0.300367566, -1.0]) #  1.0065959588
    # snub cube
    test_containment(icosidodecahedron(), [0.545453368289, 0.0001048230], [4.586745414, -0.3066122]) # 1.0008836
    test_containment(icosidodecahedron(), [3.14162126472, -0.850624425715], [0.49306421127, 0.918361892]) # 1.0008854 optimized
    test_containment(truncated_dodecahedron(), [5.707191649, 0.0983882599999], [0.7683968876898, -0.5922993299999]) # 1.00161296
    test_containment(truncated_icosahedron(), [1.32258347578, -0.0012256209999], [0.862237455, -0.835966629666]) # 1.0019614
    # rhombicosidodecahedron
    test_containment(truncated_icosidodecahedron(), [0.4969429976, -0.677999666], [2.20601622382, 1.0]) # 1.0020658
    # snub dodecahedron


    # New results:

    # Catalan 2
    test_containment(read_file('Catalan/01TriakisTetrahedron.txt')[1], [0.7733307863055, -0.01883940319377973], [6.275432212599668, -0.6802734306138366]) # 1.0000033 optimize
    test_containment(read_file('Catalan/07LpentagonalIcositetrahedron.txt')[1], [1.39437813, 0.037334784167499], [1.6361904311, -0.37044476646]) # 1.000244 optimize
    test_containment(read_file('Catalan/07LpentagonalIcositetrahedron.txt')[1], [4.6290895915288, 0.078712875009], [5.8170826169, -0.102959535816]) # 1.0004299 optimize
    test_containment(read_file('Catalan/07LpentagonalIcositetrahedron.txt')[1], [4.62855249619, 0.078693308675666], [5.817518733566667, -0.102768517482666]) # 1.000435 optimize


    # Johnson 4
    test_containment(read_file('Johnson/GyroelongatedPentagonalRotunda.txt')[1], [1.5697500508, 0.516259456], [3.44208101, -0.1893870555]) # J25 1.0000894999 optimize
    test_containment(read_file('Johnson/GyroelongatedSquareBicupola.txt')[1], [4.71940540634669, -0.57234816598], [3.148896509339, 0.002509670578455]) # J45 1.00000956 optimize
    test_containment(read_file('Johnson/GyroelongatedPentagonalCupolarotunda.txt')[1], [3.4528520562794, -0.42733970469], [3.4424819869, -0.19547975499]) # J47 1.00008028 optimize
    test_containment(read_file('Johnson/TriaugmentedTruncatedDodecahedron.txt')[1], [3.41783398253844, -0.9152295760], [0.789632179442, -0.00051359417]) # J71 1.000598658



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



if False:
  plt.plot(points[:,0], points[:,1], 'o')
  for simplex in hull.simplices:
    plt.plot(points[simplex, 0], points[simplex, 1], 'k-')

  plt.show()
