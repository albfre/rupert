from scipy.spatial import ConvexHull, convex_hull_plot_2d
import matplotlib.pyplot as plt
import numpy as np
import math
import itertools
import time
from multiprocessing import Pool
from functools import partial

golden_ratio = (1.0 + math.sqrt(5.0)) / 2.0

def hyperplanes_to_dual_points(equations):
  return [[x/equation[-1] for x in equation[:-1]] for equation in equations]

def hyperplanes_to_vertices(equations):
  for i, equation in enumerate(equations):
    if equation[-1] <= 0.0:
      print('Error in equation %s: %s' % (i, equation))
      assert(equation[-1] > 0.0)

  dual_points = hyperplanes_to_dual_points(equations)
  dual_hull = ConvexHull(dual_points)
  primal_points = hyperplanes_to_dual_points(dual_hull.equations)
  return primal_points

def project_to_plane(points, theta, phi):
  assert(all(len(p) == 3 for p in points))
  st = math.sin(theta)
  ct = math.cos(theta)
  sp = math.sin(phi)
  cp = math.cos(phi)
  return [[-st * p[0] + ct * p[1], -ct * cp * p[0] - st * cp * p[1] + sp * p[2]] for p in points]

def random(n):
  rng = np.random.default_rng(1338)
  points = rng.random((n, 3))
  points = [[p[0]-0.5, p[1]-0.5, p[2]-0.5] for p in points]
  return points

def even_permutation(point):
  assert(len(point) == 3)
  permutation_points = []
  for i in range(len(point)):
    p = point[i:] + point[:i]
    permutation_points.append(list(p))
  return permutation_points

def even_permutations(points):
  permutation_points = []
  for p in points:
    permutation_points += even_permutation(p)
  return sorted(permutation_points)

def odd_permutation(point):
  assert(len(point) == 3)
  all_permutations = [list(p) for p in itertools.permutations(point)]
  return sorted([p for p in all_permutations if p not in even_permutation(point)])

def odd_permutations(points):
  permutation_points = []
  for p in points:
    permutation_points += odd_permutation(p)
  return sorted(permutation_points)

def all_permutations(points):
  permutation_points = []
  for p in points:
    permutation_points += itertools.permutations(p)
  return [list(p) for p in sorted(list(set(permutation_points)))]

def has_even_number_of_minus_signs(p):
  return sum(1 for r in p if r < 0.0) % 2 == 0

def has_odd_number_of_minus_signs(p):
  return sum(1 for r in p if r < 0.0) % 2 == 1

def tetrahedron():
  return [[-1,-1,1],[-1,1,-1],[1,-1,-1],[1,1,1]]

def cube():
  return list(itertools.product((-1,1), (-1,1), (-1,1)))

def octahedron():
  return all_permutations(list(itertools.product((0,),(0,),(-1,1))))

def dodecahedron():
  r = list(itertools.product((-1,1), (-1,1), (-1,1)))
  r += even_permutations(list(itertools.product((0,), (-1/golden_ratio,1/golden_ratio), (-golden_ratio,golden_ratio))))
  return r

def icosahedron():
  return even_permutations(list(itertools.product((0,),(-1,1),(-golden_ratio,golden_ratio))))

def truncated_tetrahedron():
  return [p for p in all_permutations(list(itertools.product((-1,1),(-1,1),(-3,3)))) if has_even_number_of_minus_signs(p)]

def cuboctahedron():
  return all_permutations(list(itertools.product((-1,1),(-1,1),(0,))))

def truncated_cube():
  p = math.sqrt(2.0) - 1.0
  return all_permutations(list(itertools.product((-1,1),(-1,1),(-p,p))))
  
def truncated_octahedron():
  return all_permutations(list(itertools.product((0,),(-1,1),(-2,2))))

def rhombicuboctahedron():
  p = math.sqrt(2.0) + 1.0
  return all_permutations(list(itertools.product((-1,1),(-1,1),(-p,p))))

def truncated_cuboctahedron():
  p = math.sqrt(2.0)
  return all_permutations(list(itertools.product((-1,1),(-(1+p),1+p),(-(1+2*p),1+2*p))))
  
def snub_cube():
  t = 1.839286755214161132551851564
  r = []
  r += [p for p in even_permutations(list(itertools.product((-1,1),(-1/t,1/t),(-t,t)))) if has_odd_number_of_minus_signs(p)]
  r += [p for p in odd_permutations(list(itertools.product((-1,1),(-1/t,1/t),(-t,t)))) if has_even_number_of_minus_signs(p)]
  return r

def icosidodecahedron():
  r = []
  r += all_permutations(list(itertools.product((0,),(0,),(-golden_ratio,golden_ratio))))
  r += even_permutations(list(itertools.product((-0.5,0.5),(-golden_ratio/2,golden_ratio/2),(-golden_ratio**2/2,golden_ratio**2/2))))
  return r

def truncated_dodecahedron():
  r = []
  r += even_permutations(list(itertools.product((0,),(-1/golden_ratio,1/golden_ratio),(-(2+golden_ratio),2+golden_ratio))))
  r += even_permutations(list(itertools.product((-1/golden_ratio,1/golden_ratio),(-golden_ratio,golden_ratio),(-2*golden_ratio,2*golden_ratio))))
  r += even_permutations(list(itertools.product((-golden_ratio,golden_ratio),(-2,2),(-(golden_ratio+1),golden_ratio+1))))
  return r

def truncated_icosahedron():
  g = (1 + math.sqrt(5))/2.0
  r = []
  r += odd_permutations(list(itertools.product((0,), (-1,1), (-3*g, 3*g))))
  r += odd_permutations(list(itertools.product((-1,1), (-(2+g), 2+g), (-2*g, 2*g))))
  r += odd_permutations(list(itertools.product((-g, g), (-2,2), (-(2*g+1), 2*g+1))))
  return r

def rombicosidodecahedron():
  r = []
  r += even_permutations(list(itertools.product((-1,1), (-1,1), (-golden_ratio**3, golden_ratio**3))))
  r += even_permutations(list(itertools.product((-golden_ratio**2,golden_ratio**2), (-golden_ratio,golden_ratio), (-2*golden_ratio, 2*golden_ratio))))
  r += even_permutations(list(itertools.product((-(2+golden_ratio),2+golden_ratio), (0,), (-golden_ratio**2, golden_ratio**2))))
  return r

def truncated_icosidodecahedron():
  r = []
  r += even_permutations(list(itertools.product((-1/golden_ratio,1/golden_ratio), (-1/golden_ratio,1/golden_ratio), (-(3+golden_ratio), 3+golden_ratio))))
  r += even_permutations(list(itertools.product((-2/golden_ratio,2/golden_ratio), (-golden_ratio,golden_ratio), (-(1+2*golden_ratio), 1+2*golden_ratio))))
  r += even_permutations(list(itertools.product((-1/golden_ratio,1/golden_ratio), (-golden_ratio**2,golden_ratio**2), (-(-1+3*golden_ratio), -1+3*golden_ratio))))
  r += even_permutations(list(itertools.product((-(2*golden_ratio-1),2*golden_ratio-1), (-2,2), (-(2+golden_ratio), 2+golden_ratio))))
  r += even_permutations(list(itertools.product((-golden_ratio,golden_ratio), (-3,3), (-2*golden_ratio, 2*golden_ratio))))
  return r


def plot(points, hull):
  plt.plot(points[:,0], points[:,1], 'o')
  for simplex in hull.simplices:
    plt.plot(points[simplex, 0], points[simplex, 1], 'k-')

  plt.show()

def distance(p1, p2):
  return math.sqrt(sum((x-y)**2 for x, y in zip(p1, p2)))

def diameter(points):
  max_distance = 0.0
  for i, p1 in enumerate(points):
    for j in range(i + 1, len(points)):
      p2 = points[j]
      max_distance = max(max_distance, distance(p1, p2))
  return max_distance

def area(points):
  n = len(points)
  c = [sum(p[0] for p in points) / n, sum(p[1] for p in points) / n]
  area = 0.0
  for i in range(n):
    a = points[i]
    b = points[i + 1 if i + 1 < n else 0]
    area += abs(a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1])) / 2.0
  return area

def box(points):
  min_point = list(points[0])
  max_point = list(points[0])
  for p in points:
    for i in range(len(p)):
      min_point[i] = min(min_point[i], p[i] - 1e-9)
      max_point[i] = max(max_point[i], p[i] + 1e-9)
  return [min_point, max_point, [min_point[0], max_point[1]], [max_point[0], min_point[1]]]

class Polygon:
  def __init__(self, points, theta, phi, phi_bar):
    assert(all(len(p) == 2 for p in points))
    self.points = points
    self.theta = theta
    self.phi = phi
    self.phi_bar = phi_bar

    self.hull = ConvexHull(points)

    self.vertex_points = [points[i] for i in self.hull.vertices]
    self.perimeter = sum(distance(points[simplex[0]], points[simplex[1]]) for simplex in self.hull.simplices)
    self.diameter = diameter(self.vertex_points)
    self.area = area(self.vertex_points)
    self.box = None if len(self.vertex_points) <= 16 else Polygon(box(self.vertex_points), theta, phi, phi_bar)
    self.largest_vertex = None

  def compute_largest_scaling(self, other):
    equations = [] # Lij
    for a, b, offset in self.hull.equations: # j
      assert(abs(offset) > 1e-9)
      for x, y in other.vertex_points: # i
        equations.append([(a * x + b * y) / offset, (-a * y + b * x) / offset, a / offset, b / offset, 1.0])

    vertices = hyperplanes_to_vertices(equations)

    largest_scaling_squared = 0.0 #max(v[0]**2 + v[1]**2 for v in vertices)
    for vertex in vertices:
      scaling_squared = vertex[0]**2 + vertex[1]**2
      if scaling_squared > largest_scaling_squared:
        self.largest_vertex = vertex
        largest_scaling_squared = scaling_squared

    return math.sqrt(largest_scaling_squared)

  def contains(self, other):
    test = [self.perimeter < other.perimeter, self.diameter < other.diameter, self.area < other.area, False]
    if self.perimeter < other.perimeter: return False, 0.0, test
    if self.diameter < other.diameter: return False, 0.0, test
    if self.area < other.area: return False, 0.0, test

    if self.box and not self.box.contains(other):
      test[3] = True
      return False, 0.0, test

    largest_scaling = self.compute_largest_scaling(other) 
    return largest_scaling > 1.0 + 1e-14, largest_scaling, test

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

  if max_scaling > 0.0:
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

def test_single(polyhedron, q, p):
  theta_q, phi_bar_q = q
  theta_p, phi_bar_p = p
  phi_q = math.acos(phi_bar_q)
  phi_p = math.acos(phi_bar_p)

  points_q = project_to_plane(polyhedron, theta_q, phi_q)
  points_p = project_to_plane(polyhedron, theta_p, phi_p)
  p1 = Polygon(points_q, theta_q, phi_q, phi_bar_q)
  p2 = Polygon(points_p, theta_p, phi_p, phi_bar_p)
  if p1.contains(p2):
    print('(t,p) = (%s, %s) contains (t,p) = (%s, %s) with scaling=%s' % (p1.theta, p1.phi_bar, p2.theta, p2.phi_bar, p1.largest_scaling))
  else:
    print('No containment')

def run():
  c = dodecahedron()
  c = random(20)
  c = icosahedron()
  c = truncated_tetrahedron()
  c = truncated_cube()
  c = tetrahedron()
  c = truncated_icosahedron()
  c = truncated_octahedron()
  c = snub_cube()
  c = cuboctahedron()
  c = rombicosidodecahedron()
  c = cube()

  n = 21
  t = time.time()
  q, p, max_scaling = search_sphere(c, n)
  #q, p, max_scaling = search_around_point(truncated_icosahedron, n, [0.0023, -0.2542333], [0.32158333, -0.5797303], 1e-2)

  #test_single(truncated_icosahedron(), [1.5698707894615636, 0.24898946607017533], [1.2501992987692, -0.578614034754386]) # 1.00195696
  #test_single(cuboctahedron(),[2.815909397, 0.30473783], [4.712388979, 0.70710678]) # 1.014611


  #test_single(tetrahedron(), [0.7801554885282173, -0.5793576087575756], [1.5707963257948965, 0.5773572977575758]) # 1.014610373
  #test_single(dodecahedron(), [4.91885536, -0.4651241815], [0.85533109966, -0.51181376779]) # 1.010822219108
  #test_single(dodecahedron(), [4.918788, math.cos(2.0545287)], [0.8553414, math.cos(2.108091)]) # 1.010818 from paper
  #test_single(icosahedron(), [0.73701686138, -0.64815309888], [0, -0.352054176666]) # 1.010822219108 #1.01082280359
  #test_single(truncated_tetrahedron(),[0.3424291073589, 0.843452253487], [3.3333333e-10, 0.707106781]) # 1.014255728
  #test_single(truncated_cube(),[0.785398165, -0.372140328948], [0.76768607073, -1.0]) # 1.0306624
  #test_single(truncated_octahedron(),[2.815909397, 0.30473783], [4.712388979, 0.70710678]) # 1.014611

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
