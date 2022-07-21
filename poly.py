from scipy.spatial import ConvexHull, convex_hull_plot_2d
import matplotlib.pyplot as plt
import numpy as np
import math
import itertools
import time

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

def tetrahedron():
  return [[math.sqrt(8/9), 0, -1/3], [-math.sqrt(2/9), math.sqrt(2/3), -1/3], [-math.sqrt(2/9), -math.sqrt(2/3), -1/3], [0,0,1]]

def cube():
  return list(itertools.product((-1,1), (-1,1), (-1,1)))

def rombicosidodecahedron():
  g = (1 + math.sqrt(5))/2.0
  r = []
  r += list(itertools.product((-1,1), (-1,1), (-g**3, g**3)))
  r += list(itertools.product((-1,1), (-g**3, g**3), (-1,1)))
  r += list(itertools.product((-g**3, g**3), (-1,1), (-1,1)))

  r += list(itertools.product((-g**2, g**2), (-g,g), (-2*g,2*g)))
  r += list(itertools.product((-2*g, 2*g), (-g**2, g**2), (-g,g)))
  r += list(itertools.product((-g, g), (-2*g, 2*g), (-g**2,g**2)))

  r += list(itertools.product((-(2+g), 2+g), (0,), (-g**2, g**2)))
  r += list(itertools.product((-g**2, g**2), (-(2+g),2+g), (0,)))
  r += list(itertools.product((0,), (-g**2,g**2), (-(2+g), 2+g)))
  return r

def truncated_icosahedron():
  g = (1 + math.sqrt(5))/2.0
  r = []
  r += list(itertools.product((0,), (-1,1), (-3*g, 3*g)))
  r += list(itertools.product((-3*g, 3*g), (0,), (-1,1)))
  r += list(itertools.product((-1,1), (-3*g, 3*g), (0,)))

  r += list(itertools.product((-1,1), (-(2+g), 2+g), (-2*g, 2*g)))
  r += list(itertools.product((-2*g, 2*g), (-1,1), (-(2+g), 2+g)))
  r += list(itertools.product((-(2+g), 2+g), (-2*g, 2*g), (-1,1)))

  r += list(itertools.product((-g, g), (-2,2), (-(2*g+1), 2*g+1)))
  r += list(itertools.product((-(2*g+1), 2*g+1), (-g, g), (-2,2)))
  r += list(itertools.product((-2,2), (-(2*g+1), 2*g+1), (-g, g)))
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
    self.largest_scaling = 1.0

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

    self.largest_scaling = math.sqrt(largest_scaling_squared)
    return self.largest_scaling

  def contains(self, other):
    self.test = [self.perimeter < other.perimeter, self.diameter < other.diameter, self.area < other.area, False]
    if self.perimeter < other.perimeter: return False
    if self.diameter < other.diameter: return False
    if self.area < other.area: return False

    if self.box and not self.box.contains(other):
      self.test[3] = True
      return False

    return self.compute_largest_scaling(other) > 1.0 + 1e-9

def bruteforce(polyhedron, n, q_poly = None, p_poly = None, grid_size = None):
  assert((q_poly and p_poly and grid_size) or (not q_poly and not p_poly and not grid_size))

  polygons = []
  n_half = round(n/2.0)
  if q_poly:
    for qp in [q_poly, p_poly]:
      for i in range(n_half):
        theta = min(max(qp[0] - grid_size + 2.0 * grid_size * i / n_half, 0), 2 * math.pi)
        for j in range(n_half):
          phi_bar = min(max(qp[1] - grid_size + 2.0 * grid_size * j / n_half, -1.0), 1.0)
          phi = math.acos(phi_bar)
          points_2d = project_to_plane(polyhedron, theta, phi)
          polygons.append(Polygon(points_2d, theta, phi, phi_bar))
  else:
    for i in range(n):
      theta = (2 * math.pi * i) / n
      #for j in range(n):
      #  phi_bar = -1 + 2 * j / (n - 1.0)
      for j in range(n_half):
        phi_bar = -1 + 2 * j / (n - 1.0)
        phi = math.acos(phi_bar)
        points_2d = project_to_plane(polyhedron, theta, phi)
        polygons.append(Polygon(points_2d, theta, phi, phi_bar))

  max_scaling = 0.0
  found = False
  test = [0, 0, 0, 0]
  test_unique = [0, 0, 0, 0]
  n_tests = 0
  for i, p1 in enumerate(polygons):
    print(str(i) + ' of ' + str(len(polygons)) + (' ' + str(best_p1.largest_scaling))  if found else '')
    for j, p2 in enumerate(polygons):
      if i == j: continue
      n_tests += 1
      if p1.contains(p2):
        if p1.largest_scaling > max_scaling:
          found = True
          max_scaling = p1.largest_scaling
          best_p1 = p1
          best_p2 = p2
          print('(t,p) = (%s, %s) contains (t,p) = (%s, %s) with scaling=%s' % (p1.theta, p1.phi_bar, p2.theta, p2.phi_bar, p1.largest_scaling))
      else:
        test_unique[0] += 1 if p1.test[0] and not p1.test[1] and not p1.test[2] else 0
        test_unique[1] += 1 if p1.test[1] and not p1.test[0] and not p1.test[2] else 0
        test_unique[2] += 1 if p1.test[2] and not p1.test[0] and not p1.test[1] else 0
        test_unique[3] += 1 if p1.test[3] and not p1.test[0] and not p1.test[1] and not p1.test[2] else 0
        for i in range(4):
          test[i] += 1 if p1.test[i] else 0

  if found:
    p1 = best_p1
    p2 = best_p2
    p1.contains(p2)
    print('(t,p) = (%s, %s) contains (t,p) = (%s, %s) with scaling=%s' % (p1.theta, p1.phi_bar, p2.theta, p2.phi_bar, p1.largest_scaling))
  print('tests: ' + str(test) + " of: " + str(n_tests))
  print('test unique: ' + str(test_unique))

def test_single(polyhedron, q, p):
  theta_q, phi_bar_q = q
  theta_p, phi_bar_p = p
  phi_q = math.acos(phi_bar_q)
  phi_p = math.acos(phi_bar_p)

  points_2d = project_to_plane(polyhedron, theta_q, phi_q)
  p1 = Polygon(points_2d, theta_q, phi_q, phi_bar_q)
  p2 = Polygon(points_2d, theta_p, phi_p, phi_bar_p)
  if p1.contains(p2):
    print('(t,p) = (%s, %s) contains (t,p) = (%s, %s) with scaling=%s' % (p1.theta, p1.phi_bar, p2.theta, p2.phi_bar, p1.largest_scaling))
  else:
    print('No containment')

def run():
  #c = rombicosidodecahedron()
  #c = truncated_icosahedron()
  #c = cube()
  #c = tetrahedron()
  c = random(20)
  n = 10
  t = time.time()
  bruteforce(c, n)
  #bruteforce(c, 101, [0, -1.0], [4.477651349779, -0.80473785415], 0.000000001) # tetraeder

  #bruteforce(c, n, [0.4181333, -0.091333], [0.62666, 0.2499666], 0.001) # truncated icosahedron
  d = time.time() - t
  print('Time: %s' % d)



if False:
  plt.plot(points[:,0], points[:,1], 'o')
  for simplex in hull.simplices:
    plt.plot(points[simplex, 0], points[simplex, 1], 'k-')

  plt.show()
