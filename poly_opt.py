from scipy.spatial import ConvexHull, convex_hull_plot_2d
import matplotlib.pyplot as plt
import numpy as np
import math
import itertools
import time
from multiprocessing import Pool
from functools import partial
from scipy.optimize import minimize

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

def rotate(points, alpha):
  ca = math.cos(alpha)
  sa = math.sin(alpha)
  return [[ca * p[0] - sa * p[1],   sa * p[0] + ca * p[1]] for p in points]

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

def rhombicosidodecahedron():
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


def test_single(polyhedron, q_angles, p_angles):
  theta_q, phi_bar_q = q_angles
  theta_p, phi_bar_p = p_angles
  phi_q = math.acos(phi_bar_q)
  phi_p = math.acos(phi_bar_p)

  points_q = project_to_plane(polyhedron, theta_q, phi_q)
  points_p = project_to_plane(polyhedron, theta_p, phi_p)
  p1 = Polygon(points_q, theta_q, phi_q, phi_bar_q)
  p2 = Polygon(points_p, theta_p, phi_p, phi_bar_p)
  contains, largest_scaling, test = p1.contains(p2)
  if contains:
    print('(t,p) = (%s, %s) contains (t,p) = (%s, %s) with scaling=%s' % (p1.theta, p1.phi_bar, p2.theta, p2.phi_bar, largest_scaling))
  else:
    print('No containment')
  return contains

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

def optimize(q, ps): # q is a silhouette of p
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

  rng = np.random.default_rng(1338)
  x0 = list(rng.random(6))
  x0[-1] = 10

  res = minimize(objective, x0, method='SLSQP', jac=objective_grad, constraints=constraints)
  #constraints = [{'type':'ineq', 'fun': constr_test}]
  #res = minimize(obj, x0, method='SLSQP', jac=obj_grad, constraints=constraints)
  #for c in constraints:
  #  print(str(c['fun'](r.x)))
  return res, constraints

def get_silhouettes(polyhedron):
  n = 100
  silhouettes = []
  for i in range(n):
    theta = (2 * math.pi * i) / n
    for j in range(n):
      phi_bar = -1 + 2 * j / (n - 1.0)
      phi = math.acos(phi_bar)
      points_2d = project_to_plane(polyhedron, theta, phi)
      hull = ConvexHull(points_2d)
      vertices = list(hull._vertices)
      min_vertex = min(vertices)
      min_index = vertices.index(min_vertex)
      vertices = vertices[min_index:] + vertices[:min_index]
      silhouettes.append(tuple(vertices))

  return sorted(list(set(silhouettes)))

  

def run():
  q = [(-1,-1,1), (-1,1,-1), (-1,1,1), (1,-1,-1),(1,-1,1),(1,1,-1)]
  p = cube()

  p = truncated_dodecahedron()
  silhouettes = get_silhouettes(p)
  ii = 0
  any_contains = False
  for s in silhouettes:
    q = [p[i] for i in s]
    r, constraints = optimize(q, p)
    theta_q, phi_q, theta_p, phi_p, alpha, t = r.x

    print(str(ii) + ' of ' + str(len(silhouettes)) + (' * ' if any_contains else ' '), end='')
    contains = test_single(p, [theta_q, math.cos(phi_q)], [theta_p, math.cos(phi_p)])
    any_contains = any_contains or contains
    ii += 1
  return r, constraints
