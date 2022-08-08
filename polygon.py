from scipy.spatial import ConvexHull, convex_hull_plot_2d
import math
from projection import *

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

def test_containment(polyhedron, q_angles, p_angles):
  theta_q, phi_bar_q = q_angles
  theta_p, phi_bar_p = p_angles
  phi_q = math.acos(phi_bar_q)
  phi_p = math.acos(phi_bar_p)

  points_q = project_to_plane(polyhedron, theta_q, phi_q)
  points_p = project_to_plane(polyhedron, theta_p, phi_p)
  p1 = Polygon(points_q, theta_q, phi_q, phi_bar_q)
  p2 = Polygon(points_p, theta_p, phi_p, phi_bar_p)
  contains, largest_scaling, test = p1.contains(p2)
  print(str(p1.hull.vertices))
  if contains:
    print('(t,p) = (%s, %s) contains (t,p) = (%s, %s) with scaling=%s' % (p1.theta, p1.phi_bar, p2.theta, p2.phi_bar, largest_scaling))
  else:
    print('No containment. %s' % largest_scaling)
  return contains, largest_scaling

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
    self.is_centrally_symmetric = True

    if True:
      for point in self.vertex_points:
        minus_point = [-point[0], -point[1]]
        if not any(distance(minus_point, p) < 1e-5 for p in self.vertex_points):
          self.is_centrally_symmetric = False
          break

  def compute_largest_scaling(self, other):
    equations = [] # Lij
    for a, b, offset in self.hull.equations: # j
      assert(abs(offset) > 1e-9)
      for x, y in other.vertex_points: # i
        if self.is_centrally_symmetric and other.is_centrally_symmetric:
          equations.append([(a * x + b * y) / offset, (-a * y + b * x) / offset, 1.0])
        else:
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
