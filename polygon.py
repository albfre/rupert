from scipy.spatial import ConvexHull, convex_hull_plot_2d
import math
from projection import *
import matplotlib.pyplot as plt

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
  #print(str(p1.hull.vertices))
  #print(str(p2.hull.vertices))
  if contains:
    print('(t,p) = (%s, %s) contains (t,p) = (%s, %s) with scaling=%s' % (p1.theta, p1.phi, p2.theta, p2.phi, largest_scaling))
  else:
    print('No containment. %s' % largest_scaling)
  return contains, largest_scaling, p1.alpha, p1.translation

def test_containment2(polyhedron, q_angles, p_angles):
  theta_q, phi_q = q_angles
  theta_p, phi_p = p_angles
  n = 6
  theta_q = round(theta_q, n)
  phi_q = round(phi_q, n)
  theta_p = round(theta_p, n)
  phi_p = round(phi_p, n)

  points_q = project_to_plane(polyhedron, theta_q, phi_q)
  points_p = project_to_plane(polyhedron, theta_p, phi_p)
  p1 = Polygon(points_q, theta_q, phi_q)
  p2 = Polygon(points_p, theta_p, phi_p)
  contains, largest_scaling, test = p1.contains(p2)
  if contains:
    print("{:.7f}".format(p2.theta) + " & " + "{:.7f}".format(p2.phi) + " & " + "{:.7f}".format(p1.theta) + " & " + "{:.7f}".format(p1.phi) + " & " + "{:.8f}".format(largest_scaling))
    print("{:.9f}".format(p1.alpha) + " & " + "{:.9f}".format(p1.translation[0]) + " & " + "{:.9f}".format(p1.translation[1]))

    #print('(t,p) = (%s, %s) contains (t,p) = (%s, %s) with alpha=%s, translation=%s, scaling=%s' % (p1.theta, p1.phi, p2.theta, p2.phi, p1.alpha, p1.translation, largest_scaling))
  else:
    print('No containment. %s' % largest_scaling)
  return contains, largest_scaling, p1.alpha, p1.translation

def test_containment3(polyhedron, q_angles, p_angles, alpha, u, v, s = 1.0):
  '''Verify that a solution works by explicit rotation, translation, and scaling'''
  theta_q, phi_q = q_angles
  theta_p, phi_p = p_angles

  n = 6
  theta_q = round(theta_q, n)
  phi_q = round(phi_q, n)
  theta_p = round(theta_p, n)
  phi_p = round(phi_p, n)

  n = 7
  alpha = round(alpha, n)
  u = round(u, n)
  v = round(v, n)

  points_q = project_to_plane(polyhedron, theta_q, phi_q)
  points_p = project_to_plane(polyhedron, theta_p, phi_p)
  points_p = rotate(points_p, alpha)
  points_p = scale(points_p, s)
  points_p = translate(points_p, u, v)
  
  p1 = Polygon(points_q, theta_q, phi_q)
  p2 = Polygon(points_p, theta_p, phi_p)
  contains = p1.contains2(p2)

  if contains:
    print("{:.7f}".format(p2.theta) + " & " + "{:.7f}".format(p2.phi) + " & " + "{:.7f}".format(p1.theta) + " & " + "{:.7f}".format(p1.phi) + " & " + "{:.8f}".format(s))
    print("{:.8f}".format(alpha) + " & " + "{:.8f}".format(u) + " & " + "{:.8f}".format(v))

def get_front(vertices, hull, edges):
  except_hull = list(vertices - hull)
  if len(except_hull) == 0:
    return {}

  front = {except_hull[0]}

  def check_edge(v0, v1):
    if v0 not in front or v1 in front or v1 in hull:
      return False
    front.add(v1)
    return True

  change = True
  while change:
    change = False
    for v0, v1 in edges:
      if check_edge(v0, v1) or check_edge(v1, v0):
        change = True
  return front

def plot_polyhedron(edges, points, color):
  all_points = set(range(len(points)))
  q2d = Polygon(points)
  hull = set(q2d.hull.vertices)
  front = get_front(all_points, hull, edges)
  back = (all_points - front)

  visible_edges = []

  for v0, v1 in edges:
    if v0 in front or v1 in front:
      visible_edges.append((v0, v1))

  for v0, v1 in edges:
    if v0 in hull and v1 in hull:
      intersects = False

      for vis0, vis1 in visible_edges:
        if intersects:
          break
        if v0 in [vis0, vis1] or v1 in [vis0, vis1]:
          continue
        p0 = points[v0]
        p1 = points[v1]
        q0 = points[vis0]
        q1 = points[vis1]
        #p0 + alpha * p1 == q0 + beta * q1
        #p0[0] + alpha * p1[0] == q0[0] + beta * q1[0]
        #p0[1] + alpha * p1[1] == q0[1] + beta * q1[1]
        #p0[1] + p1[1] * (q0[0] + beta * q1[0] - p0[0]) / p1[0] == q0[1] + beta * q1[1]
        #p0[1] + p1[1] * q0[0] / p1[0] + (p1[1] * q1[0] / p1[0] - q1[1] ) * beta - p1[1] * p0[0] / p1[0] == q0[0]
        beta = (q0[0] + p1[1] * p0[0] / p1[0] - p0[1] - p1[1] * q0[0] / p1[0] ) / (p1[1] * q1[0] / p1[0] + q1[1])
        alpha = (q0[0] + beta * q1[0] - p0[0]) / p1[0]
        if beta >= 0 and beta <= 1 and alpha >= 0 and alpha <= 1:
          intersects = True
      if not intersects:
        visible_edges.append((v0, v1))



  for v0, v1 in edges:
    if (v0, v1) in visible_edges:
      linetype = '-'
    else:
      linetype = ':'
    x0, y0 = points[v0]
    x1, y1 = points[v1]
    plt.plot([x0, x1], [y0, y1], color + linetype)

def plot_containment(name, points, edges, q_angles, p_angles, alpha, u, v, s = 1.0):
  theta_q, phi_q = q_angles
  theta_p, phi_p = p_angles

  n = 6
  theta_q = round(theta_q, n)
  phi_q = round(phi_q, n)
  theta_p = round(theta_p, n)
  phi_p = round(phi_p, n)

  n = 7
  alpha = round(alpha, n)
  u = round(u, n)
  v = round(v, n)

  if name == 'cube':
    f = 0.69003891 #cube
  else:
    f = 1
  points_q = project_to_plane(points, theta_q, phi_q)
  points_p = project_to_plane(points, theta_p, phi_p)

  points_q = rotate(points_q, -(1-f)*alpha)
  points_p = rotate(points_p, f*alpha)

  #points_p = scale(points_p, s)
  points_p = translate(points_p, u, v)

  p0 = points_p[0]
  
  hq = ConvexHull(points_q)
  hp = ConvexHull(points_p)

  #plot_polyhedron(edges, points_q, 'r')
  plot_polyhedron(edges, points_p, 'k')

  p1 = Polygon(points_q, theta_q, phi_q)
  p2 = Polygon(points_p, theta_p, phi_p)
  contains = p1.contains2(p2)
  print('Contains: %s' % contains)

  plt.xticks([])
  plt.yticks([])

  ax = plt.gca()
  ax.axis("off")
  ax.axis('equal')
  #plt.savefig(name + '.eps', format='eps',bbox_inches='tight' )
  plt.show()

class Polygon:
  def __init__(self, points, theta=None, phi=None, phi_bar=None):
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
        self.alpha = math.atan2(vertex[1], vertex[0])
        self.translation = (vertex[2], vertex[3]) if len(vertex) > 2 else (0.0, 0.0)

    return math.sqrt(largest_scaling_squared)

  def contains2(self, other):
    for i in range(len(self.vertex_points)):
      x1, y1 = self.vertex_points[i]
      x2, y2 = self.vertex_points[(i + 1) % len(self.vertex_points)]
      
      for x, y in other.vertex_points:
        if (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1) > 0: return False

    return True

    for a, b, offset in self.hull.equations: # j
      assert(abs(offset) > 1e-9)
      for x, y in other.vertex_points: # i
        if (a * x + b * y)/offset > 1:
          print('%s * %s + %s * %s = %s < %s' % (a, x, b, y, (a * x + b * y), offset))
          print(str(self.hull.equations))
          return False
    return True
    

  def contains(self, other):
    self.alpha = 0.0
    self.translation = (0.0, 0.0)
    test = [self.perimeter < other.perimeter, self.diameter < other.diameter, self.area < other.area, False]
    if self.perimeter < other.perimeter: return False, 0.0, test
    if self.diameter < other.diameter: return False, 0.0, test
    if self.area < other.area: return False, 0.0, test

    if self.box and not self.box.contains(other):
      test[3] = True
      return False, 0.0, test

    largest_scaling = self.compute_largest_scaling(other) 
    return largest_scaling > 1.0 + 1e-14, largest_scaling, test
