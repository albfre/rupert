from scipy.spatial import ConvexHull, convex_hull_plot_2d
import math

def hyperplanes_to_dual_points(equations):
  return [[x/equation[-1] for x in equation[:-1]] for equation in equations]

def dual_polytope(polytope):
  equations = ConvexHull(polytope).equations
  p = [tuple(x) for x in hyperplanes_to_dual_points(equations)]
  p = [list(x) for x in list(set(p))]
  return p

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
  return [[ca * p[0] - sa * p[1], sa * p[0] + ca * p[1]] for p in points]

def scale(points, s):
  return [[s * x, s * y] for x,y in points]

def translate(points, u, v):
  return [[x + u, y + v] for x,y in points]

def get_silhouette(polyhedron, theta, phi):
  points_2d = project_to_plane(polyhedron, theta, phi)
  hull = ConvexHull(points_2d)
  vertices = list(hull._vertices)
  min_vertex = min(vertices)
  min_index = vertices.index(min_vertex)
  vertices = vertices[min_index:] + vertices[:min_index]
  return tuple(vertices)

def get_silhouettes(polyhedron, n = 100):
  silhouettes = []
  for i in range(n):
    theta = (2 * math.pi * i) / n
    for j in range(n):
      phi_bar = -1 + 2 * j / (n - 1.0)
      phi = math.acos(phi_bar)
      silhouettes.append(get_silhouette(polyhedron, theta, phi))
  return sorted(list(set(silhouettes)))

def plot(points, hull):
  plt.plot(points[:,0], points[:,1], 'o')
  for simplex in hull.simplices:
    plt.plot(points[simplex, 0], points[simplex, 1], 'k-')

  plt.show()


def dist2(a, b):
  d = 0.0
  for x, y in zip(a, b):
    d += (x - y) * (x - y)
  return math.sqrt(d)

def expand(c, alpha):
  hull = ConvexHull(c)
  points = []
  for simplex, equation in zip(hull.simplices, hull.equations):
    offset = equation[-1]
    normal = [x / offset for x in equation[0:3]]
    for point in [list(c[s]) for s in simplex]:
      for i in range(len(point)):
        point[i] += alpha * normal[i]
      points.append(point)
  filtered = []
  for i in range(len(points)):
    include = True
    for j in range(i + 1, len(points)):
      if dist2(points[i], points[j]) < 1e-6:
        include = False
        break
    if include:
      filtered.append(points[i])
  return filtered
