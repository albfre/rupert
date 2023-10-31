from scipy.spatial import ConvexHull, convex_hull_plot_2d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from polygon import *
from polyhedra import *
from projection import *

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

def plot_polyhedron(edges, points, color, linewidth=2, linestyle='-'):
  all_points = set(range(len(points)))
  q2d = Polygon(points)
  hull = set(q2d.hull.vertices)
  front = get_front(all_points, hull, edges)

  visible_edges = []

  # Edges with a vertex in the front are visible
  for v0, v1 in edges:
    if v0 in front or v1 in front:
      visible_edges.append((v0, v1))

  # Edges on the hull that do not intersect a visible edge are visible
  for v0, v1 in edges:
    if v0 in hull and v1 in hull:
      intersects = False

      for vis0, vis1 in visible_edges:
        if intersects:
          break
        if v0 in [vis0, vis1] or v1 in [vis0, vis1]:
          continue
        x1, y1 = points[v0]
        x2, y2 = points[v1]
        x3, y3 = points[vis0]
        x4, y4 = points[vis1]

        # (x2 - x1) * alpha - (x4 - x3) * beta = x3 - x1
        # (y2 - y1) * alpha - (y4 - y3) * beta = y3 - y1
        m = np.array([[x2 - x1, x3 - x4], [y2 - y1, y3 - y4]])
        rhs = [x3 - x1, y3 - y1]
        try:
          a,b = np.linalg.solve(m, rhs)
          eps = 1e-9
          if b > eps and b < 1 - eps and a > eps and a < 1 - eps:
            intersects = True
        except np.linalg.LinAlgError:
          pass
      if not intersects:
        visible_edges.append((v0, v1))

  for v0, v1 in edges:
    if (v0, v1) in visible_edges:
      linetype = linestyle
    else:
      linetype = ':'
    x0, y0 = points[v0]
    x1, y1 = points[v1]
    if (v0,v1) in visible_edges:
      plt.plot([x0, x1], [y0, y1], color=color, linestyle=linetype, linewidth=linewidth)

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

  # Rotate both inner and outer polyhedron for cube in order to get the inner one oriented as a non-rotated square
  if name == 'cube':
    f = 0.69003891 #cube
  else:
    f = 1

  # For the cube, we want to show the margin between inner and outer polyhedron, so the inner polyhedron is not scaled
  if 'cube' in name:
    s = 1
  s = 1

  points_q = project_to_plane(points, theta_q, phi_q)
  points_p = project_to_plane(points, theta_p, phi_p)

  points_q = rotate(points_q, -(1-f)*alpha)
  points_p = rotate(points_p, f*alpha)

  points_p = scale(points_p, s)
  points_p = translate(points_p, u, v)

  p0 = points_p[0]
  
  hq = ConvexHull(points_q)
  hp = ConvexHull(points_p)

  plot_polyhedron(edges, points_q, 'grey', 2, ':')
  plot_polyhedron(edges, points_p, 'k', 1, '-')

  p1 = Polygon(points_q, theta_q, phi_q)
  p2 = Polygon(points_p, theta_p, phi_p)
  contains = p1.contains_explicit(p2)
  print('Contains: %s' % contains)

  plt.xticks([])
  plt.yticks([])

  ax = plt.gca()
  ax.axis("off")

  # Some special name handling
  name = name.replace('(laevo)', '')
  title_name = name[0] + name[1:].lower()
  title_name = title_name.replace('(j', '(J')
  name = name.replace(' ', '')
  if '(' in name:
    name = name[:name.index('(')]
  
  if 'cube' in name:
    # Same axes for both images of cube to get the same scale of the inner polyhedron
    ax.axis([-2,2,-2,2])
    ax.set_aspect('equal')
    bbox = matplotlib.transforms.Bbox([[1.3,0.5],[5.2,4.3]])
    #plt.savefig(name + '.svg', bbox_inches='tight')
    plt.savefig(name + '.eps', format='eps', bbox_inches='tight')
  else:
    ax.axis('equal')
    #plt.savefig(name + '.svg', bbox_inches='tight' )
    plt.savefig(name + '.eps', format='eps', bbox_inches='tight' )
  plt.show()


def plot_solutions():
    ''' Plot of the solutions '''
    points = cube()
    edges = find_edges(points, 2)
    plot_containment('cube1', points,edges, [math.pi/4, 0.955316], [ 0, 0 ],0, 0,0,1.0606601)
    plot_containment('cube', points,edges, [2*math.pi + -4.63638229e-01,  8.41065611e-01], [ 5.42866038e+00, 0 ],1.038015373, 0,0,1.0606601)

    if True:
      name, points, faces, edges = read_polyhedron('Catalan/01TriakisTetrahedron.txt')
      plot_containment(name, points, edges, [1.54810735, 2.35615010], [2*math.pi-0.00005492, 0.81723402], 2*math.pi-0.016082216 , 0.000140808 , -0.000002238, 1.000004) # 1.0000040769 NM optimize

      name, points, faces, edges = read_polyhedron('Catalan/07LpentagonalIcositetrahedron.txt')
      plot_containment(name, points, edges, [2.3301605, 3.0267874], [0.4660288, 1.46766889],  2.325648663 , 0.000619928 , 0.002845302, 1.000436)

      name, points, faces, edges = read_polyhedron('Johnson/GyroelongatedPentagonalRotunda.txt')
      plot_containment(name + ' (J25)', points, edges, [1.5697500508, math.acos(0.516259456)], [3.44208101, math.acos(-0.1893870555)] , 0.0031319 , 0.0013265 , -0.0541425, 1.000089) # J25 1.0000894999 optimize

    if False:
      name, points, faces, edges = read_polyhedron('Johnson/GyroelongatedSquareBicupola.txt')
      plot_containment(name + ' (J45)', points, edges, [4.71940540634669, math.acos(-0.57234816598)], [3.148896509339, math.acos(0.002509670578455)], 0.0040334 , -0.0017357 , 0.0774202, 1.000009 ) # J45 1.00000956 optimize

      name, points, faces, edges = read_polyhedron('Johnson/GyroelongatedPentagonalCupolarotunda.txt')
      plot_containment(name + ' (J47)', points, edges, [3.4528520562794, math.acos(-0.42733970469)], [3.4424819869, math.acos(-0.19547975499)],0.0013670 , 0.0005797 , -0.0330901, 1.000080 ) # J47 1.00008028 optimize

      name, points, faces, edges = read_polyhedron('Johnson/TriaugmentedTruncatedDodecahedron.txt')
      plot_containment(name + ' (J71)', points, edges, [3.41783398253844, math.acos(-0.9152295760)], [0.789632179442, math.acos(-0.00051359417)],2.4444757 , 0.0045658 , -0.0039431 , 1.000598) # J71 1.000598658

      name, points, faces, edges = read_polyhedron('Johnson/DiminishedRhombicosidodecahedron.txt')
      plot_containment(name + ' (J76)', points,edges, [4.723800802824783, math.acos(0.322995794225)], [3.3181200432255, math.acos(-0.4325411974)],5.1601146 , 0.0003653 , 0.0103378 , 1.000269) # J76

if __name__ == '__main__':
  plot_solutions()
