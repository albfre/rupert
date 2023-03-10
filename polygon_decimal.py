from scipy.spatial import ConvexHull
import itertools
from multiprocessing import Pool
from functools import partial
from mpmath import mp, mpf
import mpmath
import math

golden_ratio = (1.0 + mpmath.sqrt(5)) / 2

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

def has_even_number_of_minus_signs(p):
  return sum(1 for r in p if r < 0.0) % 2 == 0

def has_odd_number_of_minus_signs(p):
  return sum(1 for r in p if r < 0.0) % 2 == 1

def plus_minus_iter(p):
  return list(itertools.product((-p[0], p[0]), (-p[1], p[1]), (-p[2], p[2])))

def cube():
  return plus_minus_iter([mpf(1), mpf(1), mpf(1)])

def snub_cube():
  t = mpf('1.83928675521416113255185256465328')
  r = []
  r += [p for p in even_permutations(plus_minus_iter([mpf(1), 1 / t, t])) if has_odd_number_of_minus_signs(p)]
  r += [p for p in odd_permutations(plus_minus_iter([mpf(1), 1 / t, t])) if has_even_number_of_minus_signs(p)]
  return r

def rhombicosidodecahedron():
  r = []
  r += even_permutations(plus_minus_iter([1, 1, golden_ratio**3]))
  r += even_permutations(plus_minus_iter([golden_ratio**2, golden_ratio, 2 * golden_ratio]))
  r += even_permutations(list(itertools.product((-(2+golden_ratio),2+golden_ratio), (0,), (-golden_ratio**2, golden_ratio**2))))
  return r

def snub_dodecahedron():
  xi = (golden_ratio / 2 + 0.5 * mpmath.sqrt(golden_ratio - 5.0 / 27))**(1.0 / 3) + (golden_ratio / 2 - 0.5 * mpmath.sqrt(golden_ratio - 5.0 / 27))**(1.0 / 3)
  alpha = xi - 1.0 / xi
  beta = xi * golden_ratio + golden_ratio**2 + golden_ratio / xi
  phi = golden_ratio
  r = []

  p = [2 * alpha, 2, 2 * beta]
  r += even_permutations(p for p in plus_minus_iter(p) if has_odd_number_of_minus_signs(p))

  p = [alpha + beta / phi + phi, -alpha * phi + beta + 1 / phi, alpha / phi + beta * phi - 1 ]
  r += even_permutations(p for p in plus_minus_iter(p) if has_odd_number_of_minus_signs(p))

  p = [alpha + beta / phi - phi, alpha * phi - beta + 1 / phi, alpha / phi + beta * phi + 1]
  r += even_permutations(p for p in plus_minus_iter(p) if has_odd_number_of_minus_signs(p))

  p = [-alpha / phi + beta * phi + 1, -alpha + beta / phi - phi, alpha * phi + beta - 1 / phi]
  r += even_permutations(p for p in plus_minus_iter(p) if has_odd_number_of_minus_signs(p))

  p = [-alpha / phi + beta * phi -1, alpha - beta / phi - phi, alpha * phi + beta + 1 / phi]
  r += even_permutations(p for p in plus_minus_iter(p) if has_odd_number_of_minus_signs(p))

def read_polyhedron(f):
  lines = [l for l in open(f).readlines() if len(l) > 0]
  name = lines[0].strip()
  constants = {}
  for line in lines:
    if line[0] == 'C':
      cn = line[0:line.index('=')].strip()
      l2 = line[line.index('=')+1:]
      if cn in constants:
        continue
      if '=' in l2:
        cv = mpf(l2[:l2.index('=')].strip())
      else:
        cv = mpf(l2.strip())
      constants[cn] = cv

  points = []
  for line in lines:
    if line[0] == 'V':
      p = []
      values = line[line.index('=')+1:].replace('(', '').replace(')','')
      values = values.split(',')
      for v in values:
        v = v.strip()
        if v[0] == 'C':
          p.append(constants[v])
        elif v[0] == '-' and v[1] == 'C':
          p.append(-constants[v[1:]])
        else:
          p.append(mpf(v))
      points.append(p)

  faces = []
  for line in lines:
    p = []
    if line[0] == '{':
      values = line.replace('{','').replace('}','')
      values = values.split(',')
      for v in values:
        p.append(int(v))
      faces.append(p)

  edges = []
  for face in faces:
    for i in range(len(face)):
      edges.append([face[i], face[(i+1) % len(face)]])

  return name, points, faces, edges

def project_to_plane(points, theta, phi):
  assert(all(len(p) == 3 for p in points))
  st = mpmath.sin(theta)
  ct = mpmath.cos(theta)
  sp = mpmath.sin(phi)
  cp = mpmath.cos(phi)
  return [[-st * p[0] + ct * p[1], -ct * cp * p[0] - st * cp * p[1] + sp * p[2]] for p in points]

def rotate(points, alpha):
  ca = mpmath.cos(alpha)
  sa = mpmath.sin(alpha)
  return [[ca * p[0] - sa * p[1], sa * p[0] + ca * p[1]] for p in points]

def scale(points, s):
  return [[s * x, s * y] for x,y in points]

def translate(points, u, v):
  return [[x + u, y + v] for x,y in points]

def contains(vq, vp):
    for i in range(len(vq)):
      x1, y1 = vq[i]
      x2, y2 = vq[(i + 1) % len(vq)]
      for x, y in vp:
        if (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1) >= 0: return False

    return True

def test_containment(polyhedron, q_angles, p_angles, alpha, u, v, s = 1.0):
  '''Verify that a solution works by explicit rotation, translation, and scaling'''
  theta_q, phi_q = q_angles
  theta_p, phi_p = p_angles

  n = 6
  #theta_q = round(theta_q, n)
  #phi_q = round(phi_q, n)
  #theta_p = round(theta_p, n)
  #phi_p = round(phi_p, n)

  n = 7
  #alpha = round(alpha, n)
  #u = round(u, n)
  #v = round(v, n)

  points_q = project_to_plane(polyhedron, theta_q, phi_q)
  points_p = project_to_plane(polyhedron, theta_p, phi_p)

  points_p = rotate(points_p, alpha)
  points_p = scale(points_p, s)
  points_p = translate(points_p, u, v)

  hq = ConvexHull([[float(x) for x in xyz] for xyz in points_q])
  hp = ConvexHull([[float(x) for x in xyz] for xyz in points_p])

  vq = [points_q[vi] for vi in hq.vertices]
  vp = [points_p[vi] for vi in hp.vertices]
  #print(str(vq))
  #print(str(vp))
  

  if contains(vq, vp):
    print('contains')
    print(str((theta_q, phi_q, theta_p, phi_p, alpha, u, v, s)))
    #print(str(theta_p) + " & " + str(phi_p) + " & " + str(theta_q) + " & " + str(phi_q) + " & " + str(s))
    #print(str(alpha) + " & " + str(u) + " & " + str(v))
    return True

  return False


def run():
  mp.dps = 50

#(3.5464272875133513, 2.541668861004088) contains (t,p) = (3.70287224436372, 1.7950400987173274) with alpha=1.910743879065393, translation=(-7.22261649110295e-14, 6.538120156978815e-14), scaling=1.0000000000000113

  #test_containment3(cube(), [ mpf(2.0344439 ), mpf(0.8410687) ], [mpf(2*math.pi-0.0178793 ), mpf(0.0000000) ], mpf(1.231166453), 0,0, 1.06) 
  #test_containment3(snub_cube(), [mpf(3.5464272875133513), mpf(2.541668861004088)],[mpf(3.70287224436372), mpf(1.7950400987173274)], mpf(1.910743879065393), 0, 0, mpf('1.00000000000000')) #scaling=1.0000000000000113

  #test_containment(snub_cube(), [mpf(2.6122362440845954), mpf(1.8314668919060075)],[mpf(5.198357261240355), mpf(0.5845266568498758)], mpf(-1.1558523612574354),0,0, mpf(1)) #=1.0000000000000133

  #test_containment(snub_cube(), [mpf(0), mpf()],[mpf('5.026548245743669'), mpf('3.1415926535897931')], mpf('0.62831853071795862'),0,0, mpf('1.000000000000000000001')) #=1.0000000000000133

  test_containment(snub_cube(), [mpf('0.0'), mpf('3.1415926535897932384626433832795028841971693993751068')], [mpf('3.7699111843077518861551720599354034610365199999999974'), mpf('0.0')], mpf('3.7699111843077518861551720599354034610365199999999974'), 0, 0,mpf('1.0000000000000000000000000000001'))


def test_wrapper(p, angles):
  tq, pq, tp, pp, a = angles
  c = test_containment(p, [tq, pq],[tp, pp], a,0,0, mpf('1.0000000000000000000000000000001'))
  return c

def run_brute():
  mp.dps = 50
  angles = []
  theta_q = mpf(2.6122362440845954)
  phi_q = mpf(1.8314668919060075)
  theta_p = mpf(5.198357261240355)
  phi_p = mpf(0.5845266568498758)
  alpha = mpf(-1.1558523612574354)
  s = mpf(1)
  u = mpf(0)
  v = mpf(0)

  n = 30
  thetas = [2 * mpf('3.1415926535897932384626433832795028841971') * i / n for i in range(n)]
  phis = [mpmath.acos(-1 + mpf(2) * j / (n - 1)) for j in range(n)]
  alphas = thetas
  angles = [(theta, phi) for theta in thetas for phi in phis]
  angles = [qa + pa + (alpha,) for qa in angles for pa in angles for alpha in alphas]

  print('Angles: %s ' % len(angles))
  i = 0
  any_cont = False
  p = snub_cube()

  n_starting_points = len(angles)
  n_per_chunk = 100
  n_chunks = round(n_starting_points / n_per_chunk + 1)
  largest_scaling = 0.0
  solution = None

  if True:
    with Pool() as pool:
      for chunk_i in range(n_chunks):
        begin_i = min(chunk_i * n_per_chunk, n_starting_points)
        end_i = min((chunk_i + 1) * n_per_chunk, n_starting_points)
        print('p' + str(begin_i) + ' of ' + str(len(angles)) + '. ' + str(any_cont) + ", " + str(solution))
        if begin_i == end_i: continue
        chunk_angles = angles[begin_i:end_i]
        result = pool.map(partial(test_wrapper, p), chunk_angles)
        for r, ang in zip(result, chunk_angles):
          if r:
            any_cont = True
            solution = ang
            print('Contains: ' + str(ang))
  else:
    for tq, pq, tp, pp, a in angles:
      if i % 100 == 0:
        print(str(i) + " of " + str(len(angles)) + ", " + str(any_cont) + ", " + str(solution))
      c = test_containment(p, [tq, pq],[tp, pp], a, 0,0, mpf('1.0000000000000000000000000000001')) #=1.0000000000000133
      if c:
        any_cont = True
        solution = [tq, pq, tp, pp, a]
        print('Contains: ' + str(solution))
      i += 1

