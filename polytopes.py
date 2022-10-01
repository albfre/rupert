import itertools
import numpy as np
import math

golden_ratio = (1.0 + math.sqrt(5)) / 2

def read_file(f):
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
        cv = float(l2[:l2.index('=')].strip())
      else:
        cv = float(l2.strip())
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
          p.append(float(v))
      points.append(p)
  return name, points


def pentagonal_icositetrahedron():
  return read_file('Catalan/07LpentagonalIcositetrahedron.txt')[1]

def triakis_tetrahedron():
  return read_file('Catalan/01TriakisTetrahedron.txt')[1]

def deltoidal_hexecontahedron():
  return read_file('Catalan/11DeltoidalHexecontahedron.txt')[1]

def pentagonal_hexecontahedron():
  return read_file('Catalan/12LpentagonalHexecontahedron.txt')[1]

def random_polytope(n):
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

def plus_minus_iter(p):
  return list(itertools.product((-p[0], p[0]), (-p[1], p[1]), (-p[2], p[2])))

def tetrahedron():
  return [[-1,-1,1],[-1,1,-1],[1,-1,-1],[1,1,1]]

def cube():
  return plus_minus_iter([1, 1, 1])

def octahedron():
  return all_permutations(list(itertools.product((0,),(0,),(-1,1))))

def dodecahedron():
  r = cube()
  r += even_permutations(list(itertools.product((0,), (-1/golden_ratio,1/golden_ratio), (-golden_ratio,golden_ratio))))
  return r

def icosahedron():
  return even_permutations(list(itertools.product((0,),(-1,1),(-golden_ratio,golden_ratio))))

def truncated_tetrahedron():
  return [p for p in all_permutations(plus_minus_iter([1, 1, 3])) if has_even_number_of_minus_signs(p)]

def cuboctahedron():
  return all_permutations(list(itertools.product((-1,1),(-1,1),(0,))))

def truncated_cube():
  p = math.sqrt(2.0) - 1.0
  return all_permutations(plus_minus_iter([1, 1, p]))

def truncated_octahedron():
  return all_permutations(list(itertools.product((0,),(-1,1),(-2,2))))

def rhombicuboctahedron():
  p = math.sqrt(2.0) + 1.0
  return all_permutations(plus_minus_iter([1, 1, p]))

def truncated_cuboctahedron():
  p = math.sqrt(2.0)
  return all_permutations(plus_minus_iter([1, 1 + p, 1 + 2 * p]))
  
def snub_cube():
  t = 1.83928675521416113255185256465328
  r = []
  r += [p for p in even_permutations(plus_minus_iter([1, 1 / t, t])) if has_odd_number_of_minus_signs(p)]
  r += [p for p in odd_permutations(plus_minus_iter([1, 1 / t, t])) if has_even_number_of_minus_signs(p)]
  return r

def icosidodecahedron():
  r = []
  r += all_permutations(list(itertools.product((0,),(0,),(-golden_ratio,golden_ratio))))
  r += even_permutations(plus_minus_iter([0.5, golden_ratio / 2, golden_ratio ** 2 / 2]))
  return r

def truncated_dodecahedron():
  r = []
  r += even_permutations(list(itertools.product((0,),(-1/golden_ratio,1/golden_ratio),(-(2+golden_ratio),2+golden_ratio))))
  r += even_permutations(plus_minus_iter([1 / golden_ratio, golden_ratio, 2 * golden_ratio]))
  r += even_permutations(plus_minus_iter([golden_ratio, 2, golden_ratio + 1]))
  return r

def truncated_icosahedron():
  g = (1 + math.sqrt(5))/2.0
  r = []
  r += odd_permutations(list(itertools.product((0,), (-1,1), (-3*g, 3*g))))
  r += odd_permutations(plus_minus_iter([1, 2 + g, 2 * g]))
  r += odd_permutations(plus_minus_iter([g, 2, 2 * g + 1]))
  return r

def rhombicosidodecahedron():
  r = []
  r += even_permutations(plus_minus_iter([1, 1, golden_ratio**3]))
  r += even_permutations(plus_minus_iter([golden_ratio**2, golden_ratio, 2 * golden_ratio]))
  r += even_permutations(list(itertools.product((-(2+golden_ratio),2+golden_ratio), (0,), (-golden_ratio**2, golden_ratio**2))))
  return r

def truncated_icosidodecahedron():
  r = []
  r += even_permutations(plus_minus_iter([1 / golden_ratio, 1 / golden_ratio, 3 + golden_ratio]))
  r += even_permutations(plus_minus_iter([2 / golden_ratio, golden_ratio, 1 + 2 * golden_ratio]))
  r += even_permutations(plus_minus_iter([1 / golden_ratio, golden_ratio**2, -1 + 3 * golden_ratio]))
  r += even_permutations(plus_minus_iter([2 * golden_ratio -1, 2, 2 + golden_ratio]))
  r += even_permutations(plus_minus_iter([golden_ratio, 3, 2 * golden_ratio]))
  return r


def snub_dodecahedron():
  xi = (golden_ratio / 2 + 0.5 * math.sqrt(golden_ratio - 5.0 / 27))**(1.0 / 3) + (golden_ratio / 2 - 0.5 * math.sqrt(golden_ratio - 5.0 / 27))**(1.0 / 3)
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

  return r
