from polygon import *
from polytopes import *

def test_results():
    ''' Test of containment printing the results in a Latex-friendly manner '''

    # Catalan
    test_containment(read_file('Catalan/01TriakisTetrahedron.txt')[1], [1.54810735, 2.35615010], [2*math.pi-0.00005492, 0.81723402]) # 1.0000040769 NM optimize
    test_containment(read_file('Catalan/07LpentagonalIcositetrahedron.txt')[1], [2.3301605, 3.0267874], [0.4660288, 1.46766889]) # 1.0004360874 optimize

    # Johnson
    test_containment(read_file('Johnson/GyroelongatedPentagonalRotunda.txt')[1], [1.5697500508, math.acos(0.516259456)], [3.44208101, math.acos(-0.1893870555)]) # J25 1.0000894999 optimize
    test_containment(read_file('Johnson/GyroelongatedSquareBicupola.txt')[1], [4.71940540634669, math.acos(-0.57234816598)], [3.148896509339, math.acos(0.002509670578455)]) # J45 1.00000956 optimize
    test_containment(read_file('Johnson/GyroelongatedPentagonalCupolarotunda.txt')[1], [3.4528520562794, math.acos(-0.42733970469)], [3.4424819869, math.acos(-0.19547975499)]) # J47 1.00008028 optimize
    test_containment(read_file('Johnson/TriaugmentedTruncatedDodecahedron.txt')[1], [3.41783398253844, math.acos(-0.9152295760)], [0.789632179442, math.acos(-0.00051359417)]) # J71 1.000598658
    test_containment(read_file('Johnson/DiminishedRhombicosidodecahedron.txt')[1],[4.723800802824783, math.acos(0.322995794225)], [3.3181200432255, math.acos(-0.4325411974)]) # J76 1.0002693188

def verify_results():
    ''' Test of containment using explicit rotation and translation '''

    # Catalan
    print('Triakis tetrahedron')
    test_containment_explicit(read_file('Catalan/01TriakisTetrahedron.txt')[1], [1.54810735, 2.35615010], [2*math.pi-0.00005492, 0.81723402], 2*math.pi-0.016082216 , 0.000140808 , -0.000002238, 1.000004) # 1.0000040769 NM optimize

    print('Pentagonal icositetrahedron')
    test_containment_explicit(read_file('Catalan/07LpentagonalIcositetrahedron.txt')[1], [2.3301605, 3.0267874], [0.4660288, 1.46766889],  2.325648663 , 0.000619928 , 0.002845302, 1.000436)

    # Johnson
    print('J25')
    test_containment_explicit(read_file('Johnson/GyroelongatedPentagonalRotunda.txt')[1], [1.5697500508, math.acos(0.516259456)], [3.44208101, math.acos(-0.1893870555)] , 0.0031319 , 0.0013265 , -0.0541425, 1.000089) # J25 1.0000894999 optimize
    print('J45')
    test_containment_explicit(read_file('Johnson/GyroelongatedSquareBicupola.txt')[1], [4.71940540634669, math.acos(-0.57234816598)], [3.148896509339, math.acos(0.002509670578455)], 0.0040334 , -0.0017357 , 0.0774202, 1.000009 ) # J45 1.00000956 optimize
    print('J47')
    test_containment_explicit(read_file('Johnson/GyroelongatedPentagonalCupolarotunda.txt')[1], [3.4528520562794, math.acos(-0.42733970469)], [3.4424819869, math.acos(-0.19547975499)],0.0013670 , 0.0005797 , -0.0330901, 1.000080 ) # J47 1.00008028 optimize
    print('J71')
    test_containment_explicit(read_file('Johnson/TriaugmentedTruncatedDodecahedron.txt')[1], [3.41783398253844, math.acos(-0.9152295760)], [0.789632179442, math.acos(-0.00051359417)],2.4444757 , 0.0045658 , -0.0039431 , 1.000598) # J71 1.000598658
    print('J76')
    test_containment_explicit(read_file('Johnson/DiminishedRhombicosidodecahedron.txt')[1],[4.723800802824783, math.acos(0.322995794225)], [3.3181200432255, math.acos(-0.4325411974)],5.1601146 , 0.0003653 , 0.0103378 , 1.000269) # J76 1.0002693188

if __name__ == '__main__':
  verify_results()
