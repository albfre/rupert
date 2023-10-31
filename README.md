# Optimizing for Rupertness

The main optimization algorithm is implemented in `optimize_4d.py`.

To optimize for Rupertness for a polyhedron (e.g., the triakis tetrahedron) using this algorithm, use the REPL and run

```
exec(open('run_opt.py').read())
run(triakis_tetrahedron(), 5)
```

where 5 is the discretization of starting points.

A number of polyhedra (the Platonic and the Archimedean, and some of the Catalan) are provided in `polyhedra.py`.

## Additional search methods

`bruteforce.py` contains a bruteforce search method.

`optimize_3d.py` contains an optimization method trying to maximize the distance between the rotated projection of a polyhedron to the convex hull of different silhouettes of of the polyhedron.

`optimize_scip.py` contains experimental work trying to use the global optimizer SCIP to show Rupertness.

## Verification

`verify_results.py` contains calculations for explicit verification of found results. Run from command line by:

```
python verify_results.py
```

## Plotting

`plot_results.py` contains methods to plot projections of polyhedra. Run from command line by:

```
python plot_results.py
```
