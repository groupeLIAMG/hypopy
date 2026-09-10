# hypopy

HYPOcenter location from arrival time data in PYthon

There are currently 4 hypocenter location functions in the hypo module

- **hypoloc** : Locate hypocenters for constant homogeneous velocity model
- **hypolocPS** : Locate hypocenters from P- and S-wave arrival time data for constant homogeneous velocity models
- **jointHypoVel** : Joint hypocenter-velocity inversion on a regular grid (cubic cells)
- **jointHypoVelPS** : Joint hypocenter-velocity inversion of P- and S-wave arrival time data

See the tutorials for some examples.  There is also a notebook about the theory.

## Requirements

Python 3.11 or later, with numpy, scipy and matplotlib, and

```
pip install ttcrpy
```

for the raytracing.  Version 1.5.2 or later is needed: earlier ones refuse to
compute the sensitivity matrix for the fast sweeping method, and hand a worker
process a grid whose slowness has not survived being pickled, which the
tutorials rely on.  Building from https://github.com/groupeLIAMG/ttcr works
too, if you want something newer than the release.

If VTK is installed, `save_V` and `save_rp` write velocity models and raypaths
for later viewing, in paraview say.  Without it the two options warn and write
nothing, rather than failing.

### Running on the GPU

`ttcrpy` can run the fast sweeping method through OpenCL, which the tutorials
ask for:

```python
g = Grid3d(x, y, z, nthreads, cell_slowness=True, method='FSM',
           dtype=np.float32, fsm_gpu=True)
```

`dtype=np.float32` is not incidental.  A device that does not advertise
`cl_khr_fp64` -- which includes every Apple GPU -- refuses a double precision
grid, so asking for `np.float64` quietly gives you the CPU instead.  Ask
`g.is_using_gpu` if you want to know which one you got: a refusal falls back
and still returns correct results, so the answer is otherwise invisible.

Keep sources at least one cell away from the edges of the model.  The FSM and
DSPM rebuild raypaths by descending the traveltime gradient, and one that
reaches a face of the grid before it reaches the source has no step left and
raises.

## Tests

```
python -m unittest discover -s tests -t .
```

from the root of the repository; `-t .` is what lets the tests import `hypo`.

## Residuals returned by the joint inversions

`jointHypoVel` and `jointHypoVelPS` return three residual arrays, not two:

```python
h, V, sc, res = hypo.jointHypoVel(...)
resV, resAxb, resLoc = res      # velocity, system, hypocenter
```

`res` is still a tuple, so indexing and `len` behave as before, but code
written against the older two-element form needs the third name.  `resLoc` has
shape `(maxit, nev, maxit_hypo)` and holds the traveltime misfit of each event
at each iteration of its relocation, the counterpart of what `hypoloc`
returns; entries are zero where an iteration was not reached, so mask them
with `> 0`.

## References

```
@PhdThesis{block91,
  Title                    = {Joint Hypocenter-Velocity Inversion of Local Earthquake Arrival Time Data in Two Geothermal Regions},
  Author                   = {Lisa Victoria Block},
  School                   = {Massachusetts Institute of Technology},
  Year                     = {1991}
}

@Article{block94,
  Title                    = {Seismic imaging using microearthquakes induced by hydraulic fracturing},
  Author                   = {Lisa V. Block and C. H. Cheng and Michael C. Fehler and W. Scott Phillips},
  Journal                  = {Geophysics},
  Year                     = {1994},
  Number                   = {1},
  Pages                    = {102--112},
  Volume                   = {59}
}
```
