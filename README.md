# hypopy

HYPOcenter location from arrival time data in PYthon

There are currently 4 hypocenter location functions in the hypo module

- **hypoloc** : Locate hypocenters for constant velocity model
- **hypolocPS** : Locate hypocenters from P- and S-wave arrival time data for constant velocity models
- **jointHypoVel** : Joint hypocenter-velocity inversion on a regular grid (cubic cells)
- **jointHypoVelPS** : Joint hypocenter-velocity inversion of P- and S-wave arrival time data

See the tutorials for some examples.  There is also a notebook about the theory.

## Requirements

Development is made with python version 3.6

You need to compile the python wrapper for the C++ raytracing code in https://github.com/groupeLIAMG/ttcr and add it to your PYTHONPATH to be able to run hypo.py

If you have VTK compiled with python on your system, it is possible to save velocity models and raypaths for posterior visualization (e.g. in paraview).

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
