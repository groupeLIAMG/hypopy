# hypopy

python codes to do hypocenter location from arrival time data

There are currently 4 hypocenter location functions in the hypo module

- hypoloc : Locate hypocenters for constant velocity model
- hypolocPS : Locate hypocenters from P- and S-wave arrival time data for constant velocity models
- jointHypoVel : Joint hypocenter-velocity inversion on a regular grid (cubic cells)
- jointHypoVelPS : Joint hypocenter-velocity inversion of P- ans S-wave arrival time data

See the tutorial for some examples

## Requirements

You need to compile the python wrapper for the C++ raytracing code in https://github.com/groupeLIAMG/ttcr and add it to your PYTHONPATH to be able to run hypo.py
