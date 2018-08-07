# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 10:29:32 2016

@author: giroux
"""
import sys
from multiprocessing import Process, Queue

import numpy as np
import numpy.matlib as matlib
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import matplotlib.pyplot  as plt

import h5py

from utils import nargout
import cgrid3d

def hypoloc(data, rcv, V, hinit, maxit, convh, verbose=False):
    """
    Locate hypocenters for constant velocity model

    Parameters
    ----------
    data  : a numpy array with 5 columns
             first column is event ID number
             second column is arrival time
             third column is receiver index
    rcv:  : coordinates of receivers
             first column is easting
             second column is northing
             third column is elevation
    V     : wave velocity
    hinit : initial hypocenter coordinate.  The format is the same as for data
    maxit : max number of iterations
    convh : convergence criterion (units of distance)

    Returns
    -------
    loc : hypocenter coordinates
    res : norm of residuals at each iteration for each event (nev x maxit)
    """

    if verbose:
        print('\n *** Hypocenter inversion ***\n')
    evID = np.unique(data[:,0])
    loc = hinit.copy()
    res = np.zeros((evID.size, maxit))
    nev = 0
    for eid in evID:
        ind = eid==data[:,0]

        ix = data[ind,2]
        x = np.zeros((ix.size,3))
        for i in np.arange(ix.size):
            x[i,:] = rcv[int(1.e-6+ix[i]),:]
        t = data[ind,1]

        inh = eid==loc[:,0]
        if verbose:
            print('Locating hypocenters no '+str(int(1.e-6+eid)))
            sys.stdout.flush()

        for it in range(maxit):

            xinit = loc[inh,2:]
            tinit = loc[inh,1]

            dx = x[:,0] - xinit[0,0]
            dy = x[:,1] - xinit[0,1]
            dz = x[:,2] - xinit[0,2]
            ds = np.sqrt( dx*dx + dy*dy + dz*dz )
            tcalc = tinit + ds/V

            H = np.ones((x.shape[0], 4))
            H[:,1] = -1.0/V * dx/ds
            H[:,2] = -1.0/V * dy/ds
            H[:,3] = -1.0/V * dz/ds

            r = t - tcalc
            res[nev, it] = np.linalg.norm(r)

            #dh,residuals,rank,s = np.linalg.H.T.dot(( H, r)
            dh = np.linalg.solve( H.T.dot(H), H.T.dot(r) )
            if not np.all(np.isfinite(dh)):
                try:
                    U,S,VVh = np.linalg.svd(H.T.dot(H)+1e-9*np.eye(4))
                    VV = VVh.T
                    dh = np.dot( VV, np.dot(U.T, H.T.dot(r))/S)
                except np.linalg.linalg.LinAlgError:
                    print('  Event could not be relocated (iteration no '+str(it)+'), skipping')
                    sys.stdout.flush()
                    break

            loc[inh,1:] += dh
            if np.sum(np.abs(dh[1:])<convh) == 3:
                if verbose:
                    print('     Converged at iteration '+str(it+1))
                    sys.stdout.flush()
                break
        else:
            if verbose:
                print('     Reached max number of iteration ('+str(maxit)+')')
                sys.stdout.flush()

        nev += 1

    if verbose:
        print('\n ** Inversion complete **\n')

    return loc, res

def hypolocPS(data, rcv, V, hinit, maxit, convh, verbose=False):
    """
    Locate hypocenters for constant velocity model

    Parameters
    ----------
    data  : a numpy array with 6 columns
             first column is event ID number
             second column is arrival time
             third column is receiver index
             fourth column is code for wave phase: 0 for P-wave and 1 for S-wave
    rcv:  : coordinates of receivers
             first column is easting
             second column is northing
             third column is elevation
    V     : tuple holding wave velocities, 1st value is for P-wave, 2nd for S-wave
    hinit : initial hypocenter coordinate.  The format is the same as for data
    maxit : max number of iterations
    convh : convergence criterion (units of distance)

    Returns
    -------
    loc : hypocenter coordinates
    """

    if verbose:
        print('\n *** Hypocenter inversion  --  P and S-wave data ***\n')
    evID = np.unique(data[:,0])
    loc = hinit.copy()
    res = np.zeros((evID.size, maxit))
    nev = 0
    for eid in evID:
        ind = eid==data[:,0]

        ix = data[ind,2]
        x = np.zeros((ix.size,3))
        for i in np.arange(ix.size):
            x[i,:] = rcv[int(1.e-6+ix[i]),:]

        t = data[ind,1]
        ph = data[ind,3]
        vel = np.zeros((len(ph),))
        for n in range(len(ph)):
            vel[n] = V[int(1.e-6+ph[n])]


        inh = eid==loc[:,0]
        if verbose:
            print('Locating hypocenters no '+str(int(1.e-6+eid)))
            sys.stdout.flush()

        for it in range(maxit):

            xinit = loc[inh,2:]
            tinit = loc[inh,1]

            dx = x[:,0] - xinit[0,0]
            dy = x[:,1] - xinit[0,1]
            dz = x[:,2] - xinit[0,2]
            ds = np.sqrt( dx*dx + dy*dy + dz*dz )
            tcalc = tinit + ds/vel

            H = np.ones((x.shape[0], 4))
            H[:,1] = -1.0/vel * dx/ds
            H[:,2] = -1.0/vel * dy/ds
            H[:,3] = -1.0/vel * dz/ds

            r = t - tcalc
            res[nev, it] = np.linalg.norm(r)

            #dh,residuals,rank,s = np.linalg.lstsq( H, r)
            dh = np.linalg.solve( H.T.dot(H), H.T.dot(r) )
            if not np.all(np.isfinite(dh)):
                try:
                    U,S,VVh = np.linalg.svd(H.T.dot(H)+1e-9*np.eye(4))
                    VV = VVh.T
                    dh = np.dot( VV, np.dot(U.T, H.T.dot(r))/S)
                except np.linalg.linalg.LinAlgError:
                    print('  Event could not be relocated (iteration no '+str(it)+'), skipping')
                    sys.stdout.flush()
                    break

            loc[inh,1:] += dh
            if np.sum(np.abs(dh[1:])<convh) == 3:
                if verbose:
                    print('     Converged at iteration '+str(it+1))
                    sys.stdout.flush()
                break
        else:
            if verbose:
                print('     Reached max number of iteration ('+str(maxit)+')')
                sys.stdout.flush()

        nev += 1

    if verbose:
        print('\n ** Inversion complete **\n')

    return loc, res

def _rt_worker(idx, grid, istart, iend, vTx, Rx, iRx, t0, nout, tt_queue):
    """
    worker for spanning raytracing to different processes
    """
    # slowness must have been set before raytracing
    for n in range(istart, iend):
        t = grid.raytrace(None,
                          np.atleast_2d(vTx[n, :]),
                          np.atleast_2d(Rx[iRx[n], :]),
                          t0[n],
                          nout=nout,
                          thread_no=idx)
        tt_queue.put((t, iRx[n], n))
    tt_queue.close()

class Grid3D():
    """
    Class for 3D regular grids with cubic voxels
    """
    def __init__(self, x, y, z, nthreads=1):
        """
        x: node coordinates along x
        y: node coordinates along y
        z: node coordinates along z
        """
        self.x = x
        self.y = y
        self.z = z
        self.nthreads = nthreads
        self.cgrid = None
        self.dx = x[1] - x[0]
        dy = y[1] - y[0]
        dz = z[1] - z[0]

        if np.abs(self.dx - dy)>0.000001 or np.abs(self.dx - dz)>0.000001:
            raise ValueError('Grid cells must be cubic')

    def getNumberOfNodes(self):
        return self.x.size * self.y.size * self.z.size

    @property
    def shape(self):
        return (self.x.size, self.y.size, self.z.size)

    def ind(self, i, j, k):
        return (i*self.y.size + j)*self.z.size + k

    def is_outside(self, pts):
        """
        Return True if at least one point outside grid
        """
        return ( np.min(pts[:,0]) < self.x[0] or np.max(pts[:,0]) > self.x[-1] or
                np.min(pts[:,1]) < self.y[0] or np.max(pts[:,1]) > self.y[-1] or
                np.min(pts[:,2]) < self.z[0] or np.max(pts[:,2]) > self.z[-1] )

    def set_slowness(self, slowness):
        self.cgrid.set_slowness(slowness)

    def raytrace(self, slowness, hypo, rcv, thread_no=None):

        nout = nargout()

        # check input data consistency

        if hypo.ndim != 2 or rcv.ndim != 2:
            raise ValueError('hypo and rcv should be 2D arrays')

        src = hypo[:,2:5]
        if src.shape[1] != 3 or rcv.shape[1] != 3:
            raise ValueError('src and rcv should be ndata x 3')

        if src.shape != rcv.shape:
            raise ValueError('src and rcv should be of equal size')

        if self.is_outside(src):
            raise ValueError('Source point outside grid')

        if self.is_outside(rcv):
            raise ValueError('Receiver outside grid')

        if slowness is not None:
            if len(slowness) != self.getNumberOfNodes():
                raise ValueError('Length of slowness vector should equal number of nodes')

        if self.cgrid is None:
            nx = len(self.x) - 1
            ny = len(self.y) - 1
            nz = len(self.z) - 1

            self.cgrid = cgrid3d.Grid3Drn(nx, ny, nz, self.dx,
                                          self.x[0], self.y[0], self.z[0],
                                          1.e-15, 20, True, self.nthreads)

        evID = hypo[:,0]
        eid = np.sort(np.unique(evID))
        nTx = len(eid)

        if thread_no is not None:
            # we should be here for just one event
            assert nout == 3
            assert nTx == 1
            t, r, v = self.cgrid.raytrace(None,
                                          np.atleast_2d(src[0, :]),
                                          np.atleast_2d(rcv),
                                          hypo[0,1],
                                          nout=nout,
                                          thread_no=thread_no)
            v0 = v+np.zeros((rcv.shape[0],))
            return t, r, v0

        i0 = np.empty((nTx,), dtype=np.int64)
        for n in range(nTx):
            for nn in range(evID.size):
                if eid[n] == evID[nn]:
                    i0[n] = nn
                    break

        vTx = src[i0,:]
        t0 = hypo[i0,1]
        iRx = []
        for i in eid:
            ii = evID == i
            iRx.append(ii)

        if slowness is not None:
            self.cgrid.set_slowness(slowness)

        tt = np.zeros((rcv.shape[0],))
        if nout >= 3:
            v0 = np.zeros((rcv.shape[0],))
            rays = [ [0.0] for n in range(rcv.shape[0])]
        if nout == 4:
            M = [ [] for i in range(nTx) ]

        if nTx < 1.5*self.nthreads or self.nthreads == 1:
            if nout == 1:
                for n in range(nTx):
                    t = self.cgrid.raytrace(None,
                                            np.atleast_2d(vTx[n, :]),
                                            np.atleast_2d(rcv[iRx[n], :]),
                                            t0[n],
                                            nout=nout)
                    tt[iRx[n]] = t
                return tt
            elif nout == 3:
                for n in range(nTx):
                    t, r, v = self.cgrid.raytrace(None,
                                                  np.atleast_2d(vTx[n, :]),
                                                  np.atleast_2d(rcv[iRx[n], :]),
                                                  t0[n],
                                                  nout=nout)
                    tt[iRx[n]] = t
                    v0[iRx[n]] = v
                    ii = np.where(iRx[n])[0]
                    for nn in range(len(ii)):
                        rays[ii[nn]] = r[nn]
                return tt, rays, v0
            elif nout == 4:
                for n in range(nTx):
                    t, r, v, m = self.cgrid.raytrace(None,
                                                     np.atleast_2d(vTx[n, :]),
                                                     np.atleast_2d(rcv[iRx[n], :]),
                                                     t0[n],
                                                     nout=nout)
                    tt[iRx[n]] = t
                    v0[iRx[n]] = v
                    ii = np.where(iRx[n])[0]
                    for nn in range(len(ii)):
                        rays[ii[nn]] = r[nn]
                    M[n] = m
                return tt, rays, v0, M

        else:
            blk_size = np.zeros((self.nthreads,), dtype=np.int64)
            nj = nTx
            while nj > 0:
                for n in range(self.nthreads):
                    blk_size[n] += 1
                    nj -= 1
                    if nj == 0:
                        break

            processes = []
            blk_start = 0
            tt_queue = Queue()

            for n in range(self.nthreads):
                blk_end = blk_start + blk_size[n]
                p = Process(target=_rt_worker,
                            args=(n, self.cgrid, blk_start, blk_end, vTx, rcv,
                                  iRx, t0, nout, tt_queue),
                            daemon=True)
                processes.append(p)
                p.start()
                blk_start += blk_size[n]

            if nout == 1:
                for i in range(nTx):
                    t, ind, n = tt_queue.get()
                    tt[ind] = t
                return tt
            elif nout == 3:
                for i in range(nTx):
                    t, iRx, n = tt_queue.get()
                    tt[iRx] = t[0]
                    v0[iRx] = t[2]
                    ind = np.where(iRx)[0]
                    for nn in range(len(ind)):
                        rays[ind[nn]] = t[1][nn]
                return tt, rays, v0
            elif nout == 4:
                for i in range(nTx):
                    t, iRx, n = tt_queue.get()
                    tt[iRx] = t[0]
                    v0[iRx] = t[2]
                    ind = np.where(iRx)[0]
                    for nn in range(len(ind)):
                        rays[ind[nn]] = t[1][nn]
                    M[n] = t[3]
                return tt, rays, v0, M

    def computeD(self, coord):
        """
        Return matrix of interpolation weights for velocity data points constraint
        """
        if self.is_outside(coord):
            raise ValueError('Velocity data point outside grid')

        # D is npts x nnodes
        # for each point in coord, we have 8 values in D
        ivec = np.kron(np.arange(coord.shape[0], dtype=np.int64),np.ones(8, dtype=np.int64))
        jvec = np.zeros(ivec.shape, dtype=np.int64)
        vec = np.zeros(ivec.shape)

        for n in np.arange(coord.shape[0]):
            i1 = int(1.e-6+ (coord[n,0]-self.x[0])/self.dx )
            i2 = i1+1
            j1 = int(1.e-6+ (coord[n,1]-self.y[0])/self.dx )
            j2 = j1+1
            k1 = int(1.e-6+ (coord[n,2]-self.z[0])/self.dx )
            k2 = k1+1

            ii = 0
            for i in (i1,i2):
                for j in (j1,j2):
                    for k in (k1,k2):
                        jvec[n*8+ii] = self.ind(i,j,k)
                        vec[n*8+ii] = ((1. - np.abs(coord[n,0]-self.x[i])/self.dx) *
                                       (1. - np.abs(coord[n,1]-self.y[j])/self.dx) *
                                       (1. - np.abs(coord[n,2]-self.z[k])/self.dx))
                        ii += 1

        return sp.csr_matrix((vec, (ivec,jvec)), shape=(coord.shape[0], self.getNumberOfNodes()))

    def computeK(self):
        """
        Return smoothing matrices (2nd order derivative)
        """
        # central operator f"(x) = (f(x+h)-2f(x)+f(x-h))/h^2
        # forward operator f"(x) = (f(x+2h)-2f(x+h)+f(x))/h^2
        # backward operator f"(x) = (f(x)-2f(x-h)+f(x-2h))/h^2

        nx = self.x.size
        ny = self.y.size
        nz = self.z.size
        # Kx
        iK = np.kron(np.arange(nx*ny*nz, dtype=np.int64), np.ones(3,dtype=np.int64))
        val = np.tile(np.array([1., -2., 1.]), nx*ny*nz) / (self.dx*self.dx)
        # i=0 -> forward op
        jK = np.vstack((np.arange(ny*nz,dtype=np.int64),
                        np.arange(ny*nz, 2*ny*nz,dtype=np.int64),
                        np.arange(2*ny*nz, 3*ny*nz,dtype=np.int64))).T.flatten()

        for i in np.arange(1,nx-1):
            jK = np.hstack((jK,
                            np.vstack((np.arange((i-1)*ny*nz,i*ny*nz, dtype=np.int64),
                                       np.arange(i*ny*nz, (i+1)*ny*nz,dtype=np.int64),
                                       np.arange((i+1)*ny*nz, (i+2)*ny*nz,dtype=np.int64))).T.flatten()))
        # i=nx-1 -> backward op
        jK = np.hstack((jK,
                        np.vstack((np.arange((nx-3)*ny*nz, (nx-2)*ny*nz, dtype=np.int64),
                                   np.arange((nx-2)*ny*nz, (nx-1)*ny*nz,dtype=np.int64),
                                   np.arange((nx-1)*ny*nz, nx*ny*nz,dtype=np.int64))).T.flatten()))

        Kx = sp.csr_matrix((val,(iK,jK)))

        # j=0 -> forward op
        jK = np.vstack((np.arange(nz,dtype=np.int64),
                        np.arange(nz,2*nz,dtype=np.int64),
                        np.arange(2*nz,3*nz,dtype=np.int64))).T.flatten()
        for j in np.arange(1,ny-1):
            jK = np.hstack((jK,
                            np.vstack((np.arange((j-1)*nz,j*nz,dtype=np.int64),
                                       np.arange(j*nz,(j+1)*nz,dtype=np.int64),
                                       np.arange((j+1)*nz,(j+2)*nz,dtype=np.int64))).T.flatten()))
        # j=ny-1 -> backward op
        jK = np.hstack((jK,
                        np.vstack((np.arange((ny-3)*nz,(ny-2)*nz,dtype=np.int64),
                                   np.arange((ny-2)*nz,(ny-1)*nz,dtype=np.int64),
                                   np.arange((ny-1)*nz,ny*nz,dtype=np.int64))).T.flatten()))
        tmp = jK.copy()
        for i in np.arange(1,nx):
            jK = np.hstack((jK, i*ny*nz+tmp))

        Ky = sp.csr_matrix((val,(iK,jK)))

        # k=0
        jK = np.arange(3,dtype=np.int64)
        for k in np.arange(1,nz-1):
            jK = np.hstack((jK,(k-1)+np.arange(3,dtype=np.int64)))
        # k=nz-1
        jK = np.hstack((jK, (nz-3)+np.arange(3,dtype=np.int64)))

        tmp = jK.copy()
        for j in np.arange(1,ny):
            jK = np.hstack((jK, j*nz+tmp))

        tmp = jK.copy()
        for i in np.arange(1,nx):
            jK = np.hstack((jK, i*ny*nz+tmp))

        Kz = sp.csr_matrix((val,(iK,jK)))

        return Kx, Ky, Kz

    def toXdmf(self, field, fieldname, filename):
        """
        Save a field in xdmf format (http://www.xdmf.org/index.php/Main_Page)

        INPUT
            field: data array of size equal to the number of cells in the grid
            fieldname: name to be assinged to the data (string)
            filename: name of xdmf file (string)
        """
        ox = self.x[0]
        oy = self.y[0]
        oz = self.z[0]
        dx = self.x[1] - self.x[0]
        dy = self.y[1] - self.y[0]
        dz = self.z[1] - self.z[0]
        nx = self.x.size
        ny = self.y.size
        nz = self.z.size

        f = open(filename+'.xmf','w')

        f.write('<?xml version="1.0" ?>\n')
        f.write('<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n')
        f.write('<Xdmf xmlns:xi="http://www.w3.org/2003/XInclude" Version="2.2">\n')
        f.write(' <Domain>\n')
        f.write('   <Grid Name="Structured Grid" GridType="Uniform">\n')
        f.write('     <Topology TopologyType="3DCORECTMesh" NumberOfElements="'+repr(nz)+' '+repr(ny)+' '+repr(nx)+'"/>\n')
        f.write('     <Geometry GeometryType="ORIGIN_DXDYDZ">\n')
        f.write('       <DataItem Dimensions="3 " NumberType="Float" Precision="4" Format="XML">\n')
        f.write('          '+repr(oz)+' '+repr(oy)+' '+repr(ox)+'\n')
        f.write('       </DataItem>\n')
        f.write('       <DataItem Dimensions="3 " NumberType="Float" Precision="4" Format="XML">\n')
        f.write('        '+repr(dz)+' '+repr(dy)+' '+repr(dx)+'\n')
        f.write('       </DataItem>\n')
        f.write('     </Geometry>\n')
        f.write('     <Attribute Name="'+fieldname+'" AttributeType="Scalar" Center="Node">\n')
        f.write('       <DataItem Dimensions="'+repr(nz)+' '+repr(ny)+' '+repr(nx)+'" NumberType="Float" Precision="4" Format="HDF">'+filename+'.h5:/'+fieldname+'</DataItem>\n')
        f.write('     </Attribute>\n')
        f.write('   </Grid>\n')
        f.write(' </Domain>\n')
        f.write('</Xdmf>\n')

        f.close()

        h5f = h5py.File(filename+'.h5', 'w')
        h5f.create_dataset(fieldname, data=field.reshape((nz,ny,nx), order='F').astype(np.float32))
        h5f.close()

class InvParams():
    def __init__(self, maxit, maxit_hypo, conv_hypo, Vlim, dmax, lagrangians,
                 invert_vel=True, invert_VsVp=True, hypo_2step=False,
                 use_sc=True, constr_sc=True, show_plots=True, save_V=False, verbose=True):
        """
        maxit       : max number of iterations
        maxit_hypo  :
        conv_hypo   : convergence criterion (units of distance)
        Vlim        : tuple holding (Vpmin, Vpmax, PAp, Vsmin, Vsmax, PAs) for
                        velocity penalties
                        PA is slope of penalty function
        dmax        : tuple holding max admissible corrections, i.e.
                        dVp_max
                        dx_max
                        dt_max
                        dVs_max
        lagrangians : tuple holding 4 values
                        λ : weight of smoothing constraint
                        γ : weight of penalty constraint
                        α : weight of velocity data point constraint
                        wzK   : weight for vertical smoothing (w.r. to horizontal smoothing)
        invert_vel  : perform velocity inversion if True (True by default)
        invert_VsVp : find Vs/Vp ratio rather that Vs (True by default)
        hypo_2step  : Hypocenter relocation done in 2 steps (False by default)
                        Step 1: longitude and latitude only allowed to vary
                        Step 2: all 4 parameters allowed to vary
        use_sc      : Use static corrections
        constr_sc   : Constrain sum of P-wave static corrections to zero
        show_plots  : show various plots during inversion (True by default)
        save_V      : save intermediate velocity models (False by default)
        verbose     : print information message about inversion progression (True by default)

        """
        self.maxit = maxit
        self.maxit_hypo = maxit_hypo
        self.conv_hypo = conv_hypo
        self.Vpmin = Vlim[0]
        self.Vpmax = Vlim[1]
        self.PAp   = Vlim[2]
        if len(Vlim) > 3:
            self.Vsmin = Vlim[3]
            self.Vsmax = Vlim[4]
            self.PAs   = Vlim[5]
        self.dVp_max = dmax[0]
        self.dx_max = dmax[1]
        self.dt_max = dmax[2]
        if len(dmax) > 3:
            self.dVs_max = dmax[3]
        self.λ = lagrangians[0]
        self.γ = lagrangians[1]
        self.α = lagrangians[2]
        self.wzK   = lagrangians[3]
        self.invert_vel = invert_vel
        self.invert_VsVp = invert_VsVp
        self.hypo_2step = hypo_2step
        self.use_sc = use_sc
        self.constr_sc = constr_sc
        self.show_plots = show_plots
        self.save_V = save_V
        self.verbose = verbose


def jointHypoVel(par, grid, data, rcv, Vinit, hinit, caldata=np.array([]), Vpts=np.array([])):
    """
    Joint hypocenter-velocity inversion on a regular grid

    Parameters
    ----------
    par     : instance of InvParams
    grid    : instance of Grid3D
    data    : a numpy array with 5 columns
               1st column is event ID number
               2nd column is arrival time
               3rd column is receiver index
               *** important ***
               data should sorted by event ID first, then by receiver index
    rcv:    : coordinates of receivers
               1st column is receiver easting
               2nd column is receiver northing
               3rd column is receiver elevation
    Vinit   : initial velocity model
    hinit   : initial hypocenter coordinate
               1st column is event ID number
               2nd column is origin time
               3rd column is receiver easting
               4th column is receiver northing
               5th column is receiver elevation
               *** important ***
               for efficiency reason when computing matrix M, initial hypocenters
               should _not_ be equal for any two event, e.g. they shoud all be
               different
    caldata : calibration shot data, numpy array with 8 columns
               1st column is cal shot ID number
               2nd column is arrival time
               3rd column is receiver index
               4th column is source easting
               5th column is source northing
               6th column is source elevation
               *** important ***
               cal shot data should sorted by cal shot ID first, then by receiver index
    Vpts    : known velocity points, numpy array with 4 columns
               1st column is velocity
               2nd column is easting
               3rd column is northing
               4th column is elevation

    Returns
    -------
    loc : hypocenter coordinates
    V   : velocity model
    sc  : static corrections
    res : residuals
    """

    evID = np.unique(data[:,0])
    nev = evID.size
    if par.use_sc:
        nsta = rcv.shape[0]
    else:
        nsta = 0

    sc = np.zeros(nsta)
    hyp0 = hinit.copy()
    nnodes = grid.getNumberOfNodes()

    rcv_data = np.empty((data.shape[0],3))
    for ne in np.arange(nev):
        indr = np.nonzero(data[:,0] == evID[ne])[0]
        for i in indr:
            rcv_data[i,:] = rcv[int(1.e-6+data[i,2])]

    if data.shape[0] > 0:
        tobs = data[:,1]
    else:
        tobs = np.array([])

    if caldata.shape[0] > 0:
        calID = np.unique(caldata[:,0])
        ncal = calID.size
        hcal = np.column_stack((caldata[:,0], np.zeros(caldata.shape[0]), caldata[:,3:]))
        tcal = caldata[:,1]
        rcv_cal = np.empty((caldata.shape[0],3))
        Msc_cal = []
        for nc in range(ncal):
            indr = np.nonzero(caldata[:,0] == calID[nc])[0]
            nst = np.sum(indr.size)
            for i in indr:
                rcv_cal[i,:] = rcv[int(1.e-6+caldata[i,2])]
            if par.use_sc:
                tmp = np.zeros((nst,nsta))
                for n in range(nst):
                    tmp[n,int(1.e-6+caldata[indr[n],2])] = 1.0
                Msc_cal.append(sp.csr_matrix(tmp))
    else:
        ncal = 0
        tcal = np.array([])

    if np.isscalar(Vinit):
        V = np.matrix(Vinit + np.zeros(nnodes))
        s = np.ones(nnodes)/Vinit
    else:
        V = np.matrix(Vinit)
        s = 1./Vinit
    V = V.reshape(-1,1)

    if Vpts.size > 0:
        if Vpts.shape[1] > 4:           # check if we have Vs data in array
            itmp = Vpts[:, 4] == 0
            Vpts = Vpts[itmp, :4]       # keep only Vp data

    if par.verbose:
        print('\n *** Joint hypocenter-velocity inversion ***\n')

    if par.invert_vel:
        resV = np.zeros(par.maxit+1)
        resAxb = np.zeros(par.maxit)

        P = sp.csr_matrix(np.ones(nnodes).reshape(-1,1))
        dP = sp.csr_matrix((np.ones(nnodes), (np.arange(nnodes,dtype=np.int64),
                                     np.arange(nnodes,dtype=np.int64))), shape=(nnodes,nnodes))

        deltam = np.ones(nnodes+nsta).reshape(-1,1)
        deltam[:,0] = 0.0
        deltam = sp.csr_matrix(deltam)
        if par.constr_sc:
            u1 = np.ones(nnodes+nsta).reshape(-1,1)
            u1[:nnodes,0] = 0.0
            u1 = sp.csr_matrix(u1)
        else:
            u1 = sp.csr_matrix(np.zeros(nnodes+nsta).reshape(-1,1))

        if Vpts.size > 0:
            if par.verbose:
                print('Building velocity data point matrix D')
                sys.stdout.flush()
            D = grid.computeD(Vpts[:,1:])
            D1 = sp.hstack((D, sp.coo_matrix((Vpts.shape[0],nsta)))).tocsr()
        else:
            D = 0.0

        if par.verbose:
            print('Building regularization matrix K')
            sys.stdout.flush()
        Kx, Ky, Kz = grid.computeK()
        Kx1 = sp.hstack((Kx, sp.coo_matrix((nnodes,nsta)))).tocsr()
        KtK = Kx1.T * Kx1
        Ky1 = sp.hstack((Ky, sp.coo_matrix((nnodes,nsta)))).tocsr()
        KtK += Ky1.T * Ky1
        Kz1 = sp.hstack((Kz, sp.coo_matrix((nnodes,nsta)))).tocsr()
        KtK += par.wzK * Kz1.T * Kz1
        nK = spl.norm(KtK)
        Kx = Kx.tocsr()
        Ky = Ky.tocsr()
        Kz = Kz.tocsr()
    else:
        resV = None
        resAxb = None

    if par.verbose:
        print('\nStarting iterations')

    for it in np.arange(par.maxit):

        if par.invert_vel:
            if par.verbose:
                print('\nIteration {0:d} - Updating velocity model'.format(it+1))
                print('                Updating penalty vector')
                sys.stdout.flush()

            # compute vector C
            cx = Kx * V
            cy = Ky * V
            cz = Kz * V

            # compute dP/dV, matrix of penalties derivatives
            for n in np.arange(nnodes):
                if V[n,0] < par.Vpmin:
                    P[n,0] = par.PAp * (par.Vpmin-V[n,0])
                    dP[n,n] = -par.PAp
                elif V[n,0] > par.Vpmax:
                    P[n,0] = par.PAp * (V[n,0]-par.Vpmax)
                    dP[n,n] = par.PAp
                else:
                    P[n,0] = 0.0
                    dP[n,n] = 0.0
            if par.verbose:
                npel = np.sum( P != 0.0 )
                if npel > 0:
                    print('                  Penalties applied at {0:d} nodes'.format(npel))

            if par.verbose:
                print('                Raytracing')
                sys.stdout.flush()

            if nev > 0:
                hyp = np.empty((data.shape[0],5))
                for ne in np.arange(nev):
                    indh = np.nonzero(hyp0[:,0] == evID[ne])[0]
                    indr = np.nonzero(data[:,0] == evID[ne])[0]
                    for i in indr:
                        hyp[i,:] = hyp0[indh[0],:]
                tcalc, rays, v0, Mev = grid.raytrace(s, hyp, rcv_data)
            else:
                tcalc = np.array([])

            if ncal > 0:
                tcalc_cal, _, _, Mcal = grid.raytrace(s, hcal, rcv_cal)
            else:
                tcalc_cal = np.array([])

            r1a = tobs - tcalc
            r1 = tcal - tcalc_cal
            if r1a.size > 0:
                r1 = np.hstack((np.zeros(data.shape[0]-4*nev), r1))

            if par.show_plots:
                plt.figure(1)
                plt.cla()
                plt.plot(r1a,'o')
                plt.title('Residuals - Iteration {0:d}'.format(it+1))
                plt.show(block=False)
                plt.pause(0.0001)

            resV[it] = np.linalg.norm(np.hstack((r1a, r1)))
            r1 = np.matrix( r1.reshape(-1,1) )
            r1a = np.matrix( r1a.reshape(-1,1) )

            # initializing matrix M; matrix of partial derivatives of velocity dt/dV
            if par.verbose:
                print('                Building matrix M')
                sys.stdout.flush()

            M1 = None
            ir1 = 0
            for ne in range(nev):
                if par.verbose:
                    print('                  Event ID '+str(int(1.e-6+evID[ne])))
                    sys.stdout.flush()

                indh = np.nonzero(hyp0[:,0] == evID[ne])[0]
                indr = np.nonzero(data[:,0] == evID[ne])[0]

                nst = np.sum(indr.size)
                nst2 = nst-4
                H = np.ones((nst,4))
                for ns in range(nst):
                    raysi = rays[indr[ns]]
                    V0 = v0[indr[ns]]

                    d = (raysi[1,:]-hyp0[indh,2:]).flatten()
                    ds = np.sqrt( np.sum(d*d) )
                    H[ns,1] = -1./V0 * d[0]/ds
                    H[ns,2] = -1./V0 * d[1]/ds
                    H[ns,3] = -1./V0 * d[2]/ds

                Q, _ = np.linalg.qr(H, mode='complete')
                T = sp.csr_matrix(Q[:, 4:]).T
                M = sp.csr_matrix(Mev[ne], shape=(nst,nnodes))
                if par.use_sc:
                    Msc = np.zeros((nst,nsta))
                    for ns in range(nst):
                        Msc[ns,int(1.e-6+data[indr[ns],2])] = 1.
                    M = sp.hstack((M, sp.csr_matrix(Msc)))

                M = T * M

                if M1 == None:
                    M1 = M
                else:
                    M1 = sp.vstack((M1, M))

                r1[ir1+np.arange(nst2, dtype=np.int64)] = T.dot(r1a[indr])
                ir1 += nst2

            for nc in range(ncal):
                M = Mcal[nc]
                if par.use_sc:
                    M = sp.hstack((M, Msc_cal[nc]))
                if M1 == None:
                    M1 = M
                else:
                    M1 = sp.vstack((M1, M))

            if par.verbose:
                print('                Assembling matrices and solving system')
                sys.stdout.flush()

            s = -np.sum(sc) # u1.T * deltam

            dP1 = sp.hstack((dP, sp.csr_matrix(np.zeros((nnodes,nsta))))).tocsr()  # dP prime

            # compute A & h for inversion

            M1 = M1.tocsr()

            A = M1.T * M1
            nM = spl.norm(A)

            λ = par.λ * nM / nK

            A += λ*KtK

            tmp = dP1.T * dP1
            nP = spl.norm(tmp)
            if nP != 0.0:
                γ = par.γ * nM / nP
            else:
                γ = par.γ

            A += γ*tmp
            A += u1 * u1.T

            b = M1.T * r1
            tmp2x = Kx1.T * cx
            tmp2y = Ky1.T * cy
            tmp2z = Kz1.T * cz
            tmp3 = dP1.T * P
            tmp = u1 * s
            b += - λ*tmp2x - λ*tmp2y - par.wzK*λ*tmp2z - γ*tmp3 - tmp

            if Vpts.shape[0] > 0:
                tmp = D1.T * D1
                nD = spl.norm(tmp)
                α = par.α * nM / nD
                A += α * tmp
                b += α * D1.T * (Vpts[:,0].reshape(-1,1) - D*V )

            if par.verbose:
                print('                  calling minres with system of size {0:d} x {1:d}'.format(A.shape[0], A.shape[1]))
                sys.stdout.flush()
            x = spl.minres(A, b.getA1())

            deltam = np.matrix(x[0].reshape(-1,1))
            resAxb[it] = np.linalg.norm(A*deltam - b)

            dmean = np.mean( np.abs(deltam[:nnodes]) )
            if dmean > par.dVp_max:
                if par.verbose:
                    print('                Scaling Vp perturbations by {0:e}'.format(par.dVp_max/dmean))
                deltam[:nnodes] = deltam[:nnodes] * par.dVp_max/dmean

            V += np.matrix(deltam[:nnodes].reshape(-1,1))
            s = 1. / V.getA1()
            sc += deltam[nnodes:,0].getA1()

            if par.save_V:
                if par.verbose:
                    print('                Saving Velocity model')
                grid.toXdmf(V.getA(), 'Vp', 'Vp{0:02d}'.format(it+1))

            grid.set_slowness(s)

        if nev > 0:
            if par.verbose:
                print('Iteration {0:d} - Relocating events'.format(it+1))
                sys.stdout.flush()

            if grid.nthreads == 1 or nev < 1.5*grid.nthreads:
                for ne in range(nev):
                    _reloc(ne, par, grid, evID, hyp0, data, rcv, tobs)
            else:
                # run in parallel
                blk_size = np.zeros((grid.nthreads,), dtype=np.int64)
                nj = nev
                while nj > 0:
                    for n in range(grid.nthreads):
                        blk_size[n] += 1
                        nj -= 1
                        if nj == 0:
                            break
                processes = []
                blk_start = 0
                h_queue = Queue()
                for n in range(grid.nthreads):
                    blk_end = blk_start + blk_size[n]
                    p = Process(target=_rl_worker,
                                args=(n, blk_start, blk_end, par, grid, evID, hyp0, data, rcv, tobs, h_queue),
                                daemon=True)
                    processes.append(p)
                    p.start()
                    blk_start += blk_size[n]

                for ne in range(nev):
                    h, indh = h_queue.get()
                    hyp0[indh, :] = h

                for p in processes:
                    p.join()

    if par.invert_vel:
        if nev > 0:
            hyp = np.empty((data.shape[0],5))
            for ne in np.arange(nev):
                indh = np.nonzero(hyp0[:,0] == evID[ne])[0]
                indr = np.nonzero(data[:,0] == evID[ne])[0]
                for i in indr:
                    hyp[i,:] = hyp0[indh[0],:]
            tcalc = grid.raytrace(s, hyp, rcv_data)
        else:
            tcalc = np.array([])

        if ncal > 0:
            tcalc_cal = grid.raytrace(s, hcal, rcv_cal)
        else:
            tcalc_cal = np.array([])

        r1a = tobs - tcalc
        r1 = tcal - tcalc_cal
        if r1a.size > 0:
            r1 = np.hstack((np.zeros(data.shape[0]-4*nev), r1))

        if par.show_plots:
            plt.figure(1)
            plt.cla()
            plt.plot(r1a,'o')
            plt.title('Residuals - Final step')
            plt.show(block=False)
            plt.pause(0.0001)

        resV[-1] = np.linalg.norm(np.hstack((r1a, r1)))

        r1 = np.matrix( r1.reshape(-1,1) )
        r1a = np.matrix( r1a.reshape(-1,1) )

    if par.verbose:
        print('\n ** Inversion complete **\n')

    return hyp0, V.getA1(), sc, (resV, resAxb)


def _rl_worker(thread_no, istart, iend, par, grid, evID, hyp0, data, rcv, tobs, h_queue):
    for ne in range(istart, iend):
        h, indh = _reloc(ne, par, grid, evID, hyp0, data, rcv, tobs, thread_no)
        h_queue.put((h, indh))
    h_queue.close()


def _reloc(ne, par, grid, evID, hyp0, data, rcv, tobs, thread_no=None):

    if par.verbose:
        print('                Updating event ID {0:d} ({1:d}/{2:d})'.format(int(1.e-6+evID[ne]), ne+1, evID.size))
        sys.stdout.flush()

    indh = np.nonzero(hyp0[:,0] == evID[ne])[0][0]
    indr = np.nonzero(data[:,0] == evID[ne])[0]

    hyp_save = hyp0[indh,:].copy()

    nst = np.sum(indr.size)

    hyp = np.empty((nst,5))
    stn = np.empty((nst,3))
    for i in range(nst):
        hyp[i,:] = hyp0[indh,:]
        stn[i,:] = rcv[int(1.e-6+data[indr[i],2]),:]

    if par.hypo_2step:
        if par.verbose:
            print('                  Updating latitude & longitude', end='')
            sys.stdout.flush()
        H = np.ones((nst,2))
        for itt in range(par.maxit_hypo):
            for i in range(nst):
                hyp[i,:] = hyp0[indh,:]

            tcalc, rays, v0 = grid.raytrace(None, hyp, stn, thread_no)
            for ns in range(nst):
                raysi = rays[ns]
                V0 = v0[ns]

                d = (raysi[1,:]-hyp0[indh,2:]).flatten()
                ds = np.sqrt( np.sum(d*d) )
                H[ns,0] = -1./V0 * d[0]/ds
                H[ns,1] = -1./V0 * d[1]/ds

            r = tobs[indr] - tcalc
            x = np.linalg.lstsq(H,r,rcond=None)
            deltah = x[0]

            if np.sum( np.isfinite(deltah) ) != deltah.size:
                try:
                    U,S,VVh = np.linalg.svd(H.T.dot(H)+1e-9*np.eye(2))
                    VV = VVh.T
                    deltah = np.dot( VV, np.dot(U.T, H.T.dot(r))/S)
                except np.linalg.linalg.LinAlgError:
                    print(' - Event could not be relocated, resetting and exiting')
                    hyp0[indh,:] = hyp_save
                    return hyp_save, indh

#            for n in range(2):
#                if np.abs(deltah[n]) > par.dx_max:
#                    deltah[n] = par.dx_max * np.sign(deltah[n])

            new_hyp = hyp0[indh,:].copy()
            new_hyp[2:4] += deltah
            if grid.is_outside(new_hyp[2:].reshape((1,3))):
                print('  Event could not be relocated inside the grid ({0:f}, {1:f}, {2:f}), resetting and exiting'.format(new_hyp[2], new_hyp[3], new_hyp[4]))
                hyp0[indh,:] = hyp_save
                return hyp_save, indh

            hyp0[indh,2:4] += deltah

            if np.sum(np.abs(deltah)<par.conv_hypo) == 2:
                if par.verbose:
                    print(' - converged at iteration '+str(itt+1))
                    sys.stdout.flush()
                break

        else:
            if par.verbose:
                print(' - reached max number of iterations')
                sys.stdout.flush()

    if par.verbose:
        print('                  Updating all hypocenter params', end='')
        sys.stdout.flush()

    H = np.ones((nst,4))
    for itt in range(par.maxit_hypo):
        for i in range(nst):
            hyp[i,:] = hyp0[indh,:]

        tcalc, rays, v0 = grid.raytrace(None, hyp, stn, thread_no)
        for ns in range(nst):
            raysi = rays[ns]
            V0 = v0[ns]

            d = (raysi[1,:]-hyp0[indh,2:]).flatten()
            ds = np.sqrt( np.sum(d*d) )
            H[ns,1] = -1./V0 * d[0]/ds
            H[ns,2] = -1./V0 * d[1]/ds
            H[ns,3] = -1./V0 * d[2]/ds

        r = tobs[indr] - tcalc
        x = np.linalg.lstsq(H,r,rcond=None)
        deltah = x[0]

        if np.sum( np.isfinite(deltah) ) != deltah.size:
            try:
                U,S,VVh = np.linalg.svd(H.T.dot(H)+1e-9*np.eye(4))
                VV = VVh.T
                deltah = np.dot( VV, np.dot(U.T, H.T.dot(r))/S)
            except np.linalg.linalg.LinAlgError:
                print('  Event could not be relocated, resetting and exiting')
                hyp0[indh,:] = hyp_save
                return hyp_save, indh

#        if np.abs(deltah[0]) > par.dt_max:
#            deltah[0] = par.dt_max * np.sign(deltah[0])
#        for n in range(1,4):
#            if np.abs(deltah[n]) > par.dx_max:
#                deltah[n] = par.dx_max * np.sign(deltah[n])

        new_hyp = hyp0[indh,1:] + deltah
        if grid.is_outside(new_hyp[1:].reshape((1,3))):
            print('  Event could not be relocated inside the grid ({0:f}, {1:f}, {2:f}), resetting and exiting'.format(new_hyp[2], new_hyp[3], new_hyp[4]))
            hyp0[indh,:] = hyp_save
            return hyp_save, indh

        hyp0[indh,1:] += deltah

        if np.sum(np.abs(deltah[1:])<par.conv_hypo) == 3:
            if par.verbose:
                print(' - converged at iteration '+str(itt+1))
                sys.stdout.flush()
            break

    else:
        if par.verbose:
            print(' - reached max number of iterations')
            sys.stdout.flush()

    return hyp0[indh,:], indh


def jointHypoVelPS(par, grid, data, rcv, Vinit, hinit, caldata=np.array([]), Vpts=np.array([])):
    """
    Joint hypocenter-velocity inversion on a regular grid

    Parameters
    ----------
    par     : instance of InvParams
    grid    : instances of Grid3D
    data    : a numpy array with 5 columns
               1st column is event ID number
               2nd column is arrival time
               3rd column is receiver index
               4th column is code for wave phase: 0 for P-wave and 1 for S-wave
               *** important ***
               data should sorted by event ID first, then by receiver index
    rcv:    : coordinates of receivers
               1st column is receiver easting
               2nd column is receiver northing
               3rd column is receiver elevation
    Vinit   : tuple with initial velocity model (P-wave first, S-wave second)
    hinit   : initial hypocenter coordinate
               1st column is event ID number
               2nd column is origin time
               3rd column is receiver easting
               4th column is receiver northing
               5th column is receiver elevation
               *** important ***
               for efficiency reason when computing matrix M, initial hypocenters
               should _not_ be equal for any two event, e.g. they shoud all be
               different
    caldata : calibration shot data, numpy array with 8 columns
               1st column is cal shot ID number
               2nd column is arrival time
               3rd column is receiver index
               4th column is source easting
               5th column is source northing
               6th column is source elevation
               7th column is code for wave phase: 0 for P-wave and 1 for S-wave
               *** important ***
               cal shot data should sorted by cal shot ID first, then by receiver index
    Vpts    : known velocity points, numpy array with 4 columns
               1st column is velocity
               2nd column is easting
               3rd column is northing
               4th column is elevation
               5th column is code for wave phase: 0 for P-wave and 1 for S-wave

    Returns
    -------
    loc : hypocenter coordinates
    V   : velocity models (tuple holding Vp and Vs)
    sc  : static corrections
    res : residuals
    """

    if grid.nthreads > 1:
        # we need a second instance for parallel computations
        grid_s = Grid3D(grid.x, grid.y, grid.z, grid.nthreads)
    else:
        grid_s = grid

    evID = np.unique(data[:,0])
    nev = evID.size
    if par.use_sc:
        nsta = rcv.shape[0]
    else:
        nsta = 0

    sc_p = np.zeros(nsta)
    sc_s = np.zeros(nsta)
    hyp0 = hinit.copy()
    nnodes = grid.getNumberOfNodes()

    # sort data by seismic phase (P-wave first S-wave second)
    indp = data[:,3] == 0.0
    inds = data[:,3] == 1.0
    nttp = np.sum( indp )
    ntts = np.sum( inds )
    datap = data[indp,:]
    datas = data[inds,:]
    data = np.vstack((datap, datas))

    rcv_datap = np.empty((nttp,3))
    rcv_datas = np.empty((ntts,3))
    for ne in np.arange(nev):
        indr = np.nonzero(np.logical_and(data[:,0] == evID[ne], indp))[0]
        for i in indr:
            rcv_datap[i,:] = rcv[int(1.e-6+data[i,2])]
        indr = np.nonzero(np.logical_and(data[:,0] == evID[ne], inds))[0]
        for i in indr:
            rcv_datas[i-nttp,:] = rcv[int(1.e-6+data[i,2])]


    if data.shape[0] > 0:
        tobs = data[:,1]
    else:
        tobs = np.array([])

    if caldata.shape[0] > 0:
        calID = np.unique(caldata[:,0])
        ncal = calID.size

        # sort data by seismic phase (P-wave first S-wave second)
        indcalp = caldata[:,6] == 0.0
        indcals = caldata[:,6] == 1.0
        nttcalp = np.sum( indcalp )
        nttcals = np.sum( indcals )
        caldatap = caldata[indcalp,:]
        caldatas = caldata[indcals,:]
        caldata = np.vstack((caldatap, caldatas))

        hcalp = np.column_stack((caldata[indcalp,0], np.zeros(nttcalp), caldata[indcalp,3:6]))
        hcals = np.column_stack((caldata[indcals,0], np.zeros(nttcals), caldata[indcals,3:6]))

        rcv_calp = np.empty((nttcalp,3))
        rcv_cals = np.empty((nttcals,3))

        tcal = caldata[:,1]
        Msc_cal = []
        for nc in range(ncal):
            indrp = np.nonzero(np.logical_and(caldata[:,0] == calID[nc], indcalp))[0]
            for i in indrp:
                rcv_calp[i,:] = rcv[int(1.e-6+caldata[i,2])]
            indrs = np.nonzero(np.logical_and(caldata[:,0] == calID[nc], indcals))[0]
            for i in indrs:
                rcv_cals[i-nttcalp,:] = rcv[int(1.e-6+caldata[i,2])]

            if par.use_sc:
                Mpsc = np.zeros((indrp.size,nsta))
                Mssc = np.zeros((indrs.size,nsta))
                for ns in range(indrp.size):
                    Mpsc[ns,int(1.e-6+caldata[indrp[ns],2])] = 1.
                for ns in range(indrs.size):
                    Mssc[ns,int(1.e-6+caldata[indrs[ns],2])] = 1.

                Msc_cal.append( sp.block_diag((sp.csr_matrix(Mpsc), sp.csr_matrix(Mssc))) )

    else:
        ncal = 0
        tcal = np.array([])

    if np.isscalar(Vinit[0]):
        Vp = np.matrix(Vinit[0] + np.zeros(nnodes))
        s_p = np.ones(nnodes)/Vinit[0]
    else:
        Vp = np.matrix(Vinit[0])
        s_p = 1./Vinit[0]
    Vp = Vp.reshape(-1,1)
    if np.isscalar(Vinit[1]):
        Vs = np.matrix(Vinit[1] + np.zeros(nnodes))
        s_s = np.ones(nnodes)/Vinit[1]
    else:
        Vs = np.matrix(Vinit[1])
        s_s = 1./Vinit[1]
    Vs = Vs.reshape(-1,1)
    if par.invert_VsVp:
        VsVp = Vs/Vp
        V = np.vstack((Vp, VsVp))
    else:
        V = np.vstack((Vp, Vs))

    if par.verbose:
        print('\n *** Joint hypocenter-velocity inversion  -- P and S-wave data ***\n')

    if par.invert_vel:
        resV = np.zeros(par.maxit+1)
        resAxb = np.zeros(par.maxit)

        P = sp.csr_matrix(np.ones(2*nnodes).reshape(-1,1))
        dP = sp.csr_matrix((np.ones(2*nnodes), (np.arange(2*nnodes,dtype=np.int64),
                            np.arange(2*nnodes,dtype=np.int64))),
                            shape=(2*nnodes,2*nnodes))

        deltam = np.ones(2*nnodes+2*nsta).reshape(-1,1)
        deltam[:,0] = 0.0
        deltam = sp.csr_matrix(deltam)

        if par.constr_sc:
            u1 = np.ones(2*nnodes+2*nsta).reshape(-1,1)
            u1[:2*nnodes,0] = 0.0
            u1[(2*nnodes+nsta):,0] = 0.0
            u1 = sp.csr_matrix(u1)
        else:
            u1 = sp.csr_matrix(np.zeros(2*nnodes+2*nsta).reshape(-1,1))

        if Vpts.size > 0:

            if par.verbose:
                print('Building velocity data point matrix D')
                sys.stdout.flush()

            if par.invert_VsVp:
                Vpts2 = Vpts.copy()
                i_p = np.nonzero(Vpts[:,4]==0.0)[0]
                i_s = np.nonzero(Vpts[:,4]==1.0)[0]
                for i in i_s:
                    for ii in i_p:
                        d = np.sqrt( np.sum( (Vpts2[i,1:4]-Vpts[ii,1:4])**2 ) )
                        if d < 0.00001:
                            Vpts2[i,0] = Vpts[i,0]/Vpts[ii,0]
                            break
                    else:
                        raise ValueError('Missing Vp data point for Vs data at ({0:f}, {1:f}, {2:f})'.format(Vpts[i,1], Vpts[i,2], Vpts[i,3]))

            else:
                Vpts2 = Vpts

            if par.invert_VsVp:
                D = grid.computeD(Vpts2[:,1:4])
                D = sp.hstack((D, sp.coo_matrix(D.shape))).tocsr()
            else:
                i_p = Vpts2[:,4]==0.0
                i_s = Vpts2[:,4]==1.0
                Dp = grid.computeD(Vpts2[i_p,1:4])
                Ds = grid.computeD(Vpts2[i_s,1:4])
                D = sp.block_diag((Dp, Ds)).tocsr()

            D1 = sp.hstack((D, sp.csr_matrix((Vpts2.shape[0],2*nsta)))).tocsr()
        else:
            D = 0.0

        if par.verbose:
            print('Building regularization matrix K')
            sys.stdout.flush()
        Kx, Ky, Kz = grid.computeK()
        Kx = sp.block_diag((Kx, Kx))
        Ky = sp.block_diag((Ky, Ky))
        Kz = sp.block_diag((Kz, Kz))
        Kx1 = sp.hstack((Kx, sp.coo_matrix((2*nnodes,2*nsta)))).tocsr()
        KtK = Kx1.T * Kx1
        Ky1 = sp.hstack((Ky, sp.coo_matrix((2*nnodes,2*nsta)))).tocsr()
        KtK += Ky1.T * Ky1
        Kz1 = sp.hstack((Kz, sp.coo_matrix((2*nnodes,2*nsta)))).tocsr()
        KtK += par.wzK * Kz1.T * Kz1
        nK = spl.norm(KtK)
        Kx = Kx.tocsr()
        Ky = Ky.tocsr()
        Kz = Kz.tocsr()
    else:
        resV = None
        resAxb = None

    if par.invert_VsVp:
        VsVpmin = par.Vsmin/par.Vpmax
        VsVpmax = par.Vsmax/par.Vpmin

    if par.verbose:
        print('\nStarting iterations')

    for it in np.arange(par.maxit):

        if par.invert_vel:
            if par.verbose:
                print('\nIteration {0:d} - Updating velocity model'.format(it+1))
                print('                Updating penalty vector')
                sys.stdout.flush()

            # compute vector C
            cx = Kx * V
            cy = Ky * V
            cz = Kz * V

            # compute dP/dV, matrix of penalties derivatives
            for n in np.arange(nnodes):
                if V[n,0] < par.Vpmin:
                    P[n,0] = par.PAp * (par.Vpmin-V[n,0])
                    dP[n,n] = -par.PAp
                elif V[n,0] > par.Vpmax:
                    P[n,0] = par.PAp * (V[n,0]-par.Vpmax)
                    dP[n,n] = par.PAp
                else:
                    P[n,0] = 0.0
                    dP[n,n] = 0.0
            for n in np.arange(nnodes, 2*nnodes):
                if par.invert_VsVp:
                    if VsVp[n-nnodes,0] < VsVpmin:
                        P[n,0] = par.PAs * (VsVpmin-VsVp[n-nnodes,0])
                        dP[n,n] = -par.PAs
                    elif VsVp[n-nnodes,0] > VsVpmax:
                        P[n,0] = par.PAs * (VsVp[n-nnodes,0]-VsVpmax)
                        dP[n,n] = par.PAs
                    else:
                        P[n,0] = 0.0
                        dP[n,n] = 0.0
                else:
                    if Vs[n-nnodes,0] < par.Vsmin:
                        P[n,0] = par.PAs * (par.Vsmin-Vs[n-nnodes,0])
                        dP[n,n] = -par.PAs
                    elif Vs[n-nnodes,0] > par.Vsmax:
                        P[n,0] = par.PAs * (Vs[n-nnodes,0]-par.Vsmax)
                        dP[n,n] = par.PAs
                    else:
                        P[n,0] = 0.0
                        dP[n,n] = 0.0

            if par.verbose:
                npel = np.sum( P[:nnodes,0] != 0.0 )
                if npel > 0:
                    print('                  P-wave penalties applied at {0:d} nodes'.format(npel))
                npel = np.sum( P[nnodes:2*nnodes,0] != 0.0 )
                if npel > 0:
                    print('                  S-wave penalties applied at {0:d} nodes'.format(npel))


            if par.verbose:
                print('                Raytracing')
                sys.stdout.flush()

            if nev > 0:
                hyp = np.empty((nttp,5))
                for ne in np.arange(nev):
                    indh = np.nonzero(hyp0[:,0] == evID[ne])[0]
                    indrp = np.nonzero(np.logical_and(data[:,0] == evID[ne], indp))[0]
                    for i in indrp:
                        hyp[i,:] = hyp0[indh[0],:]

                tcalcp, raysp, v0p, Mevp = grid.raytrace(s_p, hyp, rcv_datap)

                hyp = np.empty((ntts,5))
                for ne in np.arange(nev):
                    indh = np.nonzero(hyp0[:,0] == evID[ne])[0]
                    indrs = np.nonzero(np.logical_and(data[:,0] == evID[ne], inds))[0]
                    for i in indrs:
                        hyp[i-nttp,:] = hyp0[indh[0],:]
                tcalcs, rayss, v0s, Mevs = grid_s.raytrace(s_s, hyp, rcv_datas)

                tcalc = np.hstack((tcalcp, tcalcs))
                v0 = np.hstack((v0p, v0s))
                rays = []
                for r in raysp:
                    rays.append( r )
                for r in rayss:
                    rays.append( r )

                # Merge Mevp & Mevs
                Mev = [None] * nev
                for ne in np.arange(nev):

                    Mp = Mevp[ne]
                    Ms = Mevs[ne]

                    if par.invert_VsVp:
                        # Block 1991, p. 45
                        tmp1 = Ms.multiply(matlib.repmat(VsVp.T, Ms.shape[0], 1))
                        tmp2 = Ms.multiply(matlib.repmat(Vp.T, Ms.shape[0], 1))
                        tmp2 = sp.hstack((tmp1, tmp2))
                        tmp1 = sp.hstack((Mp, sp.csr_matrix(Mp.shape)))
                        Mev[ne] = sp.vstack((tmp1, tmp2))
                    else:
                        Mev[ne] = sp.block_diag((Mp, Ms))

                    if par.use_sc:
                        indrp = np.nonzero(np.logical_and(data[:,0] == evID[ne], indp))[0]
                        indrs = np.nonzero(np.logical_and(data[:,0] == evID[ne], inds))[0]

                        Mpsc = np.zeros((indrp.size,nsta))
                        Mssc = np.zeros((indrs.size,nsta))
                        for ns in range(indrp.size):
                            Mpsc[ns,int(1.e-6+data[indrp[ns],2])] = 1.
                        for ns in range(indrs.size):
                            Mssc[ns,int(1.e-6+data[indrs[ns],2])] = 1.

                        Msc = sp.block_diag((sp.csr_matrix(Mpsc), sp.csr_matrix(Mssc)))
                        # add terms for station corrections after terms for velocity because
                        # solution vector contains [Vp Vs sc_p sc_s] in that order
                        Mev[ne] = sp.hstack((Mev[ne], Msc))

            else:
                tcalc = np.array([])

            if ncal > 0:
                tcalcp_cal, _, _, Mp_cal = grid.raytrace(s_p, hcalp, rcv_calp)
                if nttcals > 0:
                    tcalcs_cal, _, _, Ms_cal = grid_s.raytrace(s_s, hcals, rcv_cals)
                    tcalc_cal = np.hstack((tcalcp_cal, tcalcs_cal))
                else:
                    tcalc_cal = tcalcp_cal
            else:
                tcalc_cal = np.array([])

            r1a = tobs - tcalc
            r1 = tcal - tcalc_cal
            if r1a.size > 0:
                r1 = np.hstack((np.zeros(data.shape[0]-4*nev), r1))

            r1 = np.matrix( r1.reshape(-1,1) )
            r1a = np.matrix( r1a.reshape(-1,1) )

            resV[it] = np.linalg.norm(np.hstack((tobs-tcalc, tcal-tcalc_cal)))

            if par.show_plots:
                plt.figure(1)
                plt.cla()
                plt.plot(r1a,'o')
                plt.title('Residuals - Iteration {0:d}'.format(it+1))
                plt.show(block=False)
                plt.pause(0.0001)

            # initializing matrix M; matrix of partial derivatives of velocity dt/dV
            if par.verbose:
                print('                Building matrix M')
                sys.stdout.flush()

            M1 = None
            ir1 = 0
            for ne in range(nev):
                if par.verbose:
                    print('                  Event ID '+str(int(1.e-6+evID[ne])))
                    sys.stdout.flush()

                indh = np.nonzero(hyp0[:,0] == evID[ne])[0]
                indr = np.nonzero(data[:,0] == evID[ne])[0]

                nst = np.sum(indr.size)
                nst2 = nst-4
                H = np.ones((nst,4))
                for ns in range(nst):
                    raysi = rays[indr[ns]]
                    V0 = v0[indr[ns]]

                    d = (raysi[1,:]-hyp0[indh,2:]).flatten()
                    ds = np.sqrt( np.sum(d*d) )
                    H[ns,1] = -1./V0 * d[0]/ds
                    H[ns,2] = -1./V0 * d[1]/ds
                    H[ns,3] = -1./V0 * d[2]/ds

                Q, _ = np.linalg.qr(H, mode='complete')
                T = sp.csr_matrix(Q[:, 4:]).T
                M = T * Mev[ne]

                if M1 == None:
                    M1 = M
                else:
                    M1 = sp.vstack((M1, M))

                r1[ir1+np.arange(nst2, dtype=np.int64)] = T.dot(r1a[indr])
                ir1 += nst2

            for nc in range(ncal):
                Mp = Mp_cal[nc]
                if nttcals > 0:
                    Ms = Ms_cal[nc]
                else:
                    Ms = sp.csr_matrix([])

                if par.invert_VsVp:
                    if nttcals > 0:
                        # Block 1991, p. 45
                        tmp1 = Ms.multiply(matlib.repmat(VsVp.T, Ms.shape[0], 1))
                        tmp2 = Ms.multiply(matlib.repmat(Vp.T, Ms.shape[0], 1))
                        tmp2 = sp.hstack((tmp1, tmp2))
                        tmp1 = sp.hstack((Mp, sp.csr_matrix(Mp.shape)))
                        M = sp.vstack((tmp1, tmp2))
                    else:
                        M = sp.hstack((Mp, sp.csr_matrix(Mp.shape)))
                else:
                    M = sp.block_diag((Mp, Ms))

                if par.use_sc:
                    M = sp.hstack((M, Msc_cal[nc]))

                if M1 == None:
                    M1 = M
                else:
                    M1 = sp.vstack((M1, M))

            if par.verbose:
                print('                Assembling matrices and solving system')
                sys.stdout.flush()

            s = -np.sum(sc_p)

            dP1 = sp.hstack((dP, sp.csr_matrix((2*nnodes,2*nsta)))).tocsr()  # dP prime

            # compute A & h for inversion

            M1 = M1.tocsr()

            A = M1.T * M1

            nM = spl.norm(A)
            λ = par.λ * nM / nK

            A += λ*KtK

            tmp = dP1.T * dP1
            nP = spl.norm(tmp)
            if nP != 0.0:
                γ = par.γ * nM / nP
            else:
                γ = par.γ

            A += γ*tmp
            A += u1 * u1.T

            b = M1.T * r1
            tmp2x = Kx1.T * cx
            tmp2y = Ky1.T * cy
            tmp2z = Kz1.T * cz
            tmp3 = dP1.T * P
            tmp = u1 * s
            b += -λ*tmp2x - λ*tmp2y - par.wzK*λ*tmp2z - γ*tmp3 - tmp

            if Vpts2.shape[0] > 0:
                tmp = D1.T * D1
                nD = spl.norm(tmp)
                α = par.α * nM / nD
                A += α * tmp
                b += α * D1.T * (Vpts2[:,0].reshape(-1,1) - D*V )

            if par.verbose:
                print('                  calling minres with system of size {0:d} x {1:d}'.format(A.shape[0], A.shape[1]))
                sys.stdout.flush()
            x = spl.minres(A, b.getA1())

            deltam = np.matrix(x[0].reshape(-1,1))
            resAxb[it] = np.linalg.norm(A*deltam - b)

            dmean = np.mean( np.abs(deltam[:nnodes]) )
            if dmean > par.dVp_max:
                if par.verbose:
                    print('                Scaling Vp perturbations by {0:e}'.format(par.dVp_max/dmean))
                deltam[:nnodes] = deltam[:nnodes] * par.dVp_max/dmean
            dmean = np.mean( np.abs(deltam[nnodes:2*nnodes]) )
            if dmean > par.dVs_max:
                if par.verbose:
                    print('                Scaling Vs perturbations by {0:e}'.format(par.dVs_max/dmean))
                deltam[nnodes:2*nnodes] = deltam[nnodes:2*nnodes] * par.dVs_max/dmean

            V += np.matrix(deltam[:2*nnodes].reshape(-1,1))
            Vp = V[:nnodes]
            if par.invert_VsVp:
                VsVp = V[nnodes:2*nnodes]
                Vs = np.multiply( VsVp, Vp )
            else:
                Vs = V[nnodes:2*nnodes]
            s_p = 1./Vp
            s_s = 1./Vs
            sc_p += deltam[2*nnodes:2*nnodes+nsta,0].getA1()
            sc_s += deltam[2*nnodes+nsta:,0].getA1()

            if par.save_V:
                if par.verbose:
                    print('                Saving Velocity models')
                grid.toXdmf(Vp.getA(), 'Vp', 'Vp{0:02d}'.format(it+1))
                if par.invert_VsVp:
                    grid.toXdmf(VsVp.getA(), 'VsVp', 'VsVp{0:02d}'.format(it+1))
                grid.toXdmf(Vs.getA(), 'Vs', 'Vs{0:02d}'.format(it+1))

        if nev > 0:
            if par.verbose:
                print('Iteration {0:d} - Relocating events'.format(it+1))
                sys.stdout.flush()

            if grid.nthreads == 1 or nev < 1.5*grid.nthreads:
                for ne in range(nev):
                    _relocPS(ne, par, (grid, grid_s), evID, hyp0, data, rcv, tobs, (s_p, s_s), (indp, inds))
            else:
                # run in parallel
                blk_size = np.zeros((grid.nthreads,), dtype=np.int64)
                nj = nev
                while nj > 0:
                    for n in range(grid.nthreads):
                        blk_size[n] += 1
                        nj -= 1
                        if nj == 0:
                            break
                processes = []
                blk_start = 0
                h_queue = Queue()
                for n in range(grid.nthreads):
                    blk_end = blk_start + blk_size[n]
                    p = Process(target=_rlPS_worker,
                                args=(n, blk_start, blk_end, par, (grid, grid_s), evID, hyp0,
                                      data, rcv, tobs, (s_p, s_s), (indp, inds), h_queue),
                                daemon=True)
                    processes.append(p)
                    p.start()
                    blk_start += blk_size[n]

                for ne in range(nev):
                    h, indh = h_queue.get()
                    hyp0[indh, :] = h

    if par.invert_vel:
        if nev > 0:
            hyp = np.empty((nttp,5))
            for ne in np.arange(nev):
                indh = np.nonzero(hyp0[:,0] == evID[ne])[0]
                indrp = np.nonzero(np.logical_and(data[:,0] == evID[ne], indp))[0]
                for i in indrp:
                    hyp[i,:] = hyp0[indh[0],:]

            tcalcp = grid.raytrace(s_p, hyp, rcv_datap)

            hyp = np.empty((ntts,5))
            for ne in np.arange(nev):
                indh = np.nonzero(hyp0[:,0] == evID[ne])[0]
                indrs = np.nonzero(np.logical_and(data[:,0] == evID[ne], inds))[0]
                for i in indrs:
                    hyp[i-nttp,:] = hyp0[indh[0],:]
            tcalcs = grid_s.raytrace(s_s, hyp, rcv_datas)

            tcalc = np.hstack((tcalcp, tcalcs))
        else:
            tcalc = np.array([])

        if ncal > 0:
            tcalcp_cal = grid.raytrace(s_p, hcalp, rcv_calp)
            tcalcs_cal = grid_s.raytrace(s_s, hcals, rcv_cals)
            tcalc_cal = np.hstack((tcalcp_cal, tcalcs_cal))
        else:
            tcalc_cal = np.array([])

        r1a = tobs - tcalc
        r1 = tcal - tcalc_cal
        if r1a.size > 0:
            r1 = np.hstack((np.zeros(data.shape[0]-4*nev), r1))

        if par.show_plots:
            plt.figure(1)
            plt.cla()
            plt.plot(r1a,'o')
            plt.title('Residuals - Final step')
            plt.show(block=False)
            plt.pause(0.0001)

        r1 = np.matrix( r1.reshape(-1,1) )
        r1a = np.matrix( r1a.reshape(-1,1) )

        resV[-1] = np.linalg.norm(np.hstack((tobs-tcalc, tcal-tcalc_cal)))

    if par.verbose:
        print('\n ** Inversion complete **\n')

    return hyp0, (Vp.getA1(), Vs.getA1()), (sc_p, sc_s), (resV, resAxb)

def _rlPS_worker(thread_no, istart, iend, par, grid, evID, hyp0, data, rcv, tobs, s, ind, h_queue):
    for ne in range(istart, iend):
        h, indh = _relocPS(ne, par, grid, evID, hyp0, data, rcv, tobs, s, ind, thread_no)
        h_queue.put((h, indh))
    h_queue.close()

def _relocPS(ne, par, grid, evID, hyp0, data, rcv, tobs, s, ind, thread_no=None):

    (grid_p, grid_s) = grid
    (indp, inds) = ind
    (s_p, s_s) = s
    if par.verbose:
        print('                Updating event ID {0:d} ({1:d}/{2:d})'.format(int(1.e-6+evID[ne]), ne+1, evID.size))
        sys.stdout.flush()

    indh = np.nonzero(hyp0[:,0] == evID[ne])[0][0]
    indrp = np.nonzero(np.logical_and(data[:,0] == evID[ne], indp))[0]
    indrs = np.nonzero(np.logical_and(data[:,0] == evID[ne], inds))[0]

    hyp_save = hyp0[indh,:].copy()

    nstp = np.sum(indrp.size)
    nsts = np.sum(indrs.size)

    hypp = np.empty((nstp,5))
    stnp = np.empty((nstp,3))
    for i in range(nstp):
        hypp[i,:] = hyp0[indh,:]
        stnp[i,:] = rcv[int(1.e-6+data[indrp[i],2]),:]
    hyps = np.empty((nsts,5))
    stns = np.empty((nsts,3))
    for i in range(nsts):
        hyps[i,:] = hyp0[indh,:]
        stns[i,:] = rcv[int(1.e-6+data[indrs[i],2]),:]

    if par.hypo_2step:
        if par.verbose:
            print('                  Updating latitude & longitude', end='')
            sys.stdout.flush()
        H = np.ones((nstp+nsts,2))
        for itt in range(par.maxit_hypo):
            for i in range(nstp):
                hypp[i,:] = hyp0[indh,:]
            tcalcp, raysp, v0p = grid_p.raytrace(s_p, hypp, stnp, thread_no)
            for i in range(nsts):
                hyps[i,:] = hyp0[indh,:]
            tcalcs, rayss, v0s = grid_s.raytrace(s_s, hyps, stns, thread_no)
            for ns in range(nstp):
                raysi = raysp[ns]
                V0 = v0p[ns]

                d = (raysi[1,:]-hyp0[indh,2:]).flatten()
                ds = np.sqrt( np.sum(d*d) )
                H[ns,0] = -1./V0 * d[0]/ds
                H[ns,1] = -1./V0 * d[1]/ds
            for ns in range(nsts):
                raysi = rayss[ns]
                V0 = v0s[ns]

                d = (raysi[1,:]-hyp0[indh,2:]).flatten()
                ds = np.sqrt( np.sum(d*d) )
                H[ns+nstp,0] = -1./V0 * d[0]/ds
                H[ns+nstp,1] = -1./V0 * d[1]/ds

            r = np.hstack((tobs[indrp] - tcalcp, tobs[indrs] - tcalcs))

            x = np.linalg.lstsq(H,r,rcond=None)
            deltah = x[0]

            if np.sum( np.isfinite(deltah) ) != deltah.size:
                try:
                    U,S,VVh = np.linalg.svd(H.T.dot(H)+1e-9*np.eye(2))
                    VV = VVh.T
                    deltah = np.dot( VV, np.dot(U.T, H.T.dot(r))/S)
                except np.linalg.linalg.LinAlgError:
                    print(' - Event could not be relocated, resetting and exiting')
                    hyp0[indh,:] = hyp_save
                    return hyp_save, indh

    #        for n in range(2):
    #            if np.abs(deltah[n]) > par.dx_max:
    #                deltah[n] = par.dx_max * np.sign(deltah[n])

            new_hyp = hyp0[indh,:].copy()
            new_hyp[2:4] += deltah
            if grid_p.is_outside(new_hyp[2:5].reshape((1,3))):
                print('  Event could not be relocated inside the grid ({0:f}, {1:f}, {2:f}), resetting and exiting'.format(new_hyp[2], new_hyp[3], new_hyp[4]))
                hyp0[indh,:] = hyp_save
                return hyp_save, indh

            hyp0[indh,2:4] += deltah

            if np.sum(np.abs(deltah)<par.conv_hypo) == 2:
                if par.verbose:
                    print(' - converged at iteration '+str(itt+1))
                    sys.stdout.flush()
                break

        else:
            if par.verbose:
                print(' - reached max number of iterations')
                sys.stdout.flush()

    if par.verbose:
        print('                  Updating all hypocenter params', end='')
        sys.stdout.flush()

    H = np.ones((nstp+nsts,4))
    for itt in range(par.maxit_hypo):
        for i in range(nstp):
            hypp[i,:] = hyp0[indh,:]
        tcalcp, raysp, v0p = grid_p.raytrace(s_p, hypp, stnp, thread_no)
        for i in range(nsts):
            hyps[i,:] = hyp0[indh,:]
        tcalcs, rayss, v0s = grid_s.raytrace(s_s, hyps, stns, thread_no)
        for ns in range(nstp):
            raysi = raysp[ns]
            V0 = v0p[ns]

            d = (raysi[1,:]-hyp0[indh,2:]).flatten()
            ds = np.sqrt( np.sum(d*d) )
            H[ns,1] = -1./V0 * d[0]/ds
            H[ns,2] = -1./V0 * d[1]/ds
            H[ns,3] = -1./V0 * d[2]/ds
        for ns in range(nsts):
            raysi = rayss[ns]
            V0 = v0s[ns]

            d = (raysi[1,:]-hyp0[indh,2:]).flatten()
            ds = np.sqrt( np.sum(d*d) )
            H[ns+nstp,1] = -1./V0 * d[0]/ds
            H[ns+nstp,2] = -1./V0 * d[1]/ds
            H[ns+nstp,3] = -1./V0 * d[2]/ds

        r = np.hstack((tobs[indrp] - tcalcp, tobs[indrs] - tcalcs))
        x = np.linalg.lstsq(H,r,rcond=None)
        deltah = x[0]

        if np.sum( np.isfinite(deltah) ) != deltah.size:
            try:
                U,S,VVh = np.linalg.svd(H.T.dot(H)+1e-9*np.eye(4))
                VV = VVh.T
                deltah = np.dot( VV, np.dot(U.T, H.T.dot(r))/S)
            except np.linalg.linalg.LinAlgError:
                print('  Event could not be relocated, resetting and exiting')
                hyp0[indh,:] = hyp_save
                return hyp_save, indh

        if np.abs(deltah[0]) > par.dt_max:
            deltah[0] = par.dt_max * np.sign(deltah[0])
        for n in range(1,4):
            if np.abs(deltah[n]) > par.dx_max:
                deltah[n] = par.dx_max * np.sign(deltah[n])

        new_hyp = hyp0[indh,1:] + deltah
        if grid_p.is_outside(new_hyp[1:].reshape((1,3))):
            print('  Event could not be relocated inside the grid ({0:f}, {1:f}, {2:f}), resetting and exiting'.format(new_hyp[2], new_hyp[3], new_hyp[4]))
            hyp0[indh,:] = hyp_save
            return hyp_save, indh

        hyp0[indh,1:] += deltah

        if np.sum(np.abs(deltah[1:])<par.conv_hypo) == 3:
            if par.verbose:
                print(' - converged at iteration '+str(itt+1))
                sys.stdout.flush()
            break

    else:
        if par.verbose:
            print(' - reached max number of iterations')
            sys.stdout.flush()

    return hyp0[indh,:], indh


if __name__ == '__main__':

    xmin = 0.090
    xmax = 0.211
    ymin = 0.080
    ymax = 0.211
    zmin = 0.0
    zmax = 0.101

    dx = 0.010   # grid cell size, we use cubic cells here

    x = np.arange(xmin, xmax, dx)
    y = np.arange(ymin, ymax, dx)
    z = np.arange(zmin, zmax, dx)

    nthreads = 4
    g = Grid3D(x, y, z, nthreads)

    testK = False
    testP = False
    testPS = True
    testParallel = False
    addNoise = True

    if testK:

        Kx, Ky, Kz = g.computeK()

        V = np.ones(g.shape)
        V[5:9,5:10,3:8] = 2.

        plt.figure(figsize=(10,8))
        plt.subplot(221)
        plt.pcolor(x,z,np.squeeze(V[:,10,:].T), cmap='CMRmap',), plt.gca().invert_yaxis()
        plt.xlabel('X')
        plt.ylabel('Z')
        plt.colorbar()
        plt.subplot(222)
        plt.pcolor(y,z,np.squeeze(V[10,:,:].T), cmap='CMRmap'), plt.gca().invert_yaxis()
        plt.xlabel('Y')
        plt.ylabel('Z')
        plt.colorbar()
        plt.subplot(223)
        plt.pcolor(x,y,np.squeeze(V[:,:,6].T), cmap='CMRmap')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.colorbar()

        plt.show()


        dVx = np.reshape(Kx.dot(V.flatten()), g.shape)
        dVy = np.reshape(Ky.dot(V.flatten()), g.shape)
        dVz = np.reshape(Kz.dot(V.flatten()), g.shape)

        K = Kx + Ky + Kz
        dV = np.reshape(K.dot(V.flatten()), g.shape)


        plt.figure(figsize=(10,8))
        plt.subplot(221)
        plt.pcolor(x,z,np.squeeze(dVx[:,10,:].T), cmap='CMRmap',), plt.gca().invert_yaxis()
        plt.xlabel('X')
        plt.ylabel('Z')
        plt.colorbar()
        plt.subplot(222)
        plt.pcolor(y,z,np.squeeze(dVx[10,:,:].T), cmap='CMRmap'), plt.gca().invert_yaxis()
        plt.xlabel('Y')
        plt.ylabel('Z')
        plt.colorbar()
        plt.subplot(223)
        plt.pcolor(x,y,np.squeeze(dVx[:,:,6].T), cmap='CMRmap')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.colorbar()

        plt.show()

        plt.figure(figsize=(10,8))
        plt.subplot(221)
        plt.pcolor(x,z,np.squeeze(dVy[:,10,:].T), cmap='CMRmap',), plt.gca().invert_yaxis()
        plt.xlabel('X')
        plt.ylabel('Z')
        plt.colorbar()
        plt.subplot(222)
        plt.pcolor(y,z,np.squeeze(dVy[10,:,:].T), cmap='CMRmap'), plt.gca().invert_yaxis()
        plt.xlabel('Y')
        plt.ylabel('Z')
        plt.colorbar()
        plt.subplot(223)
        plt.pcolor(x,y,np.squeeze(dVy[:,:,6].T), cmap='CMRmap')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.colorbar()

        plt.show()

        plt.figure(figsize=(10,8))
        plt.subplot(221)
        plt.pcolor(x,z,np.squeeze(dVz[:,10,:].T), cmap='CMRmap',), plt.gca().invert_yaxis()
        plt.xlabel('X')
        plt.ylabel('Z')
        plt.colorbar()
        plt.subplot(222)
        plt.pcolor(y,z,np.squeeze(dVz[10,:,:].T), cmap='CMRmap'), plt.gca().invert_yaxis()
        plt.xlabel('Y')
        plt.ylabel('Z')
        plt.colorbar()
        plt.subplot(223)
        plt.pcolor(x,y,np.squeeze(dVz[:,:,6].T), cmap='CMRmap')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.colorbar()

        plt.show()

        plt.figure(figsize=(10,8))
        plt.subplot(221)
        plt.pcolor(x,z,np.squeeze(dV[:,10,:].T), cmap='CMRmap',), plt.gca().invert_yaxis()
        plt.xlabel('X')
        plt.ylabel('Z')
        plt.colorbar()
        plt.subplot(222)
        plt.pcolor(y,z,np.squeeze(dV[10,:,:].T), cmap='CMRmap'), plt.gca().invert_yaxis()
        plt.xlabel('Y')
        plt.ylabel('Z')
        plt.colorbar()
        plt.subplot(223)
        plt.pcolor(x,y,np.squeeze(dV[:,:,6].T), cmap='CMRmap')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.colorbar()

        plt.show()


    if testP or testPS or testParallel:

        rcv = np.array([[0.112, 0.115, 0.013],
                        [0.111, 0.116, 0.040],
                        [0.111, 0.113, 0.090],
                        [0.151, 0.117, 0.017],
                        [0.180, 0.115, 0.016],
                        [0.113, 0.145, 0.011],
                        [0.160, 0.150, 0.017],
                        [0.185, 0.149, 0.015],
                        [0.117, 0.184, 0.011],
                        [0.155, 0.192, 0.009],
                        [0.198, 0.198, 0.010],
                        [0.198, 0.196, 0.040],
                        [0.198, 0.193, 0.090]])
        ircv = np.arange(rcv.shape[0]).reshape(-1,1)
        nsta = rcv.shape[0]

        nev = 15
        src = np.vstack((np.arange(nev),
                         np.linspace(0., 50., nev) + np.random.randn(nev),
                         0.160 + 0.005*np.random.randn(nev),
                         0.140 + 0.005*np.random.randn(nev),
                         0.060 + 0.010*np.random.randn(nev))).T

        hinit = np.vstack((np.arange(nev),
                           np.linspace(0., 50., nev),
                           0.150 + 0.0001*np.random.randn(nev),
                           0.150 + 0.0001*np.random.randn(nev),
                           0.050 + 0.0001*np.random.randn(nev))).T

        h_true = src.copy()

        def Vz(z):
            return 4.0 + 10.*(z-0.050)
        def Vz2(z):
            return 4.0 + 7.5*(z-0.050)

        Vp = np.kron(Vz(z), np.ones((g.shape[0], g.shape[1], 1)))
        Vpinit = np.kron(Vz2(z), np.ones((g.shape[0], g.shape[1], 1)))
        Vs = 2.1

        slowness = 1./Vp.flatten()

        src = np.kron(src,np.ones((nsta,1)))
        rcv_data = np.kron(np.ones((nev,1)), rcv)
        ircv_data = np.kron(np.ones((nev,1)), ircv)

        tt = g.raytrace(slowness, src, rcv_data)

        Vpts = np.array([[Vz(0.001), 0.100, 0.100, 0.001, 0.0],
                         [Vz(0.001), 0.100, 0.200, 0.001, 0.0],
                         [Vz(0.001), 0.200, 0.100, 0.001, 0.0],
                         [Vz(0.001), 0.200, 0.200, 0.001, 0.0],
                         [Vz(0.011), 0.112, 0.148, 0.011, 0.0],
                         [Vz(0.005), 0.152, 0.108, 0.005, 0.0],
                         [Vz(0.075), 0.152, 0.108, 0.075, 0.0],
                         [Vz(0.011), 0.192, 0.148, 0.011, 0.0]])

        Vinit = np.mean(Vpts[:,0])
        Vpinit = Vpinit.flatten()


        ncal = 5
        src_cal = np.vstack((5+np.arange(ncal),
                         np.zeros(ncal),
                         0.160 + 0.005*np.random.randn(ncal),
                         0.130 + 0.005*np.random.randn(ncal),
                         0.045 + 0.001*np.random.randn(ncal))).T

        src_cal = np.kron(src_cal,np.ones((nsta,1)))
        rcv_cal = np.kron(np.ones((ncal,1)), rcv)
        ircv_cal = np.kron(np.ones((ncal,1)), ircv)

        ind = np.ones(rcv_cal.shape[0], dtype=bool)
        ind[3] = 0
        ind[13] = 0
        ind[15] = 0
        src_cal = src_cal[ind,:]
        rcv_cal = rcv_cal[ind,:]
        ircv_cal = ircv_cal[ind,:]

        tcal = g.raytrace(slowness, src_cal, rcv_cal)
        caldata = np.column_stack((src_cal[:,0], tcal, ircv_cal, src_cal[:,2:], np.zeros(tcal.shape)))

        Vpmin = 3.5
        Vpmax = 4.5
        PAp = 1.0
        Vsmin = 1.9
        Vsmax = 2.3
        PAs = 1.0
        Vlim = (Vpmin, Vpmax, PAp, Vsmin, Vsmax, PAs)

        dVp_max = 0.1
        dx_max = 0.01
        dt_max = 0.01
        dVs_max = 0.1
        dmax = (dVp_max, dx_max, dt_max, dVs_max)

        λ = 2.
        γ = 1.
        α = 1.
        wzK = 0.1
        lagran = (λ, γ, α, wzK)

        if addNoise:
            noise_variance = 1.e-3;  # 1 ms
        else:
            noise_variance = 0.0

        par = InvParams(maxit=3, maxit_hypo=10, conv_hypo=0.001, Vlim=Vlim, dmax=dmax,
                        lagrangians=lagran, invert_vel=True, verbose=True)

    if testParallel:

        tt = g.raytrace(slowness, src, rcv_data)
        tt, rays, v0 = g.raytrace(slowness, src, rcv_data)
        tt, rays, v0, M = g.raytrace(slowness, src, rcv_data)



        print('done')

    if testP:

        tt += noise_variance*np.random.randn(tt.size)

        data = np.hstack((src[:,0].reshape((-1,1)), tt.reshape((-1,1)), ircv_data))

        hinit2, res = hypoloc(data, rcv, Vinit, hinit, 10, 0.001, True)

#        par.invert_vel = False
#        par.maxit = 1
#        h, V, sc, res = jointHypoVel(par, g, data, rcv, Vp.flatten(), hinit2)

        h, V, sc, res = jointHypoVel(par, g, data, rcv, Vpinit, hinit2, caldata=caldata, Vpts=Vpts)

        plt.figure()
        plt.plot(res[0])
        plt.show(block=False)

        err_xc = hinit2[:,2:5] - h_true[:,2:5]
        err_xc = np.sqrt(np.sum(err_xc**2, axis=1))
        err_tc = hinit2[:,1] - h_true[:,1]

        err_x = h[:,2:5] - h_true[:,2:5]
        err_x = np.sqrt(np.sum(err_x**2, axis=1))
        err_t = h[:,1] - h_true[:,1]

        plt.figure(figsize=(10,4))
        plt.subplot(121)
        plt.plot(err_x,'o',label=r'$\|\|\Delta x\|\|$ = {0:6.5f}'.format(np.linalg.norm(err_x)))
        plt.plot(err_xc,'r*',label=r'$\|\|\Delta x\|\|$ = {0:6.5f}'.format(np.linalg.norm(err_xc)))
        plt.ylabel(r'$\Delta x$')
        plt.xlabel('Event ID')
        plt.legend()
        plt.subplot(122)
        plt.plot(np.abs(err_t),'o',label=r'$\|\|\Delta t\|\|$ = {0:6.5f}'.format(np.linalg.norm(err_t)))
        plt.plot(np.abs(err_tc),'r*',label=r'$\|\|\Delta t\|\|$ = {0:6.5f}'.format(np.linalg.norm(err_tc)))
        plt.ylabel(r'$\Delta t$')
        plt.xlabel('Event ID')
        plt.legend()

        plt.show(block=False)

        V3d = V.reshape(g.shape)

        plt.figure(figsize=(10,8))
        plt.subplot(221)
        plt.pcolor(x,z,np.squeeze(V3d[:,9,:].T), cmap='CMRmap',), plt.gca().invert_yaxis()
        plt.xlabel('X')
        plt.ylabel('Z')
        plt.colorbar()
        plt.subplot(222)
        plt.pcolor(y,z,np.squeeze(V3d[8,:,:].T), cmap='CMRmap'), plt.gca().invert_yaxis()
        plt.xlabel('Y')
        plt.ylabel('Z')
        plt.colorbar()
        plt.subplot(223)
        plt.pcolor(x,y,np.squeeze(V3d[:,:,4].T), cmap='CMRmap')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.colorbar()

        plt.show(block=False)

        plt.figure()
        plt.plot(sc,'o')
        plt.xlabel('Station no')
        plt.ylabel('Correction')
        plt.show()

    if testPS:

        Vpts_s = np.array([[Vs, 0.100, 0.100, 0.001, 1.0],
                           [Vs, 0.100, 0.200, 0.001, 1.0],
                           [Vs, 0.200, 0.100, 0.001, 1.0],
                           [Vs, 0.200, 0.200, 0.001, 1.0],
                           [Vs, 0.112, 0.148, 0.011, 1.0],
                           [Vs, 0.152, 0.108, 0.005, 1.0],
                           [Vs, 0.152, 0.108, 0.075, 1.0],
                           [Vs, 0.192, 0.148, 0.011, 1.0]])

        Vpts = np.vstack((Vpts, Vpts_s))

        slowness_s = 1./Vs + np.zeros(g.getNumberOfNodes())

        tt_s = g.raytrace(slowness_s, src, rcv_data)

        tt += noise_variance*np.random.randn(tt.size)
        tt_s += noise_variance*np.random.randn(tt_s.size)

        # remove some values
        ind_p = np.ones(tt.shape[0], dtype=bool)
        ind_p[np.random.randint(ind_p.size,size=25)] = False
        ind_s = np.ones(tt_s.shape[0], dtype=bool)
        ind_s[np.random.randint(ind_s.size,size=25)] = False

        data_p = np.hstack((src[ind_p,0].reshape((-1,1)), tt[ind_p].reshape((-1,1)), ircv_data[ind_p,:], np.zeros((np.sum(ind_p),1))))
        data_s = np.hstack((src[ind_s,0].reshape((-1,1)), tt_s[ind_s].reshape((-1,1)), ircv_data[ind_s,:], np.ones((np.sum(ind_s),1))))

        data = np.vstack((data_p, data_s))


        tcal_s = g.raytrace(slowness_s, src_cal, rcv_cal)
        caldata_s = np.column_stack((src_cal[:,0], tcal_s, ircv_cal, src_cal[:,2:], np.ones(tcal_s.shape)))
        caldata = np.vstack((caldata, caldata_s))

        Vinit = (Vinit, 2.0)

        hinit2, res = hypolocPS(data, rcv, Vinit, hinit, 10, 0.001, True)

        par.save_V = False
        par.dVs_max = 0.01
        par.invert_VsVp = False
        par.constr_sc = False
        h, V, sc, res = jointHypoVelPS(par, g, data, rcv, (Vpinit, 2.0), hinit2, caldata=caldata, Vpts=Vpts)

        plt.figure()
        plt.plot(res[0])
        plt.show(block=False)

        err_xc = hinit2[:,2:5] - h_true[:,2:5]
        err_xc = np.sqrt(np.sum(err_xc**2, axis=1))
        err_tc = hinit2[:,1] - h_true[:,1]

        err_x = h[:,2:5] - h_true[:,2:5]
        err_x = np.sqrt(np.sum(err_x**2, axis=1))
        err_t = h[:,1] - h_true[:,1]

        plt.figure(figsize=(10,4))
        plt.subplot(121)
        plt.plot(err_x,'o',label=r'$\|\|\Delta x\|\|$ = {0:6.5f}'.format(np.linalg.norm(err_x)))
        plt.plot(err_xc,'r*',label=r'$\|\|\Delta x\|\|$ = {0:6.5f}'.format(np.linalg.norm(err_xc)))
        plt.ylabel(r'$\Delta x$')
        plt.xlabel('Event ID')
        plt.legend()
        plt.subplot(122)
        plt.plot(np.abs(err_t),'o',label=r'$\|\|\Delta t\|\|$ = {0:6.5f}'.format(np.linalg.norm(err_t)))
        plt.plot(np.abs(err_tc),'r*',label=r'$\|\|\Delta t\|\|$ = {0:6.5f}'.format(np.linalg.norm(err_tc)))
        plt.ylabel(r'$\Delta t$')
        plt.xlabel('Event ID')
        plt.legend()

        plt.show(block=False)

        V3d = V[0].reshape(g.shape)

        plt.figure(figsize=(10,8))
        plt.subplot(221)
        plt.pcolor(x,z,np.squeeze(V3d[:,9,:].T), cmap='CMRmap',), plt.gca().invert_yaxis()
        plt.xlabel('X')
        plt.ylabel('Z')
        plt.colorbar()
        plt.subplot(222)
        plt.pcolor(y,z,np.squeeze(V3d[8,:,:].T), cmap='CMRmap'), plt.gca().invert_yaxis()
        plt.xlabel('Y')
        plt.ylabel('Z')
        plt.colorbar()
        plt.subplot(223)
        plt.pcolor(x,y,np.squeeze(V3d[:,:,4].T), cmap='CMRmap')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.colorbar()
        plt.suptitle('V_p')

        plt.show(block=False)

        V3d = V[1].reshape(g.shape)

        plt.figure(figsize=(10,8))
        plt.subplot(221)
        plt.pcolor(x,z,np.squeeze(V3d[:,9,:].T), cmap='CMRmap',), plt.gca().invert_yaxis()
        plt.xlabel('X')
        plt.ylabel('Z')
        plt.colorbar()
        plt.subplot(222)
        plt.pcolor(y,z,np.squeeze(V3d[8,:,:].T), cmap='CMRmap'), plt.gca().invert_yaxis()
        plt.xlabel('Y')
        plt.ylabel('Z')
        plt.colorbar()
        plt.subplot(223)
        plt.pcolor(x,y,np.squeeze(V3d[:,:,4].T), cmap='CMRmap')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.colorbar()
        plt.suptitle('V_s')

        plt.show(block=False)

        plt.figure()
        plt.plot(sc[0],'o',label='P-wave')
        plt.plot(sc[1],'r*',label='s-wave')
        plt.xlabel('Station no')
        plt.ylabel('Correction')
        plt.legend()
        plt.show()
