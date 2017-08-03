# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 10:29:32 2016

@author: giroux
"""
import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import matplotlib.pyplot  as plt

from utils import nargout
import cgrid3d

def hypoloc(data, V, hinit, maxit, convh, verbose=False):
    """
    Locate hypocenters for constant velocity model

    Parameters
    ----------
    data  : a numpy array with 5 columns
             first column is event ID number
             second column is arrival time
             third column is easting
             fourth column is northing
             fifth column is elevation
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

        x = data[ind,2:]
        t = data[ind,1]

        inh = eid==loc[:,0]
        if verbose:
            print('Locating hypocenters no '+str(int(eid)))
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

            #dh,residuals,rank,s = np.linalg.lstsq( H, r)
            dh = np.linalg.solve( np.dot(H.T, H), np.dot(H.T, r) )
            if not np.all(np.isfinite(dh)):
                try:
                    U,S,VVh = np.linalg.svd(np.dot(H.T, H)+1e-9*np.eye(4))
                    VV = VVh.T
                    dh = np.dot( VV, np.dot(U.T, np.dot(H.T, r))/S)
                except np.linalg.linalg.LinAlgError:
                    print('  Event could not be relocated (iteration no '+str(it)+'), skipping')
                    sys.stdout.flush()
                    break

            dc = np.sqrt( np.sum( dh[1:]**2 ) )
            if dc > 100.0:
                dh[1:] /= dc
                dh[0] = 0.0

            loc[inh,1:] += dh
            if dc < convh:
                if verbose:
                    print('     Converged at iteration '+str(it))
                    sys.stdout.flush()
                break
        else:
            if verbose:
                print('     Reached max number of iteration ('+str(maxit)+')')
                sys.stdout.flush()
                    
        nev += 1

    if par.verbose:
        print('\n ** Inversion complete **\n')
        
    return loc, res

def hypolocPS(data, V, hinit, maxit, convh, verbose=False):
    """
    Locate hypocenters for constant velocity model

    Parameters
    ----------
    data  : a numpy array with 6 columns
             first column is event ID number
             second column is arrival time
             third column is easting
             fourth column is northing
             fifth column is elevation
             sixth column is code for wave phase: 0 for P-wave and 1 for S-wave
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

        x = data[ind,2:5]
        t = data[ind,1]
        ph = data[ind,5]
        vel = np.zeros((len(ph),))
        for n in range(len(ph)):
            vel[n] = V[int(ph[n])]


        inh = eid==loc[:,0]
        if verbose:
            print('Locating hypocenters no '+str(int(eid)))
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
            dh = np.linalg.solve( np.dot(H.T, H), np.dot(H.T, r) )
            if not np.all(np.isfinite(dh)):
                try:
                    U,S,VVh = np.linalg.svd(np.dot(H.T, H)+1e-9*np.eye(4))
                    VV = VVh.T
                    dh = np.dot( VV, np.dot(U.T, np.dot(H.T, r))/S)
                except np.linalg.linalg.LinAlgError:
                    print('  Event could not be relocated (iteration no '+str(it)+'), skipping')
                    sys.stdout.flush()
                    break

            dc = np.sqrt( np.sum( dh[1:]**2 ) )
            if dc > 100.0:
                dh[1:] /= dc
                dh[0] = 0.0

            loc[inh,1:] += dh
            if dc < convh:
                if verbose:
                    print('     Converged at iteration '+str(it))
                    sys.stdout.flush()
                break
        else:
            if verbose:
                print('     Reached max number of iteration ('+str(maxit)+')')
                sys.stdout.flush()

        nev += 1
        
    if par.verbose:
        print('\n ** Inversion complete **\n')
        
    return loc, res

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

        if self.dx != dy or self.dx != dz:
            raise ValueError('Grid cells must be cubic')

    def getNumberOfNodes(self):
        return self.x.size * self.y.size * self.z.size
    
    @property
    def shape(self):
        return (self.x.size, self.y.size, self.z.size)

    def ind(self, i, j, k):
        return (i*self.y.size + j)*self.z.size + k

    def isOutside(self, pts):
        """
        Return True if at least one point outside grid
        """
        return ( np.min(pts[:,0]) < self.x[0] or np.max(pts[:,0]) > self.x[-1] or
                np.min(pts[:,1]) < self.y[0] or np.max(pts[:,1]) > self.y[-1] or
                np.min(pts[:,2]) < self.z[0] or np.max(pts[:,2]) > self.z[-1] )


    def raytrace(self, slowness, hypo, rcv):

        nout = nargout()

        # check input data consistency

        if hypo.ndim != 2 or rcv.ndim != 2:
            raise ValueError('hypo and rcv should be 2D arrays')

        src = hypo[:,2:5]
        if src.shape[1] != 3 or rcv.shape[1] != 3:
            raise ValueError('src and rcv should be ndata x 3')

        if src.shape != rcv.shape:
            raise ValueError('src and rcv should be of equal size')

        if self.isOutside(src):
            raise ValueError('Source point outside grid')

        if self.isOutside(rcv):
            raise ValueError('Receiver outside grid')

        if len(slowness) != self.getNumberOfNodes():
            raise ValueError('Length of slowness vector should equal number of nodes')

        t0 = hypo[:,1]

        if self.cgrid is None:
            nx = len(self.x) - 1
            ny = len(self.y) - 1
            nz = len(self.z) - 1
            
            self.cgrid = cgrid3d.Grid3Dcpp(b'node', nx, ny, nz, self.dx,
                                           self.x[0], self.y[0], self.z[0],
                                           1.e-15, 20, True, self.nthreads)

        if nout == 1:
            tt = self.cgrid.raytrace(slowness, src, rcv, t0, nout)
            return tt+t0
        elif nout == 3:
            tt, rays, v0 = self.cgrid.raytrace(slowness, src, rcv, t0, nout)
            return tt+t0, rays, v0
        elif nout == 4:
            tt, rays, v0, M = self.cgrid.raytrace(slowness, src, rcv, t0, nout)
            return tt+t0, rays, v0, M

    def computeD(self, coord):
        """
        Return matrix of interpolation weights for velocity data points constraint
        """
        if self.isOutside(coord):
            raise ValueError('Velocity data point outside grid')

        # D is npts x nnodes
        # for each point in coord, we have 8 values in D
        ivec = np.kron(np.arange(coord.shape[0], dtype=np.int64),np.ones(8, dtype=np.int64))
        jvec = np.zeros(ivec.shape, dtype=np.int64)
        vec = np.zeros(ivec.shape)

        for n in np.arange(coord.shape[0]):
            i1 = int( (coord[n,0]-self.x[0])/self.dx )
            i2 = i1+1
            j1 = int( (coord[n,1]-self.y[0])/self.dx )
            j2 = j1+1
            k1 = int( (coord[n,2]-self.z[0])/self.dx )
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

class InvParams():
    def __init__(self, maxit, maxit_hypo, conv_hypo, Vlim, dmax, lagrangians,
                 invert_vel=True, invert_VsVp=True, show_plots=True, verbose=True):
        """
        maxit       : max number of iterations
        maxit_hypo  :
        conv_hypo   : convergence criterion (units of distance)
        Vlim        : tuple holding (Vpmin, Vpmax, PAp, Vsmin, Vsmax, PAs) for
                        velocity penalties
                        PA is slope of penalty function
        dmax        : tuple holding 3 max admissible corrections, i.e.
                        dVp_max
                        dx_max
                        dt_max
                        dVs_max
        lagrangians :
                        lmbda :
                        gamma :
                        alpha :
                        wzK   : weight for vertical smoothing (w.r. to horizontal smoothing)
        invert_vel  : perform velocity inversion if True (True by default)
        invert_VsVp : find Vs/Vp ratio rather that Vs (True by default)
        show_plots  : show various plots during inversion (True by default)
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
        self.lmbda = lagrangians[0]
        self.gamma = lagrangians[1]
        self.alpha = lagrangians[2]
        self.wzK   = lagrangians[3]
        self.invert_vel = invert_vel
        self.invert_VsVp = invert_VsVp
        self.show_plots = show_plots
        self.verbose = verbose


def jointHypoVel(par, grid, data, Vinit, hinit, caldata=np.array([]), Vpts=np.array([])):
    """
    Joint hypocenter-velocity inversion on a regular grid

    Parameters
    ----------
    par     : instance of InvParams
    grid    : instance of Grid3D
    data    : a numpy array with 5 columns
               1st column is event ID number
               2nd column is arrival time
               3rd column is receiver easting
               4th column is receiver northing
               5th column is receiver elevation
    Vinit   : initial velocity model
    hinit   : initial hypocenter coordinate. The format is the same as for data
               but the second column holds the origin time and columns 3-5 hold
               the hypocenter eastings, northings & elevations
               *** important ***
               for efficiency reason when computing matrix M, initial hypocenters
               should _not_ be equal for any two event, e.g. they shoud all be
               different
    caldata : calibration shot data, numpy array with 8 columns
               1st column is cal shot ID number
               2nd column is arrival time
               3rd column is receiver easting
               4th column is receiver northing
               5th column is receiver elevation
               6rd column is source easting
               7th column is source northing
               8th column is source elevation
    Vpts    : known velocity points, numpy array with 4 columns
               1st column is velocity
               2nd column is easting
               3rd column is northing
               4th column is elevation

    Returns
    -------
    loc : hypocenter coordinates
    V   : velocity model
    """

    evID = np.unique(data[:,0])
    nev = evID.size
    sta = np.unique(data[:,2:], axis=0)
    nsta = sta.shape[0]
    sc = np.zeros(nsta)
    hyp0 = hinit.copy()
    nnodes = grid.getNumberOfNodes()

    if data.shape[0] > 0:
        tobs = data[:,1]
    else:
        tobs = np.array([])

    if caldata.shape[0] > 0:
        calID = np.unique(caldata[:,0])
        ncal = calID.size
        hcal = np.column_stack((caldata[:,0], np.zeros(caldata.shape[0]), caldata[:,5:]))
        tcal = caldata[:,1]
    else:
        ncal = 0
        tcal = np.array([])

    if np.isscalar(Vinit):
        V = np.matrixlib.mat(Vinit + np.zeros(nnodes))
        s = np.ones(nnodes)/Vinit
    else:
        V = np.matrixlib.mat(Vinit)
        s = 1./Vinit
    V = V.reshape(-1,1)

    if par.verbose:
        print('\n *** Joint hypocenter-velocity inversion ***\n')

    if par.invert_vel:
        resV = np.zeros(par.maxit+1)
        resLSQR = np.zeros(par.maxit)

        P = sp.csr_matrix(np.ones(nnodes).reshape(-1,1))
        dP = sp.csr_matrix((np.ones(nnodes), (np.arange(nnodes,dtype=np.int64),
                                     np.arange(nnodes,dtype=np.int64))), shape=(nnodes,nnodes))

        deltam = sp.csr_matrix(np.ones(nnodes+nsta).reshape(-1,1))
        deltam[:,0] = 0.0
        u1 = sp.csr_matrix(np.ones(nnodes+nsta).reshape(-1,1))
        u1[:nnodes,0] = 0.0

        if Vpts.size > 0:
            if par.verbose:
                print('Building velocity data point matrix D')
                sys.stdout.flush()
            D = grid.computeD(Vpts[:,1:])
            D1 = sp.hstack((D, sp.csr_matrix((Vpts.shape[0],nsta))))
        else:
            D = 0.0

        if par.verbose:
            print('Building regularization matrix K')
            sys.stdout.flush()
        Kx, Ky, Kz = grid.computeK()
        Kx1 = sp.hstack((Kx, sp.csr_matrix((nnodes,nsta))))
        KtKx = Kx1.T * Kx1
        Ky1 = sp.hstack((Ky, sp.csr_matrix((nnodes,nsta))))
        KtKy = Ky1.T * Ky1
        Kz1 = sp.hstack((Kz, sp.csr_matrix((nnodes,nsta))))
        KtKz = Kz1.T * Kz1
        nK = spl.norm(KtKx)
    else:
        resV = None
        resLSQR = None

    if par.verbose:
        print('\nStarting iterations')

    for it in np.arange(par.maxit):

        if par.invert_vel:
            if par.verbose:
                print('Iteration {0:d} - Updating velocity model'.format(it+1))
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
                print('                Raytracing')
                sys.stdout.flush()

            if nev > 0:
                hyp = np.empty(data.shape)
                for ne in np.arange(nev):
                    indh = np.nonzero(hyp0[:,0] == evID[ne])[0]
                    indr = np.nonzero(data[:,0] == evID[ne])[0]
                    for i in indr:
                        hyp[i,:] = hyp0[indh[0],:]
                tcalc, rays, v0, Mev = grid.raytrace(s, hyp, data[:,2:])
            else:
                tcalc = np.array([])

            if ncal > 0:
                tcalc_cal, _, _, Mcal = grid.raytrace(s, hcal, caldata[:,2:5])
            else:
                tcalc_cal = np.array([])

            r1a = tobs - tcalc
            if r1a.size > 0:
                r1 = np.hstack((np.zeros(data.shape[0]-4*nev), tcal - tcalc_cal))
            else:
                r1 = caldata[:,1] - tcalc_cal

            if par.show_plots:
                plt.figure(1)
                plt.plot(r1a,'o')
                plt.title('Residuals - Iteration {0:d}'.format(it+1))
                plt.show(block=False)

            r1 = np.matrixlib.mat( r1.reshape(-1,1) )
            r1a = np.matrixlib.mat( r1a.reshape(-1,1) )

            resV[it] = np.linalg.norm(np.hstack((tobs-tcalc, tcal-tcalc_cal)))

            # initializing matrix M; matrix of partial derivatives of velocity dt/dV
            if par.verbose:
                print('                Building matrix M')
                sys.stdout.flush()

            M1 = None
            ir1 = 0
            for ne in range(nev):
                if par.verbose:
                    print('                  Event ID '+str(int(evID[ne])))
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
                M = sp.csr_matrix(Mev[ne], shape=(nst,nnodes+nsta))
                M = T * M

                if M1 == None:
                    M1 = M
                else:
                    M1 = sp.vstack((M1, M))

                r1[ir1+np.arange(nst2, dtype=np.int64)] = T.dot(r1a[indr])
                ir1 += nst2


            for nc in range(ncal):
                M = sp.csr_matrix(Mcal[nc], shape=(Mcal[nc][2].size-1, nnodes+nsta))
                if M1 == None:
                    M1 = M
                else:
                    M1 = sp.vstack((M1, M))

            if par.verbose:
                print('                Assembling matrices and solving system')
                sys.stdout.flush()

            s = -u1.T * deltam

            dP1 = sp.hstack((dP, sp.csr_matrix(np.zeros((nnodes,nsta)))))  # dP prime

            # compute A & h for inversion

            tmp1 = M1.T * M1
            nM = spl.norm(tmp1)
            tmp3 = dP1.T * dP1
            nP = spl.norm(tmp3)
            tmp4 = u1 * u1.T

            lmbda = par.lmbda * nM / nK
            if nP != 0.0:
                gamma = par.gamma * nM / nP
            else:
                gamma = par.gamma

            A = tmp1 + lmbda*KtKx + lmbda*KtKy + par.wzK*lmbda*KtKz + gamma*tmp3 + tmp4

            tmp1 = M1.T * r1
            tmp2x = Kx1.T * cx
            tmp2y = Ky1.T * cy
            tmp2z = Kz1.T * cz
            tmp3 = dP1.T * P
            tmp4 = u1 * s
            b = tmp1 - lmbda*tmp2x - lmbda*tmp2y - par.wzK*lmbda*tmp2z - gamma*tmp3 - tmp4

            if Vpts.shape[0] > 0:
                tmp5 = D1.T * D1
                nD = spl.norm(tmp5)
                alpha = par.alpha * nM / nD
                A += alpha * tmp5
                b += alpha * D1.T * (Vpts[:,0].reshape(-1,1) - D*V )

            x = spl.lsqr(A, b.getA1())

            deltam = np.matrixlib.mat(x[0].reshape(-1,1))
            resLSQR[it] = x[3]

            ind = np.nonzero( np.abs(deltam[:nnodes]) > par.dVp_max )[0]
            for i in ind:
                deltam[i] = par.dVp_max * np.sign(deltam[i])

            V += np.matrixlib.mat(deltam[:nnodes].reshape(-1,1))
            s = 1. / V.getA1()
            sc += deltam[nnodes:,0].getA1()


        if nev > 0:
            if par.verbose:
                print('Iteration {0:d} - Relocating events'.format(it+1))
                sys.stdout.flush()

            for ne in range(nev):  # TODO: this loop in parallel
                _reloc(ne, par, grid, evID, hyp0, data, tobs, s)

    if par.invert_vel:
        if nev > 0:
            hyp = np.empty(data.shape)
            for ne in np.arange(nev):
                indh = np.nonzero(hyp0[:,0] == evID[ne])[0]
                indr = np.nonzero(data[:,0] == evID[ne])[0]
                for i in indr:
                    hyp[i,:] = hyp0[indh[0],:]
            tcalc = grid.raytrace(s, hyp, data[:,2:])
        else:
            tcalc = np.array([])

        if ncal > 0:
            tcalc_cal = grid.raytrace(s, hcal, caldata[:,2:5])
        else:
            tcalc_cal = np.array([])

        r1a = tobs - tcalc
        if r1a.size > 0:
            r1 = np.hstack((np.zeros(data.shape[0]-4*nev), tcal - tcalc_cal))
        else:
            r1 = caldata[:,1] - tcalc_cal

        if par.show_plots:
            plt.figure(1)
            plt.plot(r1a,'o')
            plt.title('Residuals - Final step')
            plt.show(block=False)

        r1 = np.matrixlib.mat( r1.reshape(-1,1) )
        r1a = np.matrixlib.mat( r1a.reshape(-1,1) )

        resV[-1] = np.linalg.norm(np.hstack((tobs-tcalc, tcal-tcalc_cal)))

    if par.verbose:
        print('\n ** Inversion complete **\n')
        
    return hyp0, V.getA1(), sc, (resV, resLSQR)

def _reloc(ne, par, grid, evID, hyp0, data, tobs, s):

    if par.verbose:
        print('                Updating event ID {0:d} ({1:d}/{2:d})'.format(int(evID[ne]), ne+1, evID.size))
        print('                  Updating latitude & longitude', end='')
        sys.stdout.flush()

    indh = np.nonzero(hyp0[:,0] == evID[ne])[0]
    indr = np.nonzero(data[:,0] == evID[ne])[0]

    nst = np.sum(indr.size)

    hyp = np.empty((nst,5))
    stn = np.empty((nst,3))
    for i in range(nst):
        hyp[i,:] = hyp0[indh[0],:]
        stn[i,:] = data[indr[i],2:]

    H = np.ones((nst,2))
    for itt in range(par.maxit_hypo):
        for i in range(nst):
            hyp[i,:] = hyp0[indh[0],:]
        tcalc, rays, v0 = grid.raytrace(s, hyp, stn)
        for ns in range(nst):
            raysi = rays[ns]
            V0 = v0[ns]

            d = (raysi[1,:]-hyp0[indh,2:]).flatten()
            ds = np.sqrt( np.sum(d*d) )
            H[ns,0] = -1./V0 * d[0]/ds
            H[ns,1] = -1./V0 * d[1]/ds

        r = tobs[indr] - tcalc
        x = np.linalg.lstsq(H,r)
        deltah = x[0]

        if np.sum( np.isfinite(deltah) ) != deltah.size:
            try:
                U,S,VVh = np.linalg.svd(np.dot(H.T, H)+1e-9*np.eye(2))
                VV = VVh.T
                deltah = np.dot( VV, np.dot(U.T, np.dot(H.T, r))/S)
            except np.linalg.linalg.LinAlgError:
                print(' - Event could not be relocated, exiting')
                return

        for n in range(2):
            if np.abs(deltah[n]) > par.dx_max:
                deltah[n] = par.dx_max * np.sign(deltah[n])

        new_hyp = hyp0[indh[0],:].copy()
        new_hyp[2:4] += deltah
        if grid.isOutside(new_hyp[2:].reshape((1,3))):
            print('  Event could not be relocated inside the grid, exiting')
            return

        hyp0[indh[0],2:4] += deltah

        if itt == 0:
            phyp = hyp0[indh,2:4].copy()
        else:
            if np.sum(np.abs(phyp-hyp0[indh,2:4])<par.conv_hypo) == 2:
                if par.verbose:
                    print(' - converged at iteration '+str(itt+1))
                    sys.stdout.flush()
                break
            phyp = hyp0[indh,2:4].copy()

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
            hyp[i,:] = hyp0[indh[0],:]
        tcalc, rays, v0 = grid.raytrace(s, hyp, stn)
        for ns in range(nst):
            raysi = rays[ns]
            V0 = v0[ns]

            d = (raysi[1,:]-hyp0[indh,2:]).flatten()
            ds = np.sqrt( np.sum(d*d) )
            H[ns,1] = -1./V0 * d[0]/ds
            H[ns,2] = -1./V0 * d[1]/ds
            H[ns,3] = -1./V0 * d[2]/ds

        r = tobs[indr] - tcalc
        x = np.linalg.lstsq(H,r)
        deltah = x[0]

        if np.sum( np.isfinite(deltah) ) != deltah.size:
            try:
                U,S,VVh = np.linalg.svd(np.dot(H.T, H)+1e-9*np.eye(4))
                VV = VVh.T
                deltah = np.dot( VV, np.dot(U.T, np.dot(H.T, r))/S)
            except np.linalg.linalg.LinAlgError:
                print('  Event could not be relocated, exiting')
                return

        if np.abs(deltah[0]) > par.dt_max:
            deltah[0] = par.dt_max * np.sign(deltah[0])
        for n in range(1,4):
            if np.abs(deltah[n]) > par.dx_max:
                deltah[n] = par.dx_max * np.sign(deltah[n])

        new_hyp = hyp0[indh[0],1:] + deltah
        if grid.isOutside(new_hyp[1:].reshape((1,3))):
            print('  Event could not be relocated inside the grid, exiting')
            return
        
        hyp0[indh[0],1:] += deltah

        if itt == 0:
            phyp = hyp0[indh,2:].copy()
        else:
            if np.sum(np.abs(phyp-hyp0[indh,2:])<par.conv_hypo) == 3:
                if par.verbose:
                    print(' - converged at iteration '+str(itt+1))
                    sys.stdout.flush()
                break
            phyp = hyp0[indh,2:].copy()

    else:
        if par.verbose:
            print(' - reached max number of iterations')
            sys.stdout.flush()













def jointHypoVelPS(par, grid, data, Vinit, hinit, caldata=np.array([]), Vpts=np.array([])):
    """
    Joint hypocenter-velocity inversion on a regular grid

    Parameters
    ----------
    par     : instance of InvParams
    grid    : instance of Grid3D
    data    : a numpy array with 5 columns
               1st column is event ID number
               2nd column is arrival time
               3rd column is receiver easting
               4th column is receiver northing
               5th column is receiver elevation
               6th column is code for wave phase: 0 for P-wave and 1 for S-wave
    Vinit   : tuple with initial velocity model (P-wave first, S-wave second)
    hinit   : initial hypocenter coordinate. The format is the same as for data
               but the second column holds the origin time and columns 3-5 hold
               the hypocenter eastings, northings & elevations
               *** important ***
               for efficiency reason when computing matrix M, initial hypocenters
               should _not_ be equal for any two event, e.g. they shoud all be
               different
    caldata : calibration shot data, numpy array with 8 columns
               1st column is cal shot ID number
               2nd column is arrival time
               3rd column is receiver easting
               4th column is receiver northing
               5th column is receiver elevation
               6rd column is source easting
               7th column is source northing
               8th column is source elevation
               9th column is code for wave phase: 0 for P-wave and 1 for S-wave
    Vpts    : known velocity points, numpy array with 4 columns
               1st column is velocity
               2nd column is easting
               3rd column is northing
               4th column is elevation
               5th column is code for wave phase: 0 for P-wave and 1 for S-wave

    Returns
    -------
    loc : hypocenter coordinates
    V   : velocity model
    """

    evID = np.unique(data[:,0])
    nev = evID.size
    sta = np.unique(data[:,2:5], axis=0)
    nsta = sta.shape[0]
    sc_p = np.zeros(nsta)
    sc_s = np.zeros(nsta)
    hyp0 = hinit.copy()
    nnodes = grid.getNumberOfNodes()
    
    # sort data by seismic phase (P-wave first S-wave second)
    indp = data[:,5] == 0.0
    inds = data[:,5] == 1.0
    nttp = np.sum( indp )
    ntts = np.sum( inds )
    datap = data[indp,:]
    datas = data[inds,:]
    data = np.vstack((datap, datas))
    

    if data.shape[0] > 0:
        tobs = data[:,1]
    else:
        tobs = np.array([])

    if caldata.shape[0] > 0:
        # TODO : allow S-wave data
        calID = np.unique(caldata[:,0])
        ncal = calID.size
        hcal = np.column_stack((caldata[:,0], np.zeros(caldata.shape[0]), caldata[:,5:8]))
        tcal = caldata[:,1]
    else:
        ncal = 0
        tcal = np.array([])

    if np.isscalar(Vinit[0]):
        Vp = np.matrixlib.mat(Vinit[0] + np.zeros(nnodes))
        s_p = np.ones(nnodes)/Vinit[0]
    else:
        Vp = np.matrixlib.mat(Vinit[0])
        s_p = 1./Vinit[0]
    Vp = Vp.reshape(-1,1)
    if np.isscalar(Vinit[1]):
        Vs = np.matrixlib.mat(Vinit[1] + np.zeros(nnodes))
        s_s = np.ones(nnodes)/Vinit[1]
    else:
        Vs = np.matrixlib.mat(Vinit[1])
        s_s = 1./Vinit[1]
    Vs = Vs.reshape(-1,1)
    V = np.vstack((Vp, Vs))

    if par.verbose:
        print('\n *** Joint hypocenter-velocity inversion  -- P and S-wave data ***\n')

    if par.invert_vel:
        resV = np.zeros(par.maxit+1)
        resLSQR = np.zeros(par.maxit)

        P = sp.csr_matrix(np.ones(2*nnodes).reshape(-1,1))
        dP = sp.csr_matrix((np.ones(2*nnodes), (np.arange(2*nnodes,dtype=np.int64),
                            np.arange(2*nnodes,dtype=np.int64))),
                            shape=(2*nnodes,2*nnodes))

        deltam = sp.csr_matrix(np.ones(2*nnodes+2*nsta).reshape(-1,1))
        deltam[:,0] = 0.0
        u1 = sp.csr_matrix(np.ones(2*nnodes+2*nsta).reshape(-1,1))
        u1[:2*nnodes,0] = 0.0

        if Vpts.size > 0:

            # make sure that Vpts does not contain Vs values
            if par.invert_VsVp and np.sum(Vpts[:,4]==1) > 0:
                raise ValueError('Velocity data points cannot contain S-wave values when inverting for Vs/Vp')

            if par.verbose:
                print('Building velocity data point matrix D')
                sys.stdout.flush()
            if par.invert_VsVp:
                D = grid.computeD(Vpts[:,1:4])
                D = sp.hstack((D, sp.csr_matrix(D.shape)))
            else:
                i_p = Vpts[:,4]==0
                i_s = Vpts[:,4]==1
                Dp = grid.computeD(Vpts[i_p,1:4])
                Ds = grid.computeD(Vpts[i_s,1:4])
                D = sp.block_diag((Dp, Ds))
                print(Dp.shape, Ds.shape, D.shape)

            D1 = sp.hstack((D, sp.csr_matrix((Vpts.shape[0],2*nsta))))
        else:
            D = 0.0

        if par.verbose:
            print('Building regularization matrix K')
            sys.stdout.flush()
        Kx, Ky, Kz = grid.computeK()
        Kx = sp.block_diag((Kx, Kx))
        Ky = sp.block_diag((Ky, Ky))
        Kz = sp.block_diag((Kz, Kz))
        Kx1 = sp.hstack((Kx, sp.csr_matrix((2*nnodes,2*nsta))))
        KtKx = Kx1.T * Kx1
        Ky1 = sp.hstack((Ky, sp.csr_matrix((2*nnodes,2*nsta))))
        KtKy = Ky1.T * Ky1
        Kz1 = sp.hstack((Kz, sp.csr_matrix((2*nnodes,2*nsta))))
        KtKz = Kz1.T * Kz1
        nK = spl.norm(KtKx)
    else:
        resV = None
        resLSQR = None

    if par.verbose:
        print('\nStarting iterations')

    for it in np.arange(par.maxit):

        if par.invert_vel:
            if par.verbose:
                print('Iteration {0:d} - Updating velocity model'.format(it+1))
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
                if V[n,0] < par.Vsmin:
                    P[n,0] = par.PAs * (par.Vsmin-V[n,0])
                    dP[n,n] = -par.PAs
                elif V[n,0] > par.Vsmax:
                    P[n,0] = par.PAs * (V[n,0]-par.Vsmax)
                    dP[n,n] = par.PAs
                else:
                    P[n,0] = 0.0
                    dP[n,n] = 0.0

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
                        
                tcalcp, raysp, v0p, Mevp = grid.raytrace(s_p, hyp, data[indp,2:5])

                hyp = np.empty((ntts,5))
                for ne in np.arange(nev):
                    indh = np.nonzero(hyp0[:,0] == evID[ne])[0]
                    indrs = np.nonzero(np.logical_and(data[:,0] == evID[ne], inds))[0]
                    for i in indrs:
                        hyp[i-nttp,:] = hyp0[indh[0],:]
                tcalcs, rayss, v0s, Mevs = grid.raytrace(s_s, hyp, data[inds,2:5])

                ne = 0
                indrp = np.nonzero(np.logical_and(data[:,0] == evID[ne], data[:,5] == 0.0))[0]
                indrs = np.nonzero(np.logical_and(data[:,0] == evID[ne], data[:,5] == 1.0))[0]
                for ne in np.arange(1, nev):
                    indrp = np.hstack((indrp, np.nonzero(np.logical_and(data[:,0] == evID[ne], data[:,5] == 0.0))[0]))
                    indrs = np.hstack((indrs, np.nonzero(np.logical_and(data[:,0] == evID[ne], data[:,5] == 1.0))[0]))

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
                    
                    Mp = sp.csr_matrix(Mevp[ne], shape=(Mevp[ne][2].size-1,nnodes+nsta))
                    Ms = sp.csr_matrix(Mevs[ne], shape=(Mevs[ne][2].size-1,nnodes+nsta))

                    # extract terms for station corrections                    
                    Mpsc = Mp[:,nnodes:]
                    Mssc = Ms[:,nnodes:]
                    Msc = sp.block_diag((Mpsc, Mssc))
                    # keep only terms for velocity
                    Mp = Mp[:,:nnodes]
                    Ms = Ms[:,:nnodes]
                    
                    if par.invert_VsVp:
                        # Block 1991, p. 45
                        tmp1 = Ms.multiply(np.matlib.repmat(Vp.T, Ms.shape[0], 1))
                        VsVp = Vs.A / Vp.A
                        tmp2 = Ms.multiply(np.matlib.repmat(VsVp.T, Ms.shape[0], 1))
                        tmp2 = sp.hstack((tmp1, tmp2))
                        tmp1 = sp.hstack((Mp, sp.csr_matrix(Mp.shape)))
                        Mev[ne] = sp.vstack((tmp1, tmp2))
                    else:
                        Mev[ne] = sp.block_diag((Mevp[ne], Mevs[ne]))
                    # add terms for station corrections after terms for velocity because
                    # solution vector contains [Vp Vs sc_p sc_s] in that order
                    Mev[ne] = sp.hstack((Mev[ne], Msc))

            else:
                tcalc = np.array([])

            if ncal > 0:
                tcalc_cal, _, _, Mcal = grid.raytrace(s_p, hcal, caldata[:,2:5])
            else:
                tcalc_cal = np.array([])

            r1a = tobs - tcalc
            if r1a.size > 0:
                r1 = np.hstack((np.zeros(data.shape[0]-4*nev), tcal - tcalc_cal))
            else:
                r1 = caldata[:,1] - tcalc_cal

            r1 = np.matrixlib.mat( r1.reshape(-1,1) )
            r1a = np.matrixlib.mat( r1a.reshape(-1,1) )

            resV[it] = np.linalg.norm(np.hstack((tobs-tcalc, tcal-tcalc_cal)))

            if par.show_plots:
                plt.figure(1)
                plt.plot(r1a,'o')
                plt.title('Residuals - Iteration {0:d}'.format(it+1))
                plt.show(block=False)

            # initializing matrix M; matrix of partial derivatives of velocity dt/dV
            if par.verbose:
                print('                Building matrix M')
                sys.stdout.flush()

            M1 = None
            ir1 = 0
            for ne in range(nev):
                if par.verbose:
                    print('                  Event ID '+str(int(evID[ne])))
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
                M = sp.csr_matrix(Mcal[nc], shape=(Mcal[nc][2].size-1, 2*nnodes+2*nsta))
                if M1 == None:
                    M1 = M
                else:
                    M1 = sp.vstack((M1, M))

            if par.verbose:
                print('                Assembling matrices and solving system')
                sys.stdout.flush()

            s = -u1.T * deltam

            dP1 = sp.hstack((dP, sp.csr_matrix((2*nnodes,2*nsta))))  # dP prime

            # compute A & h for inversion

            tmp1 = M1.T * M1
            nM = spl.norm(tmp1)
            tmp3 = dP1.T * dP1
            nP = spl.norm(tmp3)
            tmp4 = u1 * u1.T

            lmbda = par.lmbda * nM / nK
            if nP != 0.0:
                gamma = par.gamma * nM / nP
            else:
                gamma = par.gamma

            A = tmp1 + lmbda*KtKx + lmbda*KtKy + par.wzK*lmbda*KtKz + gamma*tmp3 + tmp4

            tmp1 = M1.T * r1
            tmp2x = Kx1.T * cx
            tmp2y = Ky1.T * cy
            tmp2z = Kz1.T * cz
            tmp3 = dP1.T * P
            tmp4 = u1 * s
            b = tmp1 - lmbda*tmp2x - lmbda*tmp2y - par.wzK*lmbda*tmp2z - gamma*tmp3 - tmp4

            if Vpts.shape[0] > 0:
                tmp5 = D1.T * D1
                nD = spl.norm(tmp5)
                alpha = par.alpha * nM / nD
                A += alpha * tmp5
                b += alpha * D1.T * (Vpts[:,0].reshape(-1,1) - D*V )

            x = spl.lsqr(A, b.getA1())

            deltam = np.matrixlib.mat(x[0].reshape(-1,1))
            resLSQR[it] = x[3]

            ind = np.nonzero( np.abs(deltam[:nnodes]) > par.dVp_max )[0]
            for i in ind:
                deltam[i] = par.dVp_max * np.sign(deltam[i])
            ind = np.nonzero( np.abs(deltam[nnodes:2*nnodes]) > par.dVs_max )[0]
            for i in ind:
                deltam[nnodes+i] = par.dVs_max * np.sign(deltam[nnodes+i])
            

            V += np.matrixlib.mat(deltam[:2*nnodes].reshape(-1,1))
            s = 1. / V.getA1()
            s_p = s[:nnodes]
            s_s = s[nnodes:]
            sc_p += deltam[2*nnodes:2*nnodes+nsta,0].getA1()
            sc_s += deltam[2*nnodes+nsta:,0].getA1()


        if nev > 0:
            if par.verbose:
                print('Iteration {0:d} - Relocating events'.format(it+1))
                sys.stdout.flush()

            for ne in range(nev):  # TODO: this loop in parallel
                _relocPS(ne, par, grid, evID, hyp0, data, tobs, (s_p, s_s), (indp, inds))

    if par.invert_vel:
        if nev > 0:
            hyp = np.empty((nttp,5))
            for ne in np.arange(nev):
                indh = np.nonzero(hyp0[:,0] == evID[ne])[0]
                indrp = np.nonzero(np.logical_and(data[:,0] == evID[ne], indp))[0]
                for i in indrp:
                    hyp[i,:] = hyp0[indh[0],:]
                    
            tcalcp = grid.raytrace(s_p, hyp, data[indp,2:5])
            
            hyp = np.empty((ntts,5))
            for ne in np.arange(nev):
                indh = np.nonzero(hyp0[:,0] == evID[ne])[0]
                indrs = np.nonzero(np.logical_and(data[:,0] == evID[ne], inds))[0]
                for i in indrs:
                    hyp[i-nttp,:] = hyp0[indh[0],:]
            tcalcs = grid.raytrace(s_s, hyp, data[inds,2:5])

            tcalc = np.hstack((tcalcp, tcalcs))
        else:
            tcalc = np.array([])

        if ncal > 0:
            tcalc_cal = grid.raytrace(s_p, hcal, caldata[:,2:5])
        else:
            tcalc_cal = np.array([])

        r1a = tobs - tcalc
        if r1a.size > 0:
            r1 = np.hstack((np.zeros(data.shape[0]-4*nev), tcal - tcalc_cal))
        else:
            r1 = caldata[:,1] - tcalc_cal

        if par.show_plots:
            plt.figure(1)
            plt.plot(r1a,'o')
            plt.title('Residuals - Final step')
            plt.show(block=False)

        r1 = np.matrixlib.mat( r1.reshape(-1,1) )
        r1a = np.matrixlib.mat( r1a.reshape(-1,1) )

        resV[-1] = np.linalg.norm(np.hstack((tobs-tcalc, tcal-tcalc_cal)))

    if par.verbose:
        print('\n ** Inversion complete **\n')
        
    return hyp0, V.getA1(), (sc_p, sc_s), (resV, resLSQR)

def _relocPS(ne, par, grid, evID, hyp0, data, tobs, s, ind):

    (indp, inds) = ind
    (s_p, s_s) = s
    if par.verbose:
        print('                Updating event ID {0:d} ({1:d}/{2:d})'.format(int(evID[ne]), ne+1, evID.size))
        print('                  Updating latitude & longitude', end='')
        sys.stdout.flush()

    indh = np.nonzero(hyp0[:,0] == evID[ne])[0]
    indrp = np.nonzero(np.logical_and(data[:,0] == evID[ne], indp))[0]
    indrs = np.nonzero(np.logical_and(data[:,0] == evID[ne], inds))[0]
    
    nstp = np.sum(indrp.size)
    nsts = np.sum(indrs.size)

    hypp = np.empty((nstp,5))
    stnp = np.empty((nstp,3))
    for i in range(nstp):
        hypp[i,:] = hyp0[indh[0],:]
        stnp[i,:] = data[indrp[i],2:5]
    hyps = np.empty((nsts,5))
    stns = np.empty((nsts,3))
    for i in range(nsts):
        hyps[i,:] = hyp0[indh[0],:]
        stns[i,:] = data[indrs[i],2:5]

    H = np.ones((nstp+nsts,2))
    for itt in range(par.maxit_hypo):
        for i in range(nstp):
            hypp[i,:] = hyp0[indh[0],:]
        tcalcp, raysp, v0p = grid.raytrace(s_p, hypp, stnp)
        for i in range(nsts):
            hyps[i,:] = hyp0[indh[0],:]
        tcalcs, rayss, v0s = grid.raytrace(s_s, hyps, stns)
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
        
        x = np.linalg.lstsq(H,r)
        deltah = x[0]

        if np.sum( np.isfinite(deltah) ) != deltah.size:
            try:
                U,S,VVh = np.linalg.svd(np.dot(H.T, H)+1e-9*np.eye(2))
                VV = VVh.T
                deltah = np.dot( VV, np.dot(U.T, np.dot(H.T, r))/S)
            except np.linalg.linalg.LinAlgError:
                print(' - Event could not be relocated, exiting')
                return

        for n in range(2):
            if np.abs(deltah[n]) > par.dx_max:
                deltah[n] = par.dx_max * np.sign(deltah[n])

        new_hyp = hyp0[indh[0],:].copy()
        new_hyp[2:4] += deltah
        if grid.isOutside(new_hyp[2:5].reshape((1,3))):
            print('  Event could not be relocated inside the grid, exiting')
            return

        hyp0[indh[0],2:4] += deltah

        if itt == 0:
            phyp = hyp0[indh,2:4].copy()
        else:
            if np.sum(np.abs(phyp-hyp0[indh,2:4])<par.conv_hypo) == 2:
                if par.verbose:
                    print(' - converged at iteration '+str(itt+1))
                    sys.stdout.flush()
                break
            phyp = hyp0[indh,2:4].copy()

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
            hypp[i,:] = hyp0[indh[0],:]
        tcalcp, raysp, v0p = grid.raytrace(s_p, hypp, stnp)
        for i in range(nsts):
            hyps[i,:] = hyp0[indh[0],:]
        tcalcs, rayss, v0s = grid.raytrace(s_s, hyps, stns)
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
            H[ns,1] = -1./V0 * d[0]/ds
            H[ns,2] = -1./V0 * d[1]/ds
            H[ns,3] = -1./V0 * d[2]/ds

        r = np.hstack((tobs[indrp] - tcalcp, tobs[indrs] - tcalcs))
        x = np.linalg.lstsq(H,r)
        deltah = x[0]

        if np.sum( np.isfinite(deltah) ) != deltah.size:
            try:
                U,S,VVh = np.linalg.svd(np.dot(H.T, H)+1e-9*np.eye(4))
                VV = VVh.T
                deltah = np.dot( VV, np.dot(U.T, np.dot(H.T, r))/S)
            except np.linalg.linalg.LinAlgError:
                print('  Event could not be relocated, exiting')
                return

        if np.abs(deltah[0]) > par.dt_max:
            deltah[0] = par.dt_max * np.sign(deltah[0])
        for n in range(1,4):
            if np.abs(deltah[n]) > par.dx_max:
                deltah[n] = par.dx_max * np.sign(deltah[n])

        new_hyp = hyp0[indh[0],1:] + deltah
        if grid.isOutside(new_hyp[1:].reshape((1,3))):
            print('  Event could not be relocated inside the grid, exiting')
            return
        
        hyp0[indh[0],1:] += deltah

        if itt == 0:
            phyp = hyp0[indh,2:].copy()
        else:
            if np.sum(np.abs(phyp-hyp0[indh,2:])<par.conv_hypo) == 3:
                if par.verbose:
                    print(' - converged at iteration '+str(itt+1))
                    sys.stdout.flush()
                break
            phyp = hyp0[indh,2:].copy()

    else:
        if par.verbose:
            print(' - reached max number of iterations')
            sys.stdout.flush()






if __name__ == '__main__':

    g = Grid3D(np.arange(50., 251., 10.),np.arange(70., 271., 10.),np.arange(-20., 111., 10.))
    slowness = 1./4000.0 + np.zeros(g.getNumberOfNodes())
    
    rcv = np.array([[112., 115., 13.],
                    [151., 117., 17.],
                    [180., 115., 16.],
                    [113., 145., 11.],
                    [160., 150., 17.],
                    [185., 149., 15.],
                    [117., 184., 11.],
                    [155., 192.,  9.],
                    [188., 188., 10.],
                    [188., 186., 30.],
                    [188., 183., 60.]])
    nsta = rcv.shape[0]

    nev = 15
    src = np.vstack((np.arange(nev),
                     np.linspace(0., 50., nev) + np.random.randn(nev),
                     130. +  5.*np.random.randn(nev),
                     160. +  5.*np.random.randn(nev),
                      35. + 10.*np.random.randn(nev))).T

    hinit = np.vstack((np.arange(nev),
                       np.linspace(0., 50., nev),
                       130. + 0.1*np.random.randn(nev),
                       160. + 0.1*np.random.randn(nev),
                        35. + 0.1*np.random.randn(nev))).T

    h_true = src.copy()

    src = np.kron(src,np.ones((nsta,1)))
    rcv = np.kron(np.ones((nev,1)), rcv)

    tt = g.raytrace(slowness, src, rcv)

    Vpts = np.array([[4000.0, 110.0, 110.0, 10.0, 0],
                     [4000.0, 112.0, 148.0, 11.0, 0]])


    ncal = 3
    cal = np.vstack((5+np.arange(ncal),
                     np.zeros(ncal),
                     160. +  5.*np.random.randn(ncal),
                     130. +  5.*np.random.randn(ncal),
                       5. +     np.random.randn(ncal))).T

    cal = np.kron(cal,np.ones((nsta,1)))
    rcv_cal = np.kron(np.ones((ncal,1)), rcv[:nsta,:])

    ind = np.ones(rcv_cal.shape[0], dtype=bool)
    ind[3] = 0
    ind[13] = 0
    ind[15] = 0
    cal = cal[ind,:]
    rcv_cal = rcv_cal[ind,:]

    tcal = g.raytrace(slowness, cal, rcv_cal)
    caldata = np.column_stack((cal[:,0], tcal, rcv_cal, cal[:,2:], np.zeros(tcal.shape)))

    Vlim = (3500., 4500., 1.0, 1500., 2500., 1.0)
    dmax = (50., 5., 2.e-3, 25.)
    lagran = (1., 1., 1., 1.)

    noise_variance = 1.e-3;  # 1 ms
        
    par = InvParams(maxit=2, maxit_hypo=10, conv_hypo=2, Vlim=Vlim, dmax=dmax,
                    lagrangians=lagran, invert_vel=True, verbose=True)

    testP = True
    testPS = False
    
    if testP:
    
        tt += noise_variance*np.random.randn(tt.size)
    
        data = np.hstack((src[:,0].reshape((-1,1)), tt.reshape((-1,1)), rcv))
    
        Vinit = 3900.0
        
        hinit2, res = hypoloc(data, Vinit, hinit, 15, 1., True)
        
        h, V, sc, res = jointHypoVel(par, g, data, Vinit, hinit2, caldata=caldata, Vpts=Vpts)

    if testPS:
        
        slowness_s = 1./2200.0 + np.zeros(g.getNumberOfNodes())
        
        tt_s = g.raytrace(slowness_s, src, rcv)
        
        tt += noise_variance*np.random.randn(tt.size)
        tt_s += noise_variance*np.random.randn(tt_s.size)
        
        # remove some values
        ind_p = np.ones(tt.shape[0], dtype=bool)
        ind_p[np.random.randint(ind_p.size,size=25)] = False
        ind_s = np.ones(tt_s.shape[0], dtype=bool)
        ind_s[np.random.randint(ind_s.size,size=25)] = False
        
        data_p = np.hstack((src[ind_p,0].reshape((-1,1)), tt[ind_p].reshape((-1,1)), rcv[ind_p,:], np.zeros((np.sum(ind_p),1))))
        data_s = np.hstack((src[ind_s,0].reshape((-1,1)), tt_s[ind_s].reshape((-1,1)), rcv[ind_s,:], np.ones((np.sum(ind_s),1))))
        
        data = np.vstack((data_p, data_s))
        
        Vinit = (3900.0, 2100.0)
        
        hinit2, res = hypolocPS(data, Vinit, hinit, 15, 1., True)
        
        h, V, sc, res = jointHypoVelPS(par, g, data, Vinit, hinit2, caldata=caldata, Vpts=Vpts)