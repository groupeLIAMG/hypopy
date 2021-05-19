# -*- coding: utf-8 -*-
"""
HYPOcenter location from arrival time data in PYthon.

There are currently 4 hypocenter location functions in this module

hypoloc : Locate hypocenters for constant velocity model
hypolocPS : Locate hypocenters from P- and S-wave arrival time data for
            constant velocity models
jointHypoVel : Joint hypocenter-velocity inversion on a regular grid (cubic
               cells)
jointHypoVelPS : Joint hypocenter-velocity inversion of P- and S-wave arrival
                 time data

See the tutorials for some examples. There is also a notebook about the theory.

Created on Wed Nov  2 10:29:32 2016

@author: giroux
"""
import sys
from multiprocessing import Process, Queue

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from scipy.linalg import lstsq
import matplotlib.pyplot as plt

from ttcrpy.rgrid import Grid3d

# %% hypoloc


def hypoloc(data, rcv, V, hinit, maxit, convh, tol=1e-6, verbose=False):
    """
    Locate hypocenters for constant velocity model.

    Parameters
    ----------
    data  : array_like, shape (m, 3)
        input data
             first column is event ID number
             second column is arrival time
             third column is receiver index
    rcv : array_like, shape (n, 3)
        coordinates of receivers
            first column is easting
            second column is northing
            third column is elevation
    V : float
        wave velocity
    hinit : array_like, shape (m, 3)
        initial hypocenter coordinate.  The format is the same as for data
    maxit : int
        max number of iterations
    convh : float
        convergence criterion (units of distance)
    tol : float
        Stopping tolerance
    verbose : bool
        run in verbose mode

    Returns
    -------
    loc : ndarray of float
        hypocenter coordinates
    res : ndarray of float
        norm of residuals at each iteration for each event (nev x maxit)
    """

    if verbose:
        print('\n *** Hypocenter inversion ***\n')
    evID = np.unique(data[:, 0])
    loc = hinit.copy()
    res = np.zeros((evID.size, maxit+1))
    nev = 0
    for eid in evID:
        ind = eid == data[:, 0]

        ix = data[ind, 2]
        x = np.zeros((ix.size, 3))
        for i in np.arange(ix.size):
            x[i, :] = rcv[int(1.e-6 + ix[i]), :]
        t = data[ind, 1]

        inh = eid == loc[:, 0]
        if verbose:
            print('Locating hypocenters no ' + str(int(1.e-6 + eid)))
            sys.stdout.flush()

        xinit = loc[inh, 2:]
        tinit = loc[inh, 1]

        dx = x[:, 0] - xinit[0, 0]
        dy = x[:, 1] - xinit[0, 1]
        dz = x[:, 2] - xinit[0, 2]
        ds = np.sqrt(dx * dx + dy * dy + dz * dz)
        tcalc = tinit + ds / V

        r = t - tcalc
        res[nev, 0] = np.linalg.norm(r)

        for it in range(maxit):

            H = np.ones((x.shape[0], 4))
            H[:, 1] = -1.0 / V * dx / ds
            H[:, 2] = -1.0 / V * dy / ds
            H[:, 3] = -1.0 / V * dz / ds

            # dh,residuals,rank, s = np.linalg.H.T.dot((H, r)
            dh = np.linalg.solve(H.T.dot(H), H.T.dot(r))
            if not np.all(np.isfinite(dh)):
                try:
                    U, S, VVh = np.linalg.svd(H.T.dot(H) + 1e-9 * np.eye(4))
                    VV = VVh.T
                    dh = np.dot(VV, np.dot(U.T, H.T.dot(r)) / S)
                except np.linalg.linalg.LinAlgError:
                    print(
                        '  Event could not be relocated (iteration no ' +
                        str(it) +
                        '), skipping')
                    sys.stdout.flush()
                    break

            loc[inh, 1:] += dh

            xinit = loc[inh, 2:]
            tinit = loc[inh, 1]

            dx = x[:, 0] - xinit[0, 0]
            dy = x[:, 1] - xinit[0, 1]
            dz = x[:, 2] - xinit[0, 2]
            ds = np.sqrt(dx * dx + dy * dy + dz * dz)
            tcalc = tinit + ds / V

            r = t - tcalc
            res[nev, it+1] = np.linalg.norm(r)

            if np.sum(np.abs(dh[1:]) < convh) == 3 or res[nev, it+1] < tol:
                if verbose:
                    print('     Converged at iteration ' + str(it + 1))
                    sys.stdout.flush()
                break
        else:
            if verbose:
                print('     Reached max number of iteration (' +
                      str(maxit) + ')')
                sys.stdout.flush()

        nev += 1

    if verbose:
        print('\n ** Inversion complete **\n', flush=True)

    return loc, res


def hypolocPS(data, rcv, V, hinit, maxit, convh, tol=1e-6, verbose=False):
    """
    Locate hypocenters for constant velocity model

    Parameters
    ----------
    data : array_like, shape (m, 6)
        input data
            first column is event ID number
            second column is arrival time
            third column is receiver index
            fourth column is code for wave type: 0 for P-wave and 1 for S-wave
    rcv : array_like, shape (n, 3)
        coordinates of receivers
            first column is easting
            second column is northing
            third column is elevation
    V : tuple of float
        wave velocities, 1st value is for P-wave, 2nd for S-wave
    hinit : array_like, shape (m, 6)
        initial hypocenter coordinate.  The format is the same as for data
    maxit : int
        max number of iterations
    convh : float
        convergence criterion (units of distance)
    tol : float
        Stopping tolerance
    verbose : bool
        run in verbose mode

    Returns
    -------
    loc : ndarray of float
        hypocenter coordinates
    res : ndarray of float, shape (nev, maxit)
        norm of residuals at each iteration for each event
    """

    if verbose:
        print('\n *** Hypocenter inversion  --  P and S-wave data ***\n')
    evID = np.unique(data[:, 0])
    loc = hinit.copy()
    res = np.zeros((evID.size, maxit+1))
    nev = 0

    # set origin time to 0 and offset data accordingly
    data = data.copy()
    hyp0_dt = {}
    for n in range(loc.shape[0]):
        hyp0_dt[loc[n, 0]] = loc[n, 1]
        loc[n, 1] -= hyp0_dt[loc[n, 0]]

    for n in range(data.shape[0]):
        try:
            dt = hyp0_dt[data[n, 0]]
            data[n, 1] -= dt
        except KeyError():
            raise ValueError('Event ' +
                             str(data[n, 0]) +
                             ' not found in hinit')

    for eid in evID:
        ind = eid == data[:, 0]

        ix = data[ind, 2]
        x = np.zeros((ix.size, 3))
        for i in np.arange(ix.size):
            x[i, :] = rcv[int(1.e-6 + ix[i]), :]

        t = data[ind, 1]
        ph = data[ind, 3]
        vel = np.zeros((len(ph),))
        for n in range(len(ph)):
            vel[n] = V[int(1.e-6 + ph[n])]

        inh = eid == loc[:, 0]
        if verbose:
            print('Locating hypocenters no ' + str(int(1.e-6 + eid)))
            sys.stdout.flush()

        xinit = loc[inh, 2:]
        tinit = loc[inh, 1]

        dx = x[:, 0] - xinit[0, 0]
        dy = x[:, 1] - xinit[0, 1]
        dz = x[:, 2] - xinit[0, 2]
        ds = np.sqrt(dx * dx + dy * dy + dz * dz)
        tcalc = tinit + ds / vel

        r = t - tcalc
        res[nev, 0] = np.linalg.norm(r)

        for it in range(maxit):

            H = np.ones((x.shape[0], 4))
            H[:, 1] = -1.0 / vel * dx / ds
            H[:, 2] = -1.0 / vel * dy / ds
            H[:, 3] = -1.0 / vel * dz / ds

            # dh,residuals,rank, s = lsqsq(H, r)
            try:
                dh = np.linalg.solve(H.T.dot(H), H.T.dot(r))
            except np.linalg.linalg.LinAlgError:
                try:
                    U, S, VVh = np.linalg.svd(H.T.dot(H) + 1e-9 * np.eye(4))
                    VV = VVh.T
                    dh = np.dot(VV, np.dot(U.T, H.T.dot(r)) / S)
                except np.linalg.linalg.LinAlgError:
                    print(
                        '  Event could not be relocated (iteration no ' +
                        str(it) +
                        '), skipping')
                    sys.stdout.flush()
                    break

            loc[inh, 1:] += dh

            xinit = loc[inh, 2:]
            tinit = loc[inh, 1]

            dx = x[:, 0] - xinit[0, 0]
            dy = x[:, 1] - xinit[0, 1]
            dz = x[:, 2] - xinit[0, 2]
            ds = np.sqrt(dx * dx + dy * dy + dz * dz)
            tcalc = tinit + ds / vel

            r = t - tcalc
            res[nev, it+1] = np.linalg.norm(r)

            if np.sum(np.abs(dh[1:]) < convh) == 3 or res[nev, it+1] < tol:
                if verbose:
                    print('     Converged at iteration ' + str(it + 1))
                    sys.stdout.flush()
                break
        else:
            if verbose:
                print('     Reached max number of iteration (' +
                      str(maxit) + ')')
                sys.stdout.flush()

        nev += 1

    # add time offset
    for n in range(loc.shape[0]):
        loc[n, 1] += hyp0_dt[loc[n, 0]]

    if verbose:
        print('\n ** Inversion complete **\n', flush=True)

    return loc, res


# %% InvParams
class InvParams():
    def __init__(self, maxit, maxit_hypo, conv_hypo, Vlim, dmax, lagrangians,
                 invert_vel=True, invert_VsVp=True, hypo_2step=False,
                 use_sc=True, constr_sc=True, show_plots=False, save_V=False,
                 save_rp=False, verbose=True):
        """
        Parameters
        ----------
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
                        wzK   : weight for vertical smoothing (w.r. to
                                horizontal smoothing)
        invert_vel  : perform velocity inversion if True (True by default)
        invert_VsVp : find Vs/Vp ratio rather that Vs (True by default)
        hypo_2step  : Hypocenter relocation done in 2 steps (False by default)
                        Step 1: longitude and latitude only allowed to vary
                        Step 2: all 4 parameters allowed to vary
        use_sc      : Use static corrections
        constr_sc   : Constrain sum of P-wave static corrections to zero
        show_plots  : show various plots during inversion (True by default)
        save_V      : save intermediate velocity models (False by default)
                        save in vtk format if VTK module can be found
        save_rp     : save ray paths (False by default)
                        VTK module must be installed
        verbose     : print information message about inversion progression
                      (True by default)

        """
        self.maxit = maxit
        self.maxit_hypo = maxit_hypo
        self.conv_hypo = conv_hypo
        self.Vpmin = Vlim[0]
        self.Vpmax = Vlim[1]
        self.PAp = Vlim[2]
        if len(Vlim) > 3:
            self.Vsmin = Vlim[3]
            self.Vsmax = Vlim[4]
            self.PAs = Vlim[5]
        self.dVp_max = dmax[0]
        self.dx_max = dmax[1]
        self.dt_max = dmax[2]
        if len(dmax) > 3:
            self.dVs_max = dmax[3]
        self.λ = lagrangians[0]
        self.γ = lagrangians[1]
        self.α = lagrangians[2]
        self.wzK = lagrangians[3]
        self.invert_vel = invert_vel
        self.invert_VsVp = invert_VsVp
        self.hypo_2step = hypo_2step
        self.use_sc = use_sc
        self.constr_sc = constr_sc
        self.show_plots = show_plots
        self.save_V = save_V
        self.save_rp = save_rp
        self.verbose = verbose
        self._final_iteration = False

# %% joint hypocenter - velocity


def jointHypoVel(par, grid, data, rcv, Vinit, hinit, caldata=np.array([]),
                 Vpts=np.array([])):
    """
    Joint hypocenter-velocity inversion on a regular grid

    Parameters
    ----------
    par     : instance of InvParams
    grid    : instance of Grid3d
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
               3rd column is event easting
               4th column is event northing
               5th column is event elevation
               *** important ***
               for efficiency reason when computing matrix M, initial
               hypocenters should _not_ be equal for any two event, e.g. they
               shoud all be different
    caldata : calibration shot data, numpy array with 8 columns
               1st column is cal shot ID number
               2nd column is arrival time
               3rd column is receiver index
               4th column is source easting
               5th column is source northing
               6th column is source elevation
               *** important ***
               cal shot data should sorted by cal shot ID first, then by
               receiver index
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

    evID = np.unique(data[:, 0])
    nev = evID.size
    if par.use_sc:
        nsta = rcv.shape[0]
    else:
        nsta = 0

    sc = np.zeros(nsta)
    hyp0 = hinit.copy()
    nslowness = grid.nparams

    rcv_data = np.empty((data.shape[0], 3))
    for ne in np.arange(nev):
        indr = np.nonzero(data[:, 0] == evID[ne])[0]
        for i in indr:
            rcv_data[i, :] = rcv[int(1.e-6 + data[i, 2])]

    if data.shape[0] > 0:
        tobs = data[:, 1]
    else:
        tobs = np.array([])

    if caldata.shape[0] > 0:
        calID = np.unique(caldata[:, 0])
        ncal = calID.size
        hcal = np.column_stack(
            (caldata[:, 0], np.zeros(caldata.shape[0]), caldata[:, 3:6]))
        tcal = caldata[:, 1]
        rcv_cal = np.empty((caldata.shape[0], 3))
        Lsc_cal = []
        for nc in range(ncal):
            indr = np.nonzero(caldata[:, 0] == calID[nc])[0]
            nst = np.sum(indr.size)
            for i in indr:
                rcv_cal[i, :] = rcv[int(1.e-6 + caldata[i, 2])]
            if par.use_sc:
                tmp = np.zeros((nst, nsta))
                for n in range(nst):
                    tmp[n, int(1.e-6 + caldata[indr[n], 2])] = 1.0
                Lsc_cal.append(sp.csr_matrix(tmp))
    else:
        ncal = 0
        tcal = np.array([])

    if np.isscalar(Vinit):
        V = Vinit + np.zeros(nslowness)
        s = np.ones(nslowness) / Vinit
    else:
        V = Vinit
        s = 1. / Vinit

    if Vpts.size > 0:
        if Vpts.shape[1] > 4:           # check if we have Vs data in array
            itmp = Vpts[:, 4] == 0
            Vpts = Vpts[itmp, :4]       # keep only Vp data

    if par.verbose:
        print('\n *** Joint hypocenter-velocity inversion ***\n')

    if par.invert_vel:
        resV = np.zeros(par.maxit + 1)
        resAxb = np.zeros(par.maxit)

        P = np.ones(nslowness)
        dP = sp.csr_matrix(
            (np.ones(nslowness), (np.arange(
                nslowness, dtype=np.int64), np.arange(
                nslowness, dtype=np.int64))), shape=(
                nslowness, nslowness))

        Spmax = 1. / par.Vpmin
        Spmin = 1. / par.Vpmax

        deltam = np.zeros(nslowness + nsta)
        if par.constr_sc and par.use_sc:
            u1 = np.ones(nslowness + nsta)
            u1[:nslowness] = 0.0

            i = np.arange(nslowness, nslowness + nsta)
            j = np.arange(nslowness, nslowness + nsta)
            nn = i.size
            i = np.kron(i, np.ones((nn,)))
            j = np.kron(np.ones((nn,)), j)
            u1Tu1 = sp.csr_matrix((np.ones((i.size,)), (i, j)),
                                  shape=(nslowness + nsta, nslowness + nsta))
        else:
            u1 = np.zeros(nslowness + nsta)
            u1Tu1 = sp.csr_matrix((nslowness + nsta, nslowness + nsta))

        if Vpts.size > 0:
            if par.verbose:
                print('Building velocity data point matrix D')
                sys.stdout.flush()
            D = grid.compute_D(Vpts[:, 1:])
            D1 = sp.hstack((D, sp.coo_matrix((Vpts.shape[0], nsta)))).tocsr()
            Spts = 1. / Vpts[:, 0]
        else:
            D = 0.0

        if par.verbose:
            print('Building regularization matrix K')
            sys.stdout.flush()
        Kx, Ky, Kz = grid.compute_K()
        Kx1 = sp.hstack((Kx, sp.coo_matrix((nslowness, nsta)))).tocsr()
        KtK = Kx1.T * Kx1
        Ky1 = sp.hstack((Ky, sp.coo_matrix((nslowness, nsta)))).tocsr()
        KtK += Ky1.T * Ky1
        Kz1 = sp.hstack((Kz, sp.coo_matrix((nslowness, nsta)))).tocsr()
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

        par._final_iteration = it == par.maxit - 1

        if par.invert_vel:
            if par.verbose:
                print(
                    '\nIteration {0:d} - Updating velocity model'.format(it+1))
                print('  Updating penalty vector')
                sys.stdout.flush()

            # compute vector C
            cx = Kx * s
            cy = Ky * s
            cz = Kz * s

            # compute dP/dV, matrix of penalties derivatives
            for n in np.arange(nslowness):
                if s[n] < Spmin:
                    P[n] = par.PAp * (Spmin - s[n])
                    dP[n, n] = -par.PAp
                elif s[n] > Spmax:
                    P[n] = par.PAp * (s[n] - Spmax)
                    dP[n, n] = par.PAp
                else:
                    P[n] = 0.0
                    dP[n, n] = 0.0
            if par.verbose:
                npel = np.sum(P != 0.0)
                if npel > 0:
                    print('    Penalties applied at {0:d} nodes'.format(npel))

            if par.verbose:
                print('  Raytracing')
                sys.stdout.flush()

            if nev > 0:
                hyp = np.empty((data.shape[0], 5))
                for ne in np.arange(nev):
                    indh = np.nonzero(hyp0[:, 0] == evID[ne])[0]
                    indr = np.nonzero(data[:, 0] == evID[ne])[0]
                    for i in indr:
                        hyp[i, :] = hyp0[indh[0], :]
                tcalc, rays, Lev = grid.raytrace(hyp, rcv_data, s,
                                                 return_rays=True,
                                                 compute_L=True)
                s0 = grid.get_s0(hyp)
            else:
                tcalc = np.array([])

            if ncal > 0:
                tcalc_cal, Lcal = grid.raytrace(hcal, rcv_cal, s,
                                                compute_L=True)
            else:
                tcalc_cal = np.array([])

            r1a = tobs - tcalc
            r1 = tcal - tcalc_cal
            if r1a.size > 0:
                r1 = np.hstack((np.zeros(data.shape[0] - 4 * nev), r1))

            if par.show_plots:
                plt.figure(1)
                plt.cla()
                plt.plot(r1a, 'o')
                plt.title('Residuals - Iteration {0:d}'.format(it + 1))
                plt.show(block=False)
                plt.pause(0.0001)

            resV[it] = np.linalg.norm(np.hstack((r1a, r1)))

            # initializing matrix M; matrix of partial derivatives of velocity
            # dt/dV
            if par.verbose:
                print('  Building matrix L')
                sys.stdout.flush()

            L1 = None
            ir1 = 0
            for ne in range(nev):
                if par.verbose:
                    print('    Event ID ' + str(int(1.e-6 + evID[ne])))
                    sys.stdout.flush()

                indh = np.nonzero(hyp0[:, 0] == evID[ne])[0]
                indr = np.nonzero(data[:, 0] == evID[ne])[0]

                nst = np.sum(indr.size)
                nst2 = nst - 4
                H = np.ones((nst, 4))
                for ns in range(nst):
                    raysi = rays[indr[ns]]
                    S0 = s0[indr[ns]]

                    d = (raysi[1, :] - hyp0[indh, 2:]).flatten()
                    ds = np.sqrt(np.sum(d * d))
                    H[ns, 1] = -S0 * d[0] / ds
                    H[ns, 2] = -S0 * d[1] / ds
                    H[ns, 3] = -S0 * d[2] / ds

                Q, _ = np.linalg.qr(H, mode='complete')
                T = sp.csr_matrix(Q[:, 4:]).T
                L = sp.csr_matrix(Lev[ne], shape=(nst, nslowness))
                if par.use_sc:
                    Lsc = np.zeros((nst, nsta))
                    for ns in range(nst):
                        Lsc[ns, int(1.e-6 + data[indr[ns], 2])] = 1.
                    L = sp.hstack((L, sp.csr_matrix(Lsc)))

                L = T * L

                if L1 is None:
                    L1 = L
                else:
                    L1 = sp.vstack((L1, L))

                r1[ir1 + np.arange(nst2, dtype=np.int64)] = T.dot(r1a[indr])
                ir1 += nst2

            for nc in range(ncal):
                L = Lcal[nc]
                if par.use_sc:
                    L = sp.hstack((L, Lsc_cal[nc]))
                if L1 is None:
                    L1 = L
                else:
                    L1 = sp.vstack((L1, L))

            if par.verbose:
                print('  Assembling matrices and solving system')
                sys.stdout.flush()

            ssc = -np.sum(sc)  # u1.T * deltam

            dP1 = sp.hstack(
                (dP, sp.csr_matrix(
                    np.zeros(
                        (nslowness, nsta))))).tocsr()  # dP prime

            # compute A & h for inversion

            L1 = L1.tocsr()

            A = L1.T * L1
            nM = spl.norm(A)

            λ = par.λ * nM / nK

            A += λ * KtK

            tmp = dP1.T * dP1
            nP = spl.norm(tmp)
            if nP != 0.0:
                γ = par.γ * nM / nP
            else:
                γ = par.γ

            A += γ * tmp
            A += u1Tu1

            b = L1.T * r1
            tmp2x = Kx1.T * cx
            tmp2y = Ky1.T * cy
            tmp2z = Kz1.T * cz
            tmp3 = dP1.T * P
            tmp = u1 * ssc
            b += - λ * tmp2x - λ * tmp2y - par.wzK * λ * tmp2z - γ * tmp3 - tmp

            if Vpts.shape[0] > 0:
                tmp = D1.T * D1
                nD = spl.norm(tmp)
                α = par.α * nM / nD
                A += α * tmp
                b += α * D1.T * (Spts - D * s)

            if par.verbose:
                print('    calling minres with system of '
                      'size {0:d} x {1:d}'.format(A.shape[0], A.shape[1]))
                sys.stdout.flush()
            x = spl.minres(A, b)

            deltam = x[0]
            resAxb[it] = np.linalg.norm(A * deltam - b)

            dmean = np.mean(np.abs(deltam[:nslowness]))
            if dmean > par.dVp_max:
                if par.verbose:
                    print('  Scaling Slowness perturbations by {0:e}'.format(
                          1. / (par.dVp_max * dmean)))
                deltam[:nslowness] = deltam[:nslowness] / (par.dVp_max * dmean)

            s += deltam[:nslowness]
            V = 1. / s
            sc += deltam[nslowness:]

            if par.save_V:
                if par.verbose:
                    print('  Saving Velocity model')
                if 'vtk' in sys.modules:
                    grid.to_vtk({'Vp': V}, 'Vp{0:02d}'.format(it + 1))

            grid.set_slowness(s)

        if nev > 0:
            if par.verbose:
                print('Iteration {0:d} - Relocating events'.format(it + 1))
                sys.stdout.flush()

            if grid.n_threads == 1 or nev < grid.n_threads:
                for ne in range(nev):
                    _reloc(ne, par, grid, evID, hyp0, data, rcv, tobs)
            else:
                # run in parallel
                blk_size = np.zeros((grid.n_threads,), dtype=np.int64)
                nj = nev
                while nj > 0:
                    for n in range(grid.n_threads):
                        blk_size[n] += 1
                        nj -= 1
                        if nj == 0:
                            break
                processes = []
                blk_start = 0
                h_queue = Queue()
                for n in range(grid.n_threads):
                    blk_end = blk_start + blk_size[n]
                    p = Process(
                        target=_rl_worker,
                        args=(
                            n,
                            blk_start,
                            blk_end,
                            par,
                            grid,
                            evID,
                            hyp0,
                            data,
                            rcv,
                            tobs,
                            h_queue),
                        daemon=True)
                    processes.append(p)
                    p.start()
                    blk_start += blk_size[n]

                for ne in range(nev):
                    h, indh = h_queue.get()
                    hyp0[indh, :] = h

    if par.invert_vel:
        if nev > 0:
            hyp = np.empty((data.shape[0], 5))
            for ne in np.arange(nev):
                indh = np.nonzero(hyp0[:, 0] == evID[ne])[0]
                indr = np.nonzero(data[:, 0] == evID[ne])[0]
                for i in indr:
                    hyp[i, :] = hyp0[indh[0], :]
            tcalc = grid.raytrace(hyp, rcv_data, s)
        else:
            tcalc = np.array([])

        if ncal > 0:
            tcalc_cal = grid.raytrace(hcal, rcv_cal, s)
        else:
            tcalc_cal = np.array([])

        r1a = tobs - tcalc
        r1 = tcal - tcalc_cal
        if r1a.size > 0:
            r1 = np.hstack((np.zeros(data.shape[0] - 4 * nev), r1))

        if par.show_plots:
            plt.figure(1)
            plt.cla()
            plt.plot(r1a, 'o')
            plt.title('Residuals - Final step')
            plt.show(block=False)
            plt.pause(0.0001)

        resV[-1] = np.linalg.norm(np.hstack((r1a, r1)))

        r1 = r1.reshape(-1, 1)
        r1a = r1a.reshape(-1, 1)

    if par.verbose:
        print('\n ** Inversion complete **\n', flush=True)

    return hyp0, V, sc, (resV, resAxb)


def _rl_worker(
        thread_no,
        istart,
        iend,
        par,
        grid,
        evID,
        hyp0,
        data,
        rcv,
        tobs,
        h_queue):
    for ne in range(istart, iend):
        h, indh = _reloc(ne, par, grid, evID, hyp0, data, rcv, tobs, thread_no)
        h_queue.put((h, indh))
    h_queue.close()


def _reloc(ne, par, grid, evID, hyp0, data, rcv, tobs, thread_no=None):

    if par.verbose:
        print('  Updating event ID {0:d} ({1:d}/{2:d})'.format(
            int(1.e-6 + evID[ne]), ne + 1, evID.size))
        sys.stdout.flush()

    indh = np.nonzero(hyp0[:, 0] == evID[ne])[0][0]
    indr = np.nonzero(data[:, 0] == evID[ne])[0]

    hyp_save = hyp0[indh, :].copy()

    nst = np.sum(indr.size)

    hyp = np.empty((nst, 5))
    stn = np.empty((nst, 3))
    for i in range(nst):
        hyp[i, :] = hyp0[indh, :]
        stn[i, :] = rcv[int(1.e-6 + data[indr[i], 2]), :]

    if par.hypo_2step:
        if par.verbose:
            print('    Updating latitude & longitude', end='')
            sys.stdout.flush()
        H = np.ones((nst, 2))
        for itt in range(par.maxit_hypo):
            for i in range(nst):
                hyp[i, :] = hyp0[indh, :]

            tcalc, rays = grid.raytrace(hyp, stn, thread_no=thread_no,
                                        return_rays=True)
            s0 = grid.get_s0(hyp)
            for ns in range(nst):
                raysi = rays[ns]
                S0 = s0[ns]

                d = (raysi[1, :] - hyp0[indh, 2:]).flatten()
                ds = np.sqrt(np.sum(d * d))
                H[ns, 0] = -S0 * d[0] / ds
                H[ns, 1] = -S0 * d[1] / ds

            r = tobs[indr] - tcalc
            x = lstsq(H, r)
            deltah = x[0]

            if np.sum(np.isfinite(deltah)) != deltah.size:
                try:
                    U, S, VVh = np.linalg.svd(H.T.dot(H) + 1e-9 * np.eye(2))
                    VV = VVh.T
                    deltah = np.dot(VV, np.dot(U.T, H.T.dot(r)) / S)
                except np.linalg.linalg.LinAlgError:
                    print(' - Event could not be relocated, '
                          'resetting and exiting')
                    hyp0[indh, :] = hyp_save
                    return hyp_save, indh

            for n in range(2):
                if np.abs(deltah[n]) > par.dx_max:
                    deltah[n] = par.dx_max * np.sign(deltah[n])

            new_hyp = hyp0[indh, :].copy()
            new_hyp[2:4] += deltah
            if grid.is_outside(new_hyp[2:].reshape((1, 3))):
                print('  Event could not be relocated inside the grid'
                      ' ({0:f}, {1:f}, {2:f}), resetting and exiting'.format(
                        new_hyp[2],
                        new_hyp[3],
                        new_hyp[4]))
                hyp0[indh, :] = hyp_save
                return hyp_save, indh

            hyp0[indh, 2:4] += deltah

            if np.sum(np.abs(deltah) < par.conv_hypo) == 2:
                if par.verbose:
                    print(' - converged at iteration ' + str(itt + 1))
                    sys.stdout.flush()
                break

        else:
            if par.verbose:
                print(' - reached max number of iterations')
                sys.stdout.flush()

    if par.verbose:
        print('    Updating all hypocenter params', end='')
        sys.stdout.flush()

    H = np.ones((nst, 4))
    for itt in range(par.maxit_hypo):
        for i in range(nst):
            hyp[i, :] = hyp0[indh, :]

        tcalc, rays = grid.raytrace(hyp, stn, thread_no=thread_no,
                                    return_rays=True)
        s0 = grid.get_s0(hyp)
        for ns in range(nst):
            raysi = rays[ns]
            S0 = s0[ns]

            d = (raysi[1, :] - hyp0[indh, 2:]).flatten()
            ds = np.sqrt(np.sum(d * d))
            H[ns, 1] = -S0 * d[0] / ds
            H[ns, 2] = -S0 * d[1] / ds
            H[ns, 3] = -S0 * d[2] / ds

        r = tobs[indr] - tcalc
        x = lstsq(H, r)
        deltah = x[0]

        if np.sum(np.isfinite(deltah)) != deltah.size:
            try:
                U, S, VVh = np.linalg.svd(H.T.dot(H) + 1e-9 * np.eye(4))
                VV = VVh.T
                deltah = np.dot(VV, np.dot(U.T, H.T.dot(r)) / S)
            except np.linalg.linalg.LinAlgError:
                print('  Event could not be relocated, resetting and exiting')
                hyp0[indh, :] = hyp_save
                return hyp_save, indh

        if np.abs(deltah[0]) > par.dt_max:
            deltah[0] = par.dt_max * np.sign(deltah[0])
        for n in range(1, 4):
            if np.abs(deltah[n]) > par.dx_max:
                deltah[n] = par.dx_max * np.sign(deltah[n])

        new_hyp = hyp0[indh, 1:] + deltah
        if grid.is_outside(new_hyp[1:].reshape((1, 3))):
            print('  Event could not be relocated inside the grid'
                  ' ({0:f}, {1:f}, {2:f}), resetting and exiting'.format(
                    new_hyp[1],
                    new_hyp[2],
                    new_hyp[3]))
            hyp0[indh, :] = hyp_save
            return hyp_save, indh

        hyp0[indh, 1:] += deltah

        if np.sum(np.abs(deltah[1:]) < par.conv_hypo) == 3:
            if par.verbose:
                print(' - converged at iteration ' + str(itt + 1))
                sys.stdout.flush()
            break

    else:
        if par.verbose:
            print(' - reached max number of iterations')
            sys.stdout.flush()

    if par.save_rp and 'vtk' in sys.modules and par._final_iteration:
        if par.verbose:
            print('    Saving raypaths')
        filename = 'raypaths'
        key = 'ev_{0:d}'.format(int(1.e-6 + evID[ne]))
        grid.to_vtk({key: rays}, filename)

    return hyp0[indh, :], indh


def jointHypoVelPS(par, grid, data, rcv, Vinit, hinit, caldata=np.array([]),
                   Vpts=np.array([])):
    """
    Joint hypocenter-velocity inversion on a regular grid

    Parameters
    ----------
    par     : instance of InvParams
    grid    : instances of Grid3d
    data    : a numpy array with 5 columns
               1st column is event ID number
               2nd column is arrival time
               3rd column is receiver index
               4th column is code for wave type: 0 for P-wave and 1 for S-wave
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
               3rd column is event easting
               4th column is event northing
               5th column is event elevation
               *** important ***
               for efficiency reason when computing matrix M, initial
               hypocenters should _not_ be equal for any two event, e.g. they
               shoud all be different
    caldata : calibration shot data, numpy array with 8 columns
               1st column is cal shot ID number
               2nd column is arrival time
               3rd column is receiver index
               4th column is source easting
               5th column is source northing
               6th column is source elevation
               7th column is code for wave phase: 0 for P-wave and 1 for S-wave
               *** important ***
               cal shot data should sorted by cal shot ID first, then by
               receiver index
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

    if grid.n_threads > 1:
        # we need a second instance for parallel computations
        grid_s = Grid3d(grid.x, grid.y, grid.z, grid.n_threads,
                        cell_slowness=True)
    else:
        grid_s = grid

    evID = np.unique(data[:, 0])
    nev = evID.size
    if par.use_sc:
        nsta = rcv.shape[0]
    else:
        nsta = 0

    sc_p = np.zeros(nsta)
    sc_s = np.zeros(nsta)
    hyp0 = hinit.copy()

    # set origin time to 0 and offset data accordingly
    data = data.copy()
    hyp0_dt = {}
    for n in range(hyp0.shape[0]):
        hyp0_dt[hyp0[n, 0]] = hyp0[n, 1]
        hyp0[n, 1] -= hyp0_dt[hyp0[n, 0]]

    for n in range(data.shape[0]):
        try:
            dt = hyp0_dt[data[n, 0]]
            data[n, 1] -= dt
        except KeyError():
            raise ValueError('Event ' +
                             str(data[n, 0]) +
                             ' not fount in hinit')

    nslowness = grid.nparams

    # sort data by seismic phase (P-wave first S-wave second)
    indp = data[:, 3] == 0.0
    inds = data[:, 3] == 1.0
    nttp = np.sum(indp)
    ntts = np.sum(inds)
    datap = data[indp, :]
    datas = data[inds, :]
    data = np.vstack((datap, datas))
    # update indices
    indp = data[:, 3] == 0.0
    inds = data[:, 3] == 1.0

    rcv_datap = np.empty((nttp, 3))
    rcv_datas = np.empty((ntts, 3))
    for ne in np.arange(nev):
        indr = np.nonzero(np.logical_and(data[:, 0] == evID[ne], indp))[0]
        for i in indr:
            rcv_datap[i, :] = rcv[int(1.e-6 + data[i, 2])]
        indr = np.nonzero(np.logical_and(data[:, 0] == evID[ne], inds))[0]
        for i in indr:
            rcv_datas[i - nttp, :] = rcv[int(1.e-6 + data[i, 2])]

    if data.shape[0] > 0:
        tobs = data[:, 1]
    else:
        tobs = np.array([])

    if caldata.shape[0] > 0:
        calID = np.unique(caldata[:, 0])
        ncal = calID.size

        # sort data by seismic phase (P-wave first S-wave second)
        indcalp = caldata[:, 6] == 0.0
        indcals = caldata[:, 6] == 1.0
        nttcalp = np.sum(indcalp)
        nttcals = np.sum(indcals)
        caldatap = caldata[indcalp, :]
        caldatas = caldata[indcals, :]
        caldata = np.vstack((caldatap, caldatas))
        # update indices
        indcalp = caldata[:, 6] == 0.0
        indcals = caldata[:, 6] == 1.0

        hcalp = np.column_stack(
            (caldata[indcalp, 0], np.zeros(nttcalp), caldata[indcalp, 3:6]))
        hcals = np.column_stack(
            (caldata[indcals, 0], np.zeros(nttcals), caldata[indcals, 3:6]))

        rcv_calp = np.empty((nttcalp, 3))
        rcv_cals = np.empty((nttcals, 3))

        tcal = caldata[:, 1]
        Lsc_cal = []
        for nc in range(ncal):
            indrp = np.nonzero(np.logical_and(
                caldata[:, 0] == calID[nc], indcalp))[0]
            for i in indrp:
                rcv_calp[i, :] = rcv[int(1.e-6 + caldata[i, 2])]
            indrs = np.nonzero(np.logical_and(
                caldata[:, 0] == calID[nc], indcals))[0]
            for i in indrs:
                rcv_cals[i - nttcalp, :] = rcv[int(1.e-6 + caldata[i, 2])]

            if par.use_sc:
                Lpsc = np.zeros((indrp.size, nsta))
                Lssc = np.zeros((indrs.size, nsta))
                for ns in range(indrp.size):
                    Lpsc[ns, int(1.e-6 + caldata[indrp[ns], 2])] = 1.
                for ns in range(indrs.size):
                    Lssc[ns, int(1.e-6 + caldata[indrs[ns], 2])] = 1.

                Lsc_cal.append(
                    sp.block_diag(
                        (sp.csr_matrix(Lpsc), sp.csr_matrix(Lssc))))

    else:
        ncal = 0
        tcal = np.array([])

    if np.isscalar(Vinit[0]):
        Vp = np.zeros(nslowness) + Vinit[0]
        s_p = 1. / Vinit[0] + np.zeros(nslowness)
    else:
        Vp = Vinit[0]
        s_p = 1. / Vinit[0]
    if np.isscalar(Vinit[1]):
        Vs = np.zeros(nslowness) + Vinit[1]
        s_s = 1. / Vinit[1] + np.zeros(nslowness)
    else:
        Vs = Vinit[1]
        s_s = 1. / Vinit[1]
    if par.invert_VsVp:
        SsSp = s_s / s_p
        s = np.hstack((s_p, SsSp))
    else:
        s = np.hstack((s_p, s_s))

    # translate dVp_max into slowness (use mean Vp)
    Vp_mean = np.mean(Vp)
    dVp_max = 1. / Vp_mean - 1. / (Vp_mean + par.dVp_max)
    Vs_mean = np.mean(Vs)
    dVs_max = 1. / Vs_mean - 1. / (Vs_mean + par.dVs_max)

    if par.verbose:
        print('\n *** Joint hypocenter-velocity inversion'
              ' -- P and S-wave data ***\n')

    if par.invert_vel:
        resV = np.zeros(par.maxit + 1)
        resAxb = np.zeros(par.maxit)

        P = np.ones(2 * nslowness)
        dP = sp.csr_matrix(
            (np.ones(
                2 * nslowness),
                (np.arange(
                    2 * nslowness,
                    dtype=np.int64),
                 np.arange(
                    2 * nslowness,
                    dtype=np.int64))),
            shape=(
                2 * nslowness,
                2 * nslowness))

        Spmin = 1. / par.Vpmax
        Spmax = 1. / par.Vpmin
        Ssmin = 1. / par.Vsmax
        Ssmax = 1. / par.Vsmin

        deltam = np.zeros(2 * nslowness + 2 * nsta)

        if par.constr_sc and par.use_sc:
            u1 = np.ones(2 * nslowness + 2 * nsta)
            u1[:2 * nslowness] = 0.0
            u1[(2 * nslowness + nsta):] = 0.0

            i = np.arange(2 * nslowness, 2 * nslowness + nsta)
            j = np.arange(2 * nslowness, 2 * nslowness + nsta)
            nn = i.size
            i = np.kron(i, np.ones((nn,)))
            j = np.kron(np.ones((nn,)), j)
            u1Tu1 = sp.csr_matrix((np.ones((i.size,)), (i, j)), shape=(
                2 * nslowness + 2 * nsta, 2 * nslowness + 2 * nsta))
        else:
            u1 = np.zeros(2 * nslowness + 2 * nsta)
            u1Tu1 = sp.csr_matrix(
                (2 * nslowness + 2 * nsta, 2 * nslowness + 2 * nsta))

        Vpts2 = Vpts.copy()
        if Vpts.size > 0:

            if par.verbose:
                print('Building velocity data point matrix D')
                sys.stdout.flush()

            if par.invert_VsVp:
                i_p = np.nonzero(Vpts[:, 4] == 0.0)[0]
                i_s = np.nonzero(Vpts[:, 4] == 1.0)[0]
                for i in i_s:
                    for ii in i_p:
                        d = np.sqrt(np.sum((Vpts2[i, 1:4] - Vpts[ii, 1:4])**2))
                        if d < 0.00001:
                            Vpts2[i, 0] = Vpts[i, 0] / Vpts[ii, 0]
                            break
                    else:
                        raise ValueError('Missing Vp data point for Vs data '
                                         'at ({0:f}, {1:f}, {2:f})'
                                         .format(Vpts[i, 1], Vpts[i, 2],
                                                 Vpts[i, 3]))

            else:
                Vpts2 = Vpts

            if par.invert_VsVp:
                D = grid.compute_D(Vpts2[:, 1:4])
                D = sp.hstack((D, sp.coo_matrix(D.shape))).tocsr()
            else:
                i_p = Vpts2[:, 4] == 0.0
                i_s = Vpts2[:, 4] == 1.0
                Dp = grid.compute_D(Vpts2[i_p, 1:4])
                Ds = grid.compute_D(Vpts2[i_s, 1:4])
                D = sp.block_diag((Dp, Ds)).tocsr()

            D1 = sp.hstack(
                (D, sp.csr_matrix(
                    (Vpts2.shape[0], 2 * nsta)))).tocsr()
            Spts = 1. / Vpts2[:, 0]
        else:
            D = 0.0

        if par.verbose:
            print('Building regularization matrix K')
            sys.stdout.flush()
        Kx, Ky, Kz = grid.compute_K()
        Kx = sp.block_diag((Kx, Kx))
        Ky = sp.block_diag((Ky, Ky))
        Kz = sp.block_diag((Kz, Kz))
        Kx1 = sp.hstack((Kx, sp.coo_matrix((2 * nslowness, 2 * nsta)))).tocsr()
        KtK = Kx1.T * Kx1
        Ky1 = sp.hstack((Ky, sp.coo_matrix((2 * nslowness, 2 * nsta)))).tocsr()
        KtK += Ky1.T * Ky1
        Kz1 = sp.hstack((Kz, sp.coo_matrix((2 * nslowness, 2 * nsta)))).tocsr()
        KtK += par.wzK * Kz1.T * Kz1
        nK = spl.norm(KtK)
        Kx = Kx.tocsr()
        Ky = Ky.tocsr()
        Kz = Kz.tocsr()
    else:
        resV = None
        resAxb = None

    if par.invert_VsVp:
        SsSpmin = par.Vpmin / par.Vsmax
        SsSpmax = par.Vpmax / par.Vsmin

    if par.verbose:
        print('\nStarting iterations')

    for it in np.arange(par.maxit):

        par._final_iteration = it == par.maxit - 1

        if par.invert_vel:
            if par.verbose:
                print('\nIteration {0:d} - Updating velocity '
                      'model\n'.format(it + 1))
                print('  Updating penalty vector')
                sys.stdout.flush()

            # compute vector C
            cx = Kx * s
            cy = Ky * s
            cz = Kz * s

            # compute dP/dV, matrix of penalties derivatives
            for n in np.arange(nslowness):
                if s[n] < Spmin:
                    P[n] = par.PAp * (Spmin - s[n])
                    dP[n, n] = -par.PAp
                elif s[n] > Spmax:
                    P[n] = par.PAp * (s[n] - Spmax)
                    dP[n, n] = par.PAp
                else:
                    P[n] = 0.0
                    dP[n, n] = 0.0
            for n in np.arange(nslowness, 2 * nslowness):
                if par.invert_VsVp:
                    if SsSp[n - nslowness] < SsSpmin:
                        P[n] = par.PAs * (SsSpmin - SsSp[n - nslowness])
                        dP[n, n] = -par.PAs
                    elif SsSp[n - nslowness] > SsSpmax:
                        P[n] = par.PAs * (SsSp[n - nslowness] - SsSpmax)
                        dP[n, n] = par.PAs
                    else:
                        P[n] = 0.0
                        dP[n, n] = 0.0
                else:
                    if s_s[n - nslowness] < Ssmin:
                        P[n] = par.PAs * (Ssmin - Vs[n - nslowness])
                        dP[n, n] = -par.PAs
                    elif s_s[n - nslowness] > Ssmax:
                        P[n] = par.PAs * (s_s[n - nslowness] - Ssmax)
                        dP[n, n] = par.PAs
                    else:
                        P[n] = 0.0
                        dP[n, n] = 0.0

            if par.verbose:
                npel = np.sum(P[:nslowness] != 0.0)
                if npel > 0:
                    print('    P-wave penalties applied at'
                          ' {0:d} nodes'.format(npel))
                npel = np.sum(P[nslowness:2 * nslowness] != 0.0)
                if npel > 0:
                    print('    S-wave penalties applied at'
                          ' {0:d} nodes'.format(npel))

            if par.verbose:
                print('  Raytracing')
                sys.stdout.flush()

            ev_has_p = np.zeros((nev,), dtype=bool)
            ev_has_s = np.zeros((nev,), dtype=bool)
            if nev > 0:
                hyp = np.empty((nttp, 5))
                for ne in np.arange(nev):
                    indh = np.nonzero(hyp0[:, 0] == evID[ne])[0]
                    indrp = np.nonzero(np.logical_and(
                        data[:, 0] == evID[ne], indp))[0]
                    ev_has_p[ne] = indrp.size > 0
                    for i in indrp:
                        hyp[i, :] = hyp0[indh[0], :]

                tcalcp, raysp, Levp = grid.raytrace(hyp, rcv_datap, s_p,
                                                    return_rays=True,
                                                    compute_L=True)
                s0p = grid.get_s0(hyp)

                hyp = np.empty((ntts, 5))
                for ne in np.arange(nev):
                    indh = np.nonzero(hyp0[:, 0] == evID[ne])[0]
                    indrs = np.nonzero(np.logical_and(
                        data[:, 0] == evID[ne], inds))[0]
                    ev_has_s[ne] = indrs.size > 0
                    for i in indrs:
                        hyp[i - nttp, :] = hyp0[indh[0], :]
                tcalcs, rayss, Levs = grid_s.raytrace(hyp, rcv_datas, s_s,
                                                      return_rays=True,
                                                      compute_L=True)
                s0s = grid.get_s0(hyp)

                tcalc = np.hstack((tcalcp, tcalcs))
                s0 = np.hstack((s0p, s0s))
                rays = []
                for r in raysp:
                    rays.append(r)
                for r in rayss:
                    rays.append(r)

                # Merge Levp & Levs
                Lev = [None] * nev
                n_has_p = 0
                n_has_s = 0
                for ne in np.arange(nev):

                    if ev_has_p[ne]:
                        Lp = Levp[n_has_p]
                        n_has_p += 1
                    else:
                        Lp = None
                    if ev_has_s[ne]:
                        Ls = Levs[n_has_s]
                        n_has_s += 1
                    else:
                        Ls = None

                    if par.invert_VsVp:
                        # Block 1991, p. 45
                        if Lp is None:
                            tmp1 = Ls.multiply(np.tile(SsSp, (Ls.shape[0], 1)))
                            tmp2 = Ls.multiply(np.tile(s_p, (Ls.shape[0], 1)))
                            Lev[ne] = sp.hstack((tmp1, tmp2))
                        elif Ls is None:
                            Lev[ne] = sp.hstack((Lp, sp.csr_matrix(Lp.shape)))
                        else:
                            tmp1 = Ls.multiply(np.tile(SsSp, (Ls.shape[0], 1)))
                            tmp2 = Ls.multiply(np.tile(s_p, (Ls.shape[0], 1)))
                            tmp2 = sp.hstack((tmp1, tmp2))
                            tmp1 = sp.hstack((Lp, sp.csr_matrix(Lp.shape)))
                            Lev[ne] = sp.vstack((tmp1, tmp2))
                    else:
                        if Lp is None:
                            Lev[ne] = sp.hstack((sp.csr_matrix(Ls.shape), Ls))
                        elif Ls is None:
                            Lev[ne] = sp.hstack((Lp, sp.csr_matrix(Lp.shape)))
                        else:
                            Lev[ne] = sp.block_diag((Lp, Ls))

                    if par.use_sc:
                        if ev_has_p[ne]:
                            indrp = np.nonzero(np.logical_and(
                                data[:, 0] == evID[ne], indp))[0]
                            Lpsc = np.zeros((indrp.size, nsta))
                            for ns in range(indrp.size):
                                Lpsc[ns, int(1.e-6 + data[indrp[ns], 2])] = 1.
                        else:
                            Lpsc = None

                        if ev_has_s[ne]:
                            indrs = np.nonzero(np.logical_and(
                                data[:, 0] == evID[ne], inds))[0]
                            Lssc = np.zeros((indrs.size, nsta))
                            for ns in range(indrs.size):
                                Lssc[ns, int(1.e-6 + data[indrs[ns], 2])] = 1.
                        else:
                            Lssc = None

                        if Lpsc is None:
                            Lsc = sp.hstack((sp.csr_matrix(Lssc.shape), Lssc))
                        elif Lssc is None:
                            Lsc = sp.hstack((Lpsc, sp.csr_matrix(Lpsc.shape)))
                        else:
                            Lsc = sp.block_diag(
                                (sp.csr_matrix(Lpsc), sp.csr_matrix(Lssc)))

                        # add terms for station corrections after terms for
                        # velocity because solution vector contains
                        # [Vp Vs sc_p sc_s] in that order
                        Lev[ne] = sp.hstack((Lev[ne], Lsc))

            else:
                tcalc = np.array([])

            if ncal > 0:
                tcalcp_cal, Lp_cal = grid.raytrace(hcalp, rcv_calp, s_p,
                                                   compute_L=True)
                if nttcals > 0:
                    tcalcs_cal, Ls_cal = grid_s.raytrace(hcals, rcv_cals, s_s,
                                                         compute_L=True)
                    tcalc_cal = np.hstack((tcalcp_cal, tcalcs_cal))
                else:
                    tcalc_cal = tcalcp_cal
            else:
                tcalc_cal = np.array([])

            r1a = tobs - tcalc
            r1 = tcal - tcalc_cal
            if r1a.size > 0:
                r1 = np.hstack((np.zeros(data.shape[0] - 4 * nev), r1))

            resV[it] = np.linalg.norm(
                np.hstack((tobs - tcalc, tcal - tcalc_cal)))

            if par.show_plots:
                plt.figure(1)
                plt.cla()
                plt.plot(r1a, 'o')
                plt.title('Residuals - Iteration {0:d}'.format(it + 1))
                plt.show(block=False)
                plt.pause(0.0001)

            # initializing matrix M; matrix of partial derivatives of velocity
            # dt/dV
            if par.verbose:
                print('  Building matrix M')
                sys.stdout.flush()

            L1 = None
            ir1 = 0
            for ne in range(nev):
                if par.verbose:
                    print('    Event ID ' + str(int(1.e-6 + evID[ne])))
                    sys.stdout.flush()

                indh = np.nonzero(hyp0[:, 0] == evID[ne])[0]
                indr = np.nonzero(data[:, 0] == evID[ne])[0]

                nst = np.sum(indr.size)
                nst2 = nst - 4
                H = np.ones((nst, 4))
                for ns in range(nst):
                    raysi = rays[indr[ns]]
                    S0 = s0[indr[ns]]

                    d = (raysi[1, :] - hyp0[indh, 2:]).flatten()
                    ds = np.sqrt(np.sum(d * d))
                    H[ns, 1] = -S0 * d[0] / ds
                    H[ns, 2] = -S0 * d[1] / ds
                    H[ns, 3] = -S0 * d[2] / ds

                Q, _ = np.linalg.qr(H, mode='complete')
                T = sp.csr_matrix(Q[:, 4:]).T
                L = T * Lev[ne]

                if L1 is None:
                    L1 = L
                else:
                    L1 = sp.vstack((L1, L))

                r1[ir1 + np.arange(nst2, dtype=np.int64)] = T.dot(r1a[indr])
                ir1 += nst2

            for nc in range(ncal):
                Lp = Lp_cal[nc]
                if nttcals > 0:
                    Ls = Ls_cal[nc]
                else:
                    Ls = sp.csr_matrix([])

                if par.invert_VsVp:
                    if nttcals > 0:
                        # Block 1991, p. 45
                        tmp1 = Ls.multiply(np.tile(SsSp, (Ls.shape[0], 1)))
                        tmp2 = Ls.multiply(np.tile(Vp, (Ls.shape[0], 1)))
                        tmp2 = sp.hstack((tmp1, tmp2))
                        tmp1 = sp.hstack((Lp, sp.csr_matrix(Lp.shape)))
                        L = sp.vstack((tmp1, tmp2))
                    else:
                        L = sp.hstack((Lp, sp.csr_matrix(Lp.shape)))
                else:
                    L = sp.block_diag((Lp, Ls))

                if par.use_sc:
                    L = sp.hstack((L, Lsc_cal[nc]))

                if L1 is None:
                    L1 = L
                else:
                    L1 = sp.vstack((L1, L))

            if par.verbose:
                print('  Assembling matrices and solving system')
                sys.stdout.flush()

            ssc = -np.sum(sc_p)

            dP1 = sp.hstack(
                (dP, sp.csr_matrix(
                    (2 * nslowness, 2 * nsta)))).tocsr()  # dP prime

            # compute A & h for inversion

            L1 = L1.tocsr()

            A = L1.T * L1

            nM = spl.norm(A)
            λ = par.λ * nM / nK

            A += λ * KtK

            tmp = dP1.T * dP1
            nP = spl.norm(tmp)
            if nP != 0.0:
                γ = par.γ * nM / nP
            else:
                γ = par.γ

            A += γ * tmp
            A += u1Tu1

            b = L1.T * r1
            tmp2x = Kx1.T * cx
            tmp2y = Ky1.T * cy
            tmp2z = Kz1.T * cz
            tmp3 = dP1.T * P
            tmp = u1 * ssc
            b += -λ * tmp2x - λ * tmp2y - par.wzK * λ * tmp2z - γ * tmp3 - tmp

            if Vpts2.shape[0] > 0:
                tmp = D1.T * D1
                nD = spl.norm(tmp)
                α = par.α * nM / nD
                A += α * tmp
                b += α * D1.T * (Spts - D * s)

            if par.verbose:
                print('    calling minres with system of size'
                      ' {0:d} x {1:d}'.format(A.shape[0], A.shape[1]))
                sys.stdout.flush()
            x = spl.minres(A, b)

            deltam = x[0]
            resAxb[it] = np.linalg.norm(A * deltam - b)

            dmax = np.max(np.abs(deltam[:nslowness]))
            if dmax > dVp_max:
                if par.verbose:
                    print('  Scaling P slowness perturbations '
                          'by {0:e}'.format(dVp_max / dmax))
                deltam[:nslowness] = deltam[:nslowness] * dVp_max / dmax
            dmax = np.max(np.abs(deltam[nslowness:2 * nslowness]))
            if dmax > dVs_max:
                if par.verbose:
                    print('  Scaling S slowness perturbations '
                          'by {0:e}'.format(dVs_max / dmax))
                deltam[nslowness:2*nslowness] = deltam[nslowness:2*nslowness] \
                    * dVs_max / dmax

            s += deltam[:2 * nslowness]
            s_p = s[:nslowness]
            if par.invert_VsVp:
                SsSp = s[nslowness:2 * nslowness]
                s_s = SsSp * s_p
            else:
                s_s = s[nslowness:2 * nslowness]
            Vp = 1. / s_p
            Vs = 1. / s_s
            sc_p += deltam[2 * nslowness:2 * nslowness + nsta]
            sc_s += deltam[2 * nslowness + nsta:]

            if par.save_V:
                if par.verbose:
                    print('  Saving Velocity models')
                if 'vtk' in sys.modules:
                    grid.to_vtk({'Vp': Vp}, 'Vp{0:02d}'.format(it + 1))
                if par.invert_VsVp:
                    if 'vtk' in sys.modules:
                        grid.to_vtk({'VsVp': SsSp},
                                    'VsVp{0:02d}'.format(it + 1))
                if 'vtk' in sys.modules:
                    grid.to_vtk({'Vs': Vs}, 'Vs{0:02d}'.format(it + 1))

        if nev > 0:
            if par.verbose:
                print('Iteration {0:d} - Relocating events'.format(it + 1))
                sys.stdout.flush()

            if grid.n_threads == 1 or nev < grid.n_threads:
                for ne in range(nev):
                    _relocPS(ne, par, (grid, grid_s), evID, hyp0, data, rcv,
                             tobs, (s_p, s_s), (indp, inds))
            else:
                # run in parallel
                blk_size = np.zeros((grid.n_threads,), dtype=np.int64)
                nj = nev
                while nj > 0:
                    for n in range(grid.n_threads):
                        blk_size[n] += 1
                        nj -= 1
                        if nj == 0:
                            break
                processes = []
                blk_start = 0
                h_queue = Queue()
                for n in range(grid.n_threads):
                    blk_end = blk_start + blk_size[n]
                    p = Process(
                        target=_rlPS_worker,
                        args=(
                            n,
                            blk_start,
                            blk_end,
                            par,
                            (grid,
                             grid_s),
                            evID,
                            hyp0,
                            data,
                            rcv,
                            tobs,
                            (s_p,
                             s_s),
                            (indp,
                             inds),
                            h_queue),
                        daemon=True)
                    processes.append(p)
                    p.start()
                    blk_start += blk_size[n]

                for ne in range(nev):
                    h, indh = h_queue.get()
                    hyp0[indh, :] = h

    if par.invert_vel:
        if nev > 0:
            hyp = np.empty((nttp, 5))
            for ne in np.arange(nev):
                indh = np.nonzero(hyp0[:, 0] == evID[ne])[0]
                indrp = np.nonzero(np.logical_and(
                    data[:, 0] == evID[ne], indp))[0]
                for i in indrp:
                    hyp[i, :] = hyp0[indh[0], :]

            tcalcp = grid.raytrace(hyp, rcv_datap, s_p)

            hyp = np.empty((ntts, 5))
            for ne in np.arange(nev):
                indh = np.nonzero(hyp0[:, 0] == evID[ne])[0]
                indrs = np.nonzero(np.logical_and(
                    data[:, 0] == evID[ne], inds))[0]
                for i in indrs:
                    hyp[i - nttp, :] = hyp0[indh[0], :]
            tcalcs = grid_s.raytrace(hyp, rcv_datas, s_s)

            tcalc = np.hstack((tcalcp, tcalcs))
        else:
            tcalc = np.array([])

        if ncal > 0:
            tcalcp_cal = grid.raytrace(hcalp, rcv_calp, s_p)
            tcalcs_cal = grid_s.raytrace(hcals, rcv_cals, s_s)
            tcalc_cal = np.hstack((tcalcp_cal, tcalcs_cal))
        else:
            tcalc_cal = np.array([])

        r1a = tobs - tcalc
        r1 = tcal - tcalc_cal
        if r1a.size > 0:
            r1 = np.hstack((np.zeros(data.shape[0] - 4 * nev), r1))

        if par.show_plots:
            plt.figure(1)
            plt.cla()
            plt.plot(r1a, 'o')
            plt.title('Residuals - Final step')
            plt.show(block=False)
            plt.pause(0.0001)

        r1 = r1.reshape(-1, 1)
        r1a = r1a.reshape(-1, 1)

        resV[-1] = np.linalg.norm(np.hstack((tobs - tcalc, tcal - tcalc_cal)))

    # add time offset
    for n in range(hyp0.shape[0]):
        hyp0[n, 1] += hyp0_dt[hyp0[n, 0]]

    if par.verbose:
        print('\n ** Inversion complete **\n', flush=True)

    return hyp0, (Vp, Vs), (sc_p, sc_s), (resV, resAxb)


def _rlPS_worker(
        thread_no,
        istart,
        iend,
        par,
        grid,
        evID,
        hyp0,
        data,
        rcv,
        tobs,
        s,
        ind,
        h_queue):
    for ne in range(istart, iend):
        h, indh = _relocPS(ne, par, grid, evID, hyp0, data,
                           rcv, tobs, s, ind, thread_no)
        h_queue.put((h, indh))
    h_queue.close()


def _relocPS(
        ne,
        par,
        grid,
        evID,
        hyp0,
        data,
        rcv,
        tobs,
        s,
        ind,
        thread_no=None):

    (grid_p, grid_s) = grid
    (indp, inds) = ind
    (s_p, s_s) = s
    if par.verbose:
        print('  Updating event ID {0:d} ({1:d}/{2:d})'.format(
            int(1.e-6 + evID[ne]), ne + 1, evID.size))
        sys.stdout.flush()

    indh = np.nonzero(hyp0[:, 0] == evID[ne])[0][0]
    indrp = np.nonzero(np.logical_and(data[:, 0] == evID[ne], indp))[0]
    indrs = np.nonzero(np.logical_and(data[:, 0] == evID[ne], inds))[0]

    hyp_save = hyp0[indh, :].copy()

    nstp = np.sum(indrp.size)
    nsts = np.sum(indrs.size)

    hypp = np.empty((nstp, 5))
    stnp = np.empty((nstp, 3))
    for i in range(nstp):
        hypp[i, :] = hyp0[indh, :]
        stnp[i, :] = rcv[int(1.e-6 + data[indrp[i], 2]), :]
    hyps = np.empty((nsts, 5))
    stns = np.empty((nsts, 3))
    for i in range(nsts):
        hyps[i, :] = hyp0[indh, :]
        stns[i, :] = rcv[int(1.e-6 + data[indrs[i], 2]), :]

    if par.hypo_2step:
        if par.verbose:
            print('    Updating latitude & longitude', end='')
            sys.stdout.flush()
        H = np.ones((nstp + nsts, 2))
        for itt in range(par.maxit_hypo):
            for i in range(nstp):
                hypp[i, :] = hyp0[indh, :]

            try:
                tcalcp, raysp = grid_p.raytrace(hypp, stnp, s_p, thread_no,
                                                return_rays=True)
            except RuntimeError as rte:
                if 'going outside grid' in str(rte):
                    print('  Problem while computing P-wave traveltimes, '
                          'resetting and exiting')
                    hyp0[indh, :] = hyp_save
                    return hyp_save, indh
                else:
                    raise rte

            s0p = grid_p.get_s0(hypp)
            for i in range(nsts):
                hyps[i, :] = hyp0[indh, :]

            try:
                tcalcs, rayss = grid_s.raytrace(hyps, stns, s_s, thread_no,
                                                return_rays=True)
            except RuntimeError as rte:
                if 'going outside grid' in str(rte):
                    print('  Problem while computing S-wave traveltimes, '
                          'resetting and exiting')
                    hyp0[indh, :] = hyp_save
                    return hyp_save, indh
                else:
                    raise rte

            s0s = grid_s.get_s0(hyps)
            for ns in range(nstp):
                raysi = raysp[ns]
                S0 = s0p[ns]

                d = (raysi[1, :] - hyp0[indh, 2:]).flatten()
                ds = np.sqrt(np.sum(d * d))
                H[ns, 0] = -S0 * d[0] / ds
                H[ns, 1] = -S0 * d[1] / ds
            for ns in range(nsts):
                raysi = rayss[ns]
                S0 = s0s[ns]

                d = (raysi[1, :] - hyp0[indh, 2:]).flatten()
                ds = np.sqrt(np.sum(d * d))
                H[ns + nstp, 0] = -S0 * d[0] / ds
                H[ns + nstp, 1] = -S0 * d[1] / ds

            r = np.hstack((tobs[indrp] - tcalcp, tobs[indrs] - tcalcs))

            x = lstsq(H, r)
            deltah = x[0]

            if np.sum(np.isfinite(deltah)) != deltah.size:
                try:
                    U, S, VVh = np.linalg.svd(H.T.dot(H) + 1e-9 * np.eye(2))
                    VV = VVh.T
                    deltah = np.dot(VV, np.dot(U.T, H.T.dot(r)) / S)
                except np.linalg.linalg.LinAlgError:
                    print(' - Event could not be relocated, '
                          'resetting and exiting')
                    hyp0[indh, :] = hyp_save
                    return hyp_save, indh

            for n in range(2):
                if np.abs(deltah[n]) > par.dx_max:
                    deltah[n] = par.dx_max * np.sign(deltah[n])

            new_hyp = hyp0[indh, :].copy()
            new_hyp[2:4] += deltah
            if grid_p.is_outside(new_hyp[2:5].reshape((1, 3))):
                print('  Event could not be relocated inside the grid '
                      '({0:f}, {1:f}, {2:f}), resetting and exiting'
                      .format(new_hyp[2], new_hyp[3], new_hyp[4]))
                hyp0[indh, :] = hyp_save
                return hyp_save, indh

            hyp0[indh, 2:4] += deltah

            if np.sum(np.abs(deltah) < par.conv_hypo) == 2:
                if par.verbose:
                    print(' - converged at iteration ' + str(itt + 1))
                    sys.stdout.flush()
                break

        else:
            if par.verbose:
                print(' - reached max number of iterations')
                sys.stdout.flush()

    if par.verbose:
        print('    Updating all hypocenter params', end='')
        sys.stdout.flush()

    H = np.ones((nstp + nsts, 4))
    for itt in range(par.maxit_hypo):
        if nstp > 0:
            for i in range(nstp):
                hypp[i, :] = hyp0[indh, :]
            tcalcp, raysp = grid_p.raytrace(hypp, stnp, s_p, thread_no,
                                            return_rays=True)
            s0p = grid_p.get_s0(hypp)
        else:
            tcalcp = np.array([])
            raysp = []
        if nsts > 0:
            for i in range(nsts):
                hyps[i, :] = hyp0[indh, :]
            tcalcs, rayss = grid_s.raytrace(hyps, stns, s_s, thread_no,
                                            return_rays=True)
            s0s = grid_s.get_s0(hyps)
        else:
            tcalcs = np.array([])
            rayss = []

        for ns in range(nstp):
            raysi = raysp[ns]
            S0 = s0p[ns]

            d = (raysi[1, :] - hyp0[indh, 2:]).flatten()
            ds = np.sqrt(np.sum(d * d))
            H[ns, 1] = -S0 * d[0] / ds
            H[ns, 2] = -S0 * d[1] / ds
            H[ns, 3] = -S0 * d[2] / ds
        for ns in range(nsts):
            raysi = rayss[ns]
            S0 = s0s[ns]

            d = (raysi[1, :] - hyp0[indh, 2:]).flatten()
            ds = np.sqrt(np.sum(d * d))
            H[ns + nstp, 1] = -S0 * d[0] / ds
            H[ns + nstp, 2] = -S0 * d[1] / ds
            H[ns + nstp, 3] = -S0 * d[2] / ds

        r = np.hstack((tobs[indrp] - tcalcp, tobs[indrs] - tcalcs))
        x = lstsq(H, r)
        deltah = x[0]

        if np.sum(np.isfinite(deltah)) != deltah.size:
            try:
                U, S, VVh = np.linalg.svd(H.T.dot(H) + 1e-9 * np.eye(4))
                VV = VVh.T
                deltah = np.dot(VV, np.dot(U.T, H.T.dot(r)) / S)
            except np.linalg.linalg.LinAlgError:
                print('  Event could not be relocated, resetting and exiting')
                hyp0[indh, :] = hyp_save
                return hyp_save, indh

        if np.abs(deltah[0]) > par.dt_max:
            deltah[0] = par.dt_max * np.sign(deltah[0])
        for n in range(1, 4):
            if np.abs(deltah[n]) > par.dx_max:
                deltah[n] = par.dx_max * np.sign(deltah[n])

        new_hyp = hyp0[indh, 1:] + deltah
        if grid_p.is_outside(new_hyp[1:].reshape((1, 3))):
            print('  Event could not be relocated inside the grid '
                  '({0:f}, {1:f}, {2:f}), resetting and exiting'
                  .format(new_hyp[1], new_hyp[2], new_hyp[3]))
            hyp0[indh, :] = hyp_save
            return hyp_save, indh

        hyp0[indh, 1:] += deltah

        if np.sum(np.abs(deltah[1:]) < par.conv_hypo) == 3:
            if par.verbose:
                print(' - converged at iteration ' + str(itt + 1))
                sys.stdout.flush()
            break

    else:
        if par.verbose:
            print(' - reached max number of iterations')
            sys.stdout.flush()

    if par.save_rp and 'vtk' in sys.modules and par._final_iteration:
        if par.verbose:
            print('    Saving raypaths')
        filename = 'raypaths'
        key = 'P_ev_{0:d}'.format(int(1.e-6 + evID[ne]))
        grid_p.to_vtk({key: raysp}, filename)
        key = 'S_ev_{0:d}'.format(int(1.e-6 + evID[ne]))
        grid_s.to_vtk({key: rayss}, filename)

    return hyp0[indh, :], indh


# %% main
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

    nthreads = 1
    g1 = Grid3d(x, y, z, nthreads, cell_slowness=False)

    g2 = Grid3d(x, y, z, nthreads, cell_slowness=True)

    testK = True
    testParallel = True
    addNoise = True
    testC = True
    testCp = True
    testCps = True

    if testK:

        g = g2

        Kx, Ky, Kz = g.compute_K()

        plt.figure(figsize=(9, 3))
        plt.subplot(131)
        plt.imshow(Kx.toarray())
        plt.subplot(132)
        plt.imshow(Ky.toarray())
        plt.subplot(133)
        plt.imshow(Kz.toarray())
        plt.show(block=False)

        print(g.shape, Kx.shape, Ky.shape, Kz.shape)

        V = np.ones(g.shape)
        V[5:9, 5:10, 3:8] = 2.

        plt.figure(figsize=(8, 6))
        plt.subplot(221)
        plt.pcolor(x, z, np.squeeze(V[:, 8, :].T),
                   cmap='CMRmap',), plt.gca().invert_yaxis()
        plt.grid()
        plt.xlabel('X')
        plt.ylabel('Z')
        plt.colorbar()
        plt.subplot(222)
        plt.pcolor(y, z, np.squeeze(V[7, :, :].T),
                   cmap='CMRmap'), plt.gca().invert_yaxis()
        plt.grid()
        plt.xlabel('Y')
        plt.ylabel('Z')
        plt.colorbar()
        plt.subplot(223)
        plt.pcolor(x, y, np.squeeze(V[:, :, 6].T), cmap='CMRmap')
        plt.grid()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.colorbar()

        plt.show(block=False)

        dVx = np.reshape(Kx.dot(V.flatten()), g.shape)
        dVy = np.reshape(Ky.dot(V.flatten()), g.shape)
        dVz = np.reshape(Kz.dot(V.flatten()), g.shape)

        K = Kx + Ky + Kz
        dV = np.reshape(K.dot(V.flatten()), g.shape)

        plt.figure(figsize=(8, 6))
        plt.subplot(221)
        plt.pcolor(x, z, np.squeeze(
            dVx[:, 8, :].T), cmap='CMRmap',), plt.gca().invert_yaxis()
        plt.xlabel('X')
        plt.ylabel('Z')
        plt.colorbar()
        plt.subplot(222)
        plt.pcolor(y, z, np.squeeze(
            dVx[7, :, :].T), cmap='CMRmap'), plt.gca().invert_yaxis()
        plt.xlabel('Y')
        plt.ylabel('Z')
        plt.colorbar()
        plt.subplot(223)
        plt.pcolor(x, y, np.squeeze(dVx[:, :, 6].T), cmap='CMRmap')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.colorbar()

        plt.show(block=False)

        plt.figure(figsize=(8, 6))
        plt.subplot(221)
        plt.pcolor(x, z, np.squeeze(
            dVy[:, 8, :].T), cmap='CMRmap',), plt.gca().invert_yaxis()
        plt.xlabel('X')
        plt.ylabel('Z')
        plt.colorbar()
        plt.subplot(222)
        plt.pcolor(y, z, np.squeeze(
            dVy[7, :, :].T), cmap='CMRmap'), plt.gca().invert_yaxis()
        plt.xlabel('Y')
        plt.ylabel('Z')
        plt.colorbar()
        plt.subplot(223)
        plt.pcolor(x, y, np.squeeze(dVy[:, :, 6].T), cmap='CMRmap')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.colorbar()

        plt.show(block=False)

        plt.figure(figsize=(8, 6))
        plt.subplot(221)
        plt.pcolor(x, z, np.squeeze(
            dVz[:, 8, :].T), cmap='CMRmap',), plt.gca().invert_yaxis()
        plt.xlabel('X')
        plt.ylabel('Z')
        plt.colorbar()
        plt.subplot(222)
        plt.pcolor(y, z, np.squeeze(
            dVz[7, :, :].T), cmap='CMRmap'), plt.gca().invert_yaxis()
        plt.xlabel('Y')
        plt.ylabel('Z')
        plt.colorbar()
        plt.subplot(223)
        plt.pcolor(x, y, np.squeeze(dVz[:, :, 6].T), cmap='CMRmap')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.colorbar()

        plt.show(block=False)

        plt.figure(figsize=(8, 6))
        plt.subplot(221)
        plt.pcolor(x, z, np.squeeze(
            dV[:, 8, :].T), cmap='CMRmap',), plt.gca().invert_yaxis()
        plt.xlabel('X')
        plt.ylabel('Z')
        plt.colorbar()
        plt.subplot(222)
        plt.pcolor(y, z, np.squeeze(
            dV[7, :, :].T), cmap='CMRmap'), plt.gca().invert_yaxis()
        plt.xlabel('Y')
        plt.ylabel('Z')
        plt.colorbar()
        plt.subplot(223)
        plt.pcolor(x, y, np.squeeze(dV[:, :, 6].T), cmap='CMRmap')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.colorbar()

        plt.show()

    if testParallel:

        g = g1

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
        ircv = np.arange(rcv.shape[0]).reshape(-1, 1)
        nsta = rcv.shape[0]

        nev = 15
        src = np.vstack((np.arange(nev),
                         np.linspace(0., 50., nev) + np.random.randn(nev),
                         0.160 + 0.005 * np.random.randn(nev),
                         0.140 + 0.005 * np.random.randn(nev),
                         0.060 + 0.010 * np.random.randn(nev))).T

        hinit = np.vstack((np.arange(nev),
                           np.linspace(0., 50., nev),
                           0.150 + 0.0001 * np.random.randn(nev),
                           0.150 + 0.0001 * np.random.randn(nev),
                           0.050 + 0.0001 * np.random.randn(nev))).T

        h_true = src.copy()

        def Vz(z):
            return 4.0 + 10. * (z - 0.050)

        def Vz2(z):
            return 4.0 + 7.5 * (z - 0.050)

        Vp = np.kron(Vz(z), np.ones((g.shape[0], g.shape[1], 1)))
        Vpinit = np.kron(Vz2(z), np.ones((g.shape[0], g.shape[1], 1)))
        Vs = 2.1

        slowness = 1. / Vp.flatten()

        src = np.kron(src, np.ones((nsta, 1)))
        rcv_data = np.kron(np.ones((nev, 1)), rcv)
        ircv_data = np.kron(np.ones((nev, 1)), ircv)

        tt = g.raytrace(src, rcv_data, slowness)

        Vpts = np.array([[Vz(0.001), 0.100, 0.100, 0.001, 0.0],
                         [Vz(0.001), 0.100, 0.200, 0.001, 0.0],
                         [Vz(0.001), 0.200, 0.100, 0.001, 0.0],
                         [Vz(0.001), 0.200, 0.200, 0.001, 0.0],
                         [Vz(0.011), 0.112, 0.148, 0.011, 0.0],
                         [Vz(0.005), 0.152, 0.108, 0.005, 0.0],
                         [Vz(0.075), 0.152, 0.108, 0.075, 0.0],
                         [Vz(0.011), 0.192, 0.148, 0.011, 0.0]])

        Vinit = np.mean(Vpts[:, 0])
        Vpinit = Vpinit.flatten()

        ncal = 5
        src_cal = np.vstack((5 + np.arange(ncal),
                             np.zeros(ncal),
                             0.160 + 0.005 * np.random.randn(ncal),
                             0.130 + 0.005 * np.random.randn(ncal),
                             0.045 + 0.001 * np.random.randn(ncal))).T

        src_cal = np.kron(src_cal, np.ones((nsta, 1)))
        rcv_cal = np.kron(np.ones((ncal, 1)), rcv)
        ircv_cal = np.kron(np.ones((ncal, 1)), ircv)

        ind = np.ones(rcv_cal.shape[0], dtype=bool)
        ind[3] = 0
        ind[13] = 0
        ind[15] = 0
        src_cal = src_cal[ind, :]
        rcv_cal = rcv_cal[ind, :]
        ircv_cal = ircv_cal[ind, :]

        tcal = g.raytrace(src_cal, rcv_cal, slowness)
        caldata = np.column_stack((src_cal[:, 0], tcal, ircv_cal,
                                   src_cal[:, 2:],
                                   np.zeros(tcal.shape)))

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
            noise_variance = 1.e-3  # 1 ms
        else:
            noise_variance = 0.0

        par = InvParams(
            maxit=3,
            maxit_hypo=10,
            conv_hypo=0.001,
            Vlim=Vlim,
            dmax=dmax,
            lagrangians=lagran,
            invert_vel=True,
            verbose=True)

        g = g1

        tt = g.raytrace(src, rcv_data, slowness)
        tt, rays = g.raytrace(src, rcv_data, slowness, return_rays=True)
        tt, rays, M = g.raytrace(src, rcv_data, slowness, return_rays=True,
                                 compute_M=True)

        print('done')


# %% testC
    if testC:

        g = g2
        xx = x[1:] - dx / 2
        yy = y[1:] - dx / 2
        zz = z[1:] - dx / 2

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
        ircv = np.arange(rcv.shape[0]).reshape(-1, 1)
        nsta = rcv.shape[0]

        nev = 15
        src = np.vstack((np.arange(nev),
                         np.linspace(0., 50., nev) + np.random.randn(nev),
                         0.160 + 0.005 * np.random.randn(nev),
                         0.140 + 0.005 * np.random.randn(nev),
                         0.060 + 0.010 * np.random.randn(nev))).T

        hinit = np.vstack((np.arange(nev),
                           np.linspace(0., 50., nev),
                           0.150 + 0.0001 * np.random.randn(nev),
                           0.150 + 0.0001 * np.random.randn(nev),
                           0.050 + 0.0001 * np.random.randn(nev))).T

        h_true = src.copy()

        def Vz(z):
            return 4.0 + 10. * (z - 0.050)

        def Vz2(z):
            return 4.0 + 7.5 * (z - 0.050)

        Vp = np.kron(Vz(zz), np.ones((g.shape[0], g.shape[1], 1)))
        Vpinit = np.kron(Vz2(zz), np.ones((g.shape[0], g.shape[1], 1)))
        Vs = 2.1

        slowness = 1. / Vp.flatten()

        src = np.kron(src, np.ones((nsta, 1)))
        rcv_data = np.kron(np.ones((nev, 1)), rcv)
        ircv_data = np.kron(np.ones((nev, 1)), ircv)

        tt = g.raytrace(src, rcv_data, slowness)

        Vpts = np.array([[Vz(0.001), 0.100, 0.100, 0.001, 0.0],
                         [Vz(0.001), 0.100, 0.200, 0.001, 0.0],
                         [Vz(0.001), 0.200, 0.100, 0.001, 0.0],
                         [Vz(0.001), 0.200, 0.200, 0.001, 0.0],
                         [Vz(0.011), 0.112, 0.148, 0.011, 0.0],
                         [Vz(0.005), 0.152, 0.108, 0.005, 0.0],
                         [Vz(0.075), 0.152, 0.108, 0.075, 0.0],
                         [Vz(0.011), 0.192, 0.148, 0.011, 0.0]])

        Vinit = np.mean(Vpts[:, 0])
        Vpinit = Vpinit.flatten()

        ncal = 5
        src_cal = np.vstack((5 + np.arange(ncal),
                             np.zeros(ncal),
                             0.160 + 0.005 * np.random.randn(ncal),
                             0.130 + 0.005 * np.random.randn(ncal),
                             0.045 + 0.001 * np.random.randn(ncal))).T

        src_cal = np.kron(src_cal, np.ones((nsta, 1)))
        rcv_cal = np.kron(np.ones((ncal, 1)), rcv)
        ircv_cal = np.kron(np.ones((ncal, 1)), ircv)

        ind = np.ones(rcv_cal.shape[0], dtype=bool)
        ind[3] = 0
        ind[13] = 0
        ind[15] = 0
        src_cal = src_cal[ind, :]
        rcv_cal = rcv_cal[ind, :]
        ircv_cal = ircv_cal[ind, :]

        tcal = g.raytrace(src_cal, rcv_cal, slowness)
        caldata = np.column_stack((src_cal[:, 0], tcal, ircv_cal,
                                   src_cal[:, 2:], np.zeros(tcal.shape)))

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
            noise_variance = 1.e-3  # 1 ms
        else:
            noise_variance = 0.0

        par = InvParams(maxit=3, maxit_hypo=10, conv_hypo=0.001, Vlim=Vlim,
                        dmax=dmax, lagrangians=lagran, invert_vel=True,
                        verbose=True, save_V=True, show_plots=True)

# %% testCp
    if testCp:

        tt += noise_variance * np.random.randn(tt.size)

        data = np.hstack((src[:, 0].reshape((-1, 1)),
                          tt.reshape((-1, 1)), ircv_data))

        hinit2, res = hypoloc(data, rcv, Vinit, hinit, 10, 0.001, True)

        h, V, sc, res = jointHypoVel(
            par, g, data, rcv, Vpinit, hinit2, caldata=caldata, Vpts=Vpts)

        plt.figure()
        plt.plot(res[0])
        plt.show(block=False)

        err_xc = hinit2[:, 2:5] - h_true[:, 2:5]
        err_xc = np.sqrt(np.sum(err_xc**2, axis=1))
        err_tc = hinit2[:, 1] - h_true[:, 1]

        err_x = h[:, 2:5] - h_true[:, 2:5]
        err_x = np.sqrt(np.sum(err_x**2, axis=1))
        err_t = h[:, 1] - h_true[:, 1]

        plt.figure(figsize=(10, 4))
        plt.subplot(121)
        plt.plot(
            err_x,
            'o',
            label=r'$\|\|\Delta x\|\|$ = {0:6.5f}'.format(
                np.linalg.norm(err_x)))
        plt.plot(
            err_xc,
            'r*',
            label=r'$\|\|\Delta x\|\|$ = {0:6.5f}'.format(
                np.linalg.norm(err_xc)))
        plt.ylabel(r'$\Delta x$')
        plt.xlabel('Event ID')
        plt.legend()
        plt.subplot(122)
        plt.plot(
            np.abs(err_t),
            'o',
            label=r'$\|\|\Delta t\|\|$ = {0:6.5f}'.format(
                np.linalg.norm(err_t)))
        plt.plot(
            np.abs(err_tc),
            'r*',
            label=r'$\|\|\Delta t\|\|$ = {0:6.5f}'.format(
                np.linalg.norm(err_tc)))
        plt.ylabel(r'$\Delta t$')
        plt.xlabel('Event ID')
        plt.legend()

        plt.show(block=False)

        V3d = V.reshape(g.shape)

        plt.figure(figsize=(10, 8))
        plt.subplot(221)
        plt.pcolor(x, z, np.squeeze(V3d[:, 9, :].T),
                   cmap='CMRmap',), plt.gca().invert_yaxis()
        plt.xlabel('X')
        plt.ylabel('Z')
        plt.colorbar()
        plt.subplot(222)
        plt.pcolor(y, z, np.squeeze(V3d[8, :, :].T),
                   cmap='CMRmap'), plt.gca().invert_yaxis()
        plt.xlabel('Y')
        plt.ylabel('Z')
        plt.colorbar()
        plt.subplot(223)
        plt.pcolor(x, y, np.squeeze(V3d[:, :, 4].T), cmap='CMRmap')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.colorbar()

        plt.show(block=False)

        plt.figure()
        plt.plot(sc, 'o')
        plt.xlabel('Station no')
        plt.ylabel('Correction')
        plt.show()

        print('done')

# %% testCps
    if testCps:

        Vpts_s = np.array([[Vs, 0.100, 0.100, 0.001, 1.0],
                           [Vs, 0.100, 0.200, 0.001, 1.0],
                           [Vs, 0.200, 0.100, 0.001, 1.0],
                           [Vs, 0.200, 0.200, 0.001, 1.0],
                           [Vs, 0.112, 0.148, 0.011, 1.0],
                           [Vs, 0.152, 0.108, 0.005, 1.0],
                           [Vs, 0.152, 0.108, 0.075, 1.0],
                           [Vs, 0.192, 0.148, 0.011, 1.0]])

        Vpts = np.vstack((Vpts, Vpts_s))

        slowness_s = 1. / Vs + np.zeros(g.get_number_of_cells())

        tt_s = g.raytrace(src, rcv_data, slowness_s)

        tt += noise_variance * np.random.randn(tt.size)
        tt_s += noise_variance * np.random.randn(tt_s.size)

        # remove some values
        ind_p = np.ones(tt.shape[0], dtype=bool)
        ind_p[np.random.randint(ind_p.size, size=25)] = False
        ind_s = np.ones(tt_s.shape[0], dtype=bool)
        ind_s[np.random.randint(ind_s.size, size=25)] = False

        data_p = np.hstack((src[ind_p, 0].reshape((-1, 1)),
                            tt[ind_p].reshape((-1, 1)),
                            ircv_data[ind_p, :],
                            np.zeros((np.sum(ind_p), 1))))
        data_s = np.hstack((src[ind_s, 0].reshape((-1, 1)),
                            tt_s[ind_s].reshape((-1, 1)),
                            ircv_data[ind_s, :],
                            np.ones((np.sum(ind_s), 1))))

        ind = data_s[:, 0] != 8.

        data = np.vstack((data_p, data_s[ind, :]))

        tcal_s = g.raytrace(src_cal, rcv_cal, slowness_s)
        caldata_s = np.column_stack((src_cal[:, 0],
                                     tcal_s,
                                     ircv_cal,
                                     src_cal[:, 2:],
                                     np.ones(tcal_s.shape)))
        caldata = np.vstack((caldata, caldata_s))

        Vinit = (Vinit, 2.0)

        hinit2, res = hypolocPS(data, rcv, Vinit, hinit, 10, 0.001, True)

        par.save_V = True
        par.save_rp = True
        par.dVs_max = 0.01
        par.invert_VsVp = False
        par.constr_sc = True
        par.show_plots = True
        h, V, sc, res = jointHypoVelPS(par, g, data, rcv, (Vpinit, 2.0),
                                       hinit2, caldata=caldata,
                                       Vpts=Vpts)

        plt.figure()
        plt.plot(res[0])
        plt.show(block=False)

        err_xc = hinit2[:, 2:5] - h_true[:, 2:5]
        err_xc = np.sqrt(np.sum(err_xc**2, axis=1))
        err_tc = hinit2[:, 1] - h_true[:, 1]

        err_x = h[:, 2:5] - h_true[:, 2:5]
        err_x = np.sqrt(np.sum(err_x**2, axis=1))
        err_t = h[:, 1] - h_true[:, 1]

        plt.figure(figsize=(10, 4))
        plt.subplot(121)
        label = (r'jhv - $\|\|\Delta x\|\|$ = {0:6.5f}'
                 .format(np.linalg.norm(err_x)))
        plt.plot(err_x, 'o', label=label)
        label = (r'cst v - $\|\|\Delta x\|\|$ = {0:6.5f}'
                 .format(np.linalg.norm(err_xc)))
        plt.plot(err_xc, 'r*', label=label)
        plt.ylabel(r'$\Delta x$')
        plt.xlabel('Event ID')
        plt.legend()
        plt.subplot(122)
        label = (r'jhv - $\|\|\Delta t\|\|$ = {0:6.5f}'
                 .format(np.linalg.norm(err_t)))
        plt.plot(np.abs(err_t), 'o', label=label)
        plt.plot(
            np.abs(err_tc),
            'r*',
            label=r'cst v - $\|\|\Delta t\|\|$ = {0:6.5f}'.format(
                np.linalg.norm(err_tc)))
        plt.ylabel(r'$\Delta t$')
        plt.xlabel('Event ID')
        plt.tight_layout()
        plt.legend()

        plt.show(block=False)

        V3d = V[0].reshape(g.shape)

        plt.figure(figsize=(10, 8))
        plt.subplot(221)
        plt.pcolor(x, z, np.squeeze(V3d[:, 9, :].T),
                   cmap='CMRmap',), plt.gca().invert_yaxis()
        plt.xlabel('X')
        plt.ylabel('Z')
        plt.colorbar()
        plt.subplot(222)
        plt.pcolor(y, z, np.squeeze(V3d[8, :, :].T),
                   cmap='CMRmap'), plt.gca().invert_yaxis()
        plt.xlabel('Y')
        plt.ylabel('Z')
        plt.colorbar()
        plt.subplot(223)
        plt.pcolor(x, y, np.squeeze(V3d[:, :, 4].T), cmap='CMRmap')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.colorbar()
        plt.suptitle('V_p')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        plt.show(block=False)

        V3d = V[1].reshape(g.shape)

        plt.figure(figsize=(10, 8))
        plt.subplot(221)
        plt.pcolor(x, z, np.squeeze(V3d[:, 9, :].T),
                   cmap='CMRmap',), plt.gca().invert_yaxis()
        plt.xlabel('X')
        plt.ylabel('Z')
        plt.colorbar()
        plt.subplot(222)
        plt.pcolor(y, z, np.squeeze(V3d[8, :, :].T),
                   cmap='CMRmap'), plt.gca().invert_yaxis()
        plt.xlabel('Y')
        plt.ylabel('Z')
        plt.colorbar()
        plt.subplot(223)
        plt.pcolor(x, y, np.squeeze(V3d[:, :, 4].T), cmap='CMRmap')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.colorbar()
        plt.suptitle('V_s')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        plt.show(block=False)

        plt.figure()
        plt.plot(sc[0], 'o', label='P-wave')
        plt.plot(sc[1], 'r*', label='S-wave')
        plt.xlabel('Station no')
        plt.ylabel('Correction')
        plt.legend()
        plt.tight_layout()
        plt.show()
