# -*- coding: utf-8 -*-
"""Tests for hypo.py.

Run from the root of the repository, so that ``import hypo`` resolves:

    python -m unittest discover -s tests -t .

The joint inversions are exercised on a deliberately small grid.  They are
still the slow part of the suite, and they are here because the things most
worth protecting -- that relocating in parallel gives the same answer as
relocating serially, and that a worker dying is reported rather than waited
on -- cannot be checked any other way.
"""
import ast
import inspect
import multiprocessing as mp
import re
import unittest
import warnings

import numpy as np

import hypo
from ttcrpy.rgrid import Grid3d


def _model(n=8, h=0.015, v0=4.0, grad=6.0, nthreads=1):
    """A small cell-slowness grid with a vertical velocity gradient."""
    x = (np.arange(n + 1) * h).astype(np.float32)
    zc = (x[:-1] + h / 2).astype(np.float32)
    g = Grid3d(x, x.copy(), x.copy(), nthreads, cell_slowness=True,
               method='FSM', dtype=np.float32)
    V = np.kron(v0 + grad * (zc - x[-1] / 2),
                np.ones((g.shape[0], g.shape[1], 1), dtype=np.float32))
    return g, x, (1.0 / V.flatten()).astype(np.float32)


def _receivers(x):
    hi, lo = float(x[-1]) * 0.9, float(x[-1]) * 0.1
    mid = float(x[-1]) * 0.5
    return np.array([[lo, lo, lo], [hi, lo, mid], [lo, hi, mid],
                     [hi, hi, lo], [mid, lo, hi], [mid, hi, hi]],
                    dtype=np.float32)


def _events(x, nev=4, seed=11):
    """Events well inside the model, clear of the half-cell boundary band."""
    rng = np.random.RandomState(seed)
    lo, hi = float(x[-1]) * 0.3, float(x[-1]) * 0.7
    return np.vstack((np.arange(nev),
                      np.linspace(0.0, 5.0, nev),
                      rng.uniform(lo, hi, nev),
                      rng.uniform(lo, hi, nev),
                      rng.uniform(lo, hi, nev))).T


def _synthetic(nthreads=1, nev=4):
    g, x, s = _model(nthreads=nthreads)
    rcv = _receivers(x)
    src = _events(x, nev)
    nsta = rcv.shape[0]
    srck = np.kron(src, np.ones((nsta, 1), dtype=np.float32))
    rcvk = np.kron(np.ones((nev, 1), dtype=np.float32), rcv)
    ircv = np.kron(np.ones((nev, 1), dtype=int),
                   np.arange(nsta, dtype=int).reshape(-1, 1))
    tt = g.raytrace(srck, rcvk, s)
    data = np.hstack((srck[:, 0].reshape(-1, 1), tt.reshape(-1, 1), ircv))
    hinit = np.vstack((np.arange(nev), np.linspace(0.0, 5.0, nev),
                       np.full(nev, float(x[-1]) * 0.5),
                       np.full(nev, float(x[-1]) * 0.5),
                       np.full(nev, float(x[-1]) * 0.5))).T
    return g, x, s, rcv, src, data, hinit


def _params(**kw):
    kw.setdefault('maxit', 2)
    kw.setdefault('maxit_hypo', 4)
    kw.setdefault('conv_hypo', 1e-3)
    kw.setdefault('Vlim', (3.0, 5.0, 1.0, 1.5, 2.5, 1.0))
    kw.setdefault('dmax', (0.1, 0.02, 0.01, 0.1))
    kw.setdefault('lagrangians', (2.0, 1.0, 1.0, 0.1))
    kw.setdefault('invert_vel', True)
    kw.setdefault('verbose', False)
    return hypo.InvParams(**kw)


class TestGaussNewtonStep(unittest.TestCase):
    """The step that six call sites share.

    Each used to carry its own copy, including its own spelling of the numpy
    exception -- which is how the same defect came to exist seven times.
    """

    def _system(self):
        rng = np.random.RandomState(3)
        H = np.hstack((np.ones((12, 1)), rng.randn(12, 3)))
        return H, rng.randn(12)

    def test_agrees_with_a_direct_solve(self):
        H, r = self._system()
        want = np.linalg.solve(H.T.dot(H), H.T.dot(r))
        got = hypo._gauss_newton_step(H, r, use_lstsq=False)
        np.testing.assert_allclose(got, want, rtol=1e-10)

    def test_both_routes_agree_when_well_conditioned(self):
        # they part company only once H is ill conditioned, which is why the
        # callers are left to choose
        H, r = self._system()
        a = hypo._gauss_newton_step(H, r, use_lstsq=False)
        b = hypo._gauss_newton_step(H, r, use_lstsq=True)
        np.testing.assert_allclose(a, b, rtol=1e-8)

    def test_singular_system_falls_back_and_stays_finite(self):
        H = np.ones((10, 3))            # rank 1: normal equations are singular
        r = np.arange(10.0)
        for use_lstsq in (False, True):
            with self.subTest(use_lstsq=use_lstsq):
                dh = hypo._gauss_newton_step(H, r, use_lstsq=use_lstsq)
                self.assertIsNotNone(dh)
                self.assertTrue(np.all(np.isfinite(dh)))

    def test_returns_none_when_nothing_works(self):
        H = np.full((6, 3), np.nan)
        r = np.zeros(6)
        for use_lstsq in (False, True):
            with self.subTest(use_lstsq=use_lstsq):
                self.assertIsNone(
                    hypo._gauss_newton_step(H, r, use_lstsq=use_lstsq))

    def test_step_size_follows_the_number_of_unknowns(self):
        # the regularising identity used to be spelt np.eye(2) or np.eye(4) by
        # hand at each site; it is taken from H now
        rng = np.random.RandomState(5)
        for ncol in (2, 4):
            with self.subTest(ncol=ncol):
                H = rng.randn(9, ncol)
                dh = hypo._gauss_newton_step(H, rng.randn(9), use_lstsq=True)
                self.assertEqual(dh.shape, (ncol,))


class TestHypoloc(unittest.TestCase):
    """Location in a homogeneous medium."""

    def test_recovers_known_events(self):
        g, x, s, rcv, src, data, hinit = _synthetic()
        # constant slowness, so the analytic velocity is exact
        v = 4.0
        s_const = np.full_like(s, 1.0 / v)
        nsta = rcv.shape[0]
        nev = src.shape[0]
        srck = np.kron(src, np.ones((nsta, 1), dtype=np.float32))
        rcvk = np.kron(np.ones((nev, 1), dtype=np.float32), rcv)
        ircv = np.kron(np.ones((nev, 1), dtype=int),
                       np.arange(nsta, dtype=int).reshape(-1, 1))
        tt = g.raytrace(srck, rcvk, s_const)
        d = np.hstack((srck[:, 0].reshape(-1, 1), tt.reshape(-1, 1), ircv))
        loc, res = hypo.hypoloc(d, rcv, v, hinit.copy(), 15, 1e-4,
                                verbose=False)
        err = np.linalg.norm(loc[:, 2:] - src[:, 2:], axis=1)
        self.assertLess(err.max(), 0.5 * 0.015,
                        'events not recovered to within half a cell')

    def test_residual_shape_and_decrease(self):
        g, x, s, rcv, src, data, hinit = _synthetic()
        loc, res = hypo.hypoloc(data, rcv, 4.0, hinit.copy(), 8, 1e-4,
                                verbose=False)
        self.assertEqual(res.shape, (src.shape[0], 8 + 1))
        for n in range(res.shape[0]):
            row = res[n][res[n] > 0]
            self.assertGreater(row.size, 0)
            self.assertLessEqual(row[-1], row[0] + 1e-12,
                                 'residuals should not grow')


class TestHypolocPS(unittest.TestCase):

    def test_runs_and_reports_residuals(self):
        g, x, s, rcv, src, data, hinit = _synthetic()
        ttS = g.raytrace(np.kron(src, np.ones((rcv.shape[0], 1), dtype=np.float32)),
                         np.kron(np.ones((src.shape[0], 1), dtype=np.float32), rcv),
                         (s * 2.0).astype(np.float32))
        P = np.hstack((data, np.zeros((data.shape[0], 1))))
        S = np.hstack((data[:, :1], ttS.reshape(-1, 1), data[:, 2:3],
                       np.ones((ttS.size, 1))))
        both = np.vstack((P, S))
        both = both[np.lexsort((both[:, 2], both[:, 0]))]
        loc, res = hypo.hypolocPS(both, rcv, np.array([4.0, 2.0]),
                                  hinit.copy(), 10, 1e-4, verbose=False)
        self.assertEqual(loc.shape, hinit.shape)
        self.assertEqual(res.shape, (src.shape[0], 10 + 1))
        self.assertTrue(np.all(np.isfinite(loc)))


class TestResiduals(unittest.TestCase):

    def test_is_still_a_tuple(self):
        r = hypo.Residuals(np.zeros(3), np.zeros(2), np.zeros((2, 4, 5)))
        self.assertEqual(len(r), 3)
        self.assertIs(r[0], r.velocity)
        self.assertIs(r[1], r.system)
        self.assertIs(r[2], r.hypocenter)
        a, b, c = r                      # three names, not two
        self.assertIs(c, r.hypocenter)


class TestJointHypoVel(unittest.TestCase):

    def test_returns_residuals_of_the_documented_shape(self):
        g, x, s, rcv, src, data, hinit = _synthetic()
        par = _params()
        res = hypo.jointHypoVel(par, g, data, rcv, 1.0 / s, hinit.copy())[3]
        self.assertEqual(res.velocity.shape, (par.maxit + 1,))
        self.assertEqual(res.system.shape, (par.maxit,))
        self.assertEqual(res.hypocenter.shape,
                         (par.maxit, src.shape[0], par.maxit_hypo))

    def test_relocation_residuals_are_recorded(self):
        g, x, s, rcv, src, data, hinit = _synthetic()
        par = _params()
        res = hypo.jointHypoVel(par, g, data, rcv, 1.0 / s, hinit.copy())[3]
        rl = res.hypocenter
        self.assertTrue(np.all(rl >= 0.0))
        self.assertTrue(np.any(rl > 0.0), 'no relocation residual recorded')
        # every event contributes something on the first iteration
        self.assertTrue(np.all((rl[0] > 0).sum(axis=1) >= 1))


class TestSerialAndParallelAgree(unittest.TestCase):
    """Relocating on several processes must not change the answer.

    The workers are separate processes, so anything they are not sent and
    anything they do not send back is lost.  That is how they once ran against
    a grid whose slowness had not survived being pickled, and how the
    relocation residuals could have been recorded into an array the parent
    never sees.
    """

    def test_same_hypocenters_and_residuals(self):
        nev = 6
        out = {}
        for nthreads in (1, 4):
            g, x, s, rcv, src, data, hinit = _synthetic(nthreads, nev)
            par = _params()
            h, V, sc, res = hypo.jointHypoVel(par, g, data, rcv, 1.0 / s,
                                              hinit.copy())
            out[nthreads] = (h, V, res.velocity, res.system, res.hypocenter)
        for name, a, b in zip(('hypocenters', 'velocity', 'resV', 'resAxb',
                               'resLoc'), out[1], out[4]):
            with self.subTest(array=name):
                np.testing.assert_array_equal(a, b)


def _worker_that_dies(q):
    raise RuntimeError('deliberate failure')


def _worker_that_answers(q):
    q.put((np.zeros(5), 0, 0, np.zeros(3)))


class TestWorkerFailureIsReported(unittest.TestCase):
    """A worker that dies used to leave the parent waiting on the queue."""

    def test_raises_instead_of_waiting(self):
        ctx = mp.get_context('spawn')
        q = ctx.Queue()
        procs = [ctx.Process(target=_worker_that_answers, args=(q,)),
                 ctx.Process(target=_worker_that_dies, args=(q,))]
        for p in procs:
            p.start()
        hyp0 = np.zeros((1, 5))
        resLoc = np.zeros((1, 2, 3))
        with self.assertRaises(RuntimeError) as cm:
            hypo._collect_relocations(procs, q, 2, hyp0, resLoc, 0)
        self.assertIn('ended without returning results', str(cm.exception))
        for p in procs:
            self.assertFalse(p.is_alive(), 'workers should have been reaped')


class TestVtkReporting(unittest.TestCase):

    def test_warns_when_asked_to_save_without_vtk(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            hypo._warn_no_vtk('save_V')
        self.assertEqual(len(w), 1)
        self.assertIs(w[0].category, RuntimeWarning)
        self.assertIn('save_V', str(w[0].message))


class TestDocstrings(unittest.TestCase):
    """Every parameter is named in the docstring, and says something.

    maxit_hypo was listed with nothing after the colon, which a check for the
    name alone counts as documented -- so the check looks at the description
    too.
    """

    def _entries(self):
        tree = ast.parse(inspect.getsource(hypo))
        for n in tree.body:
            if isinstance(n, ast.FunctionDef) and not n.name.startswith('_'):
                yield n.name, [a.arg for a in n.args.args], ast.get_docstring(n)
            elif isinstance(n, ast.ClassDef):
                init = next((f for f in n.body
                             if isinstance(f, ast.FunctionDef)
                             and f.name == '__init__'), None)
                if init is not None:
                    yield (n.name, [a.arg for a in init.args.args][1:],
                           ast.get_docstring(init))

    def test_every_parameter_is_described(self):
        for name, args, doc in self._entries():
            lines = (doc or '').splitlines()
            for a in args:
                with self.subTest(where=name, param=a):
                    pat = re.compile(r'^\s*%s\s*:(.*)$' % re.escape(a))
                    hit = next(((i, m.group(1).strip())
                                for i, l in enumerate(lines)
                                for m in [pat.match(l)] if m), None)
                    self.assertIsNotNone(hit, '%s: %s is not documented'
                                         % (name, a))
                    i, tail = hit
                    if not tail:
                        nxt = lines[i+1].strip() if i + 1 < len(lines) else ''
                        self.assertTrue(
                            nxt and not re.match(r'^\w+\s*:', nxt),
                            '%s: %s is listed with no description' % (name, a))


if __name__ == '__main__':
    unittest.main()
