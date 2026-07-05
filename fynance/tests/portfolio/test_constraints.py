""" Tests for feasible-set projection (fynance.portfolio.constraints). """

import numpy as np
import pytest

from fynance.portfolio.constraints import project_weights

# =========================================================================== #
#                                idempotence                                  #
# =========================================================================== #


class TestIdempotence:
    """ An already-feasible weight vector should come back unchanged. """

    def test_fast_path_exact_equality(self):
        """ Fast path (box + gross) leaves a feasible w untouched, bit-exact. """
        w = np.array([0.2, -0.1, 0.05])

        v = project_weights(w, box=(-1.0, 1.0), gross_max=1.0)

        assert np.array_equal(v, w)

    def test_exact_path_atol(self):
        """ Exact (SLSQP) path leaves a feasible w unchanged within tolerance. """
        w = np.array([0.2, -0.1, 0.05])

        v = project_weights(w, box=(-1.0, 1.0), gross_max=1.0, method='exact')

        assert np.allclose(v, w, atol=1e-8)


# =========================================================================== #
#                          feasibility of the output                         #
# =========================================================================== #


class TestFeasibility:
    """ Random long-short books: output must respect every active constraint. """

    def setup_method(self):
        rng = np.random.default_rng(1)
        self.T, self.N = 50, 8
        self.W = rng.uniform(-2.0, 2.0, size=(self.T, self.N))
        self.box = (-1.0, 1.0)
        self.gross_max = 3.0
        self.net_range = (-0.5, 0.5)
        self.groups = {'a': [0, 1, 2, 3], 'b': [4, 5, 6, 7]}
        self.group_bounds = {'a': (-0.5, 0.5), 'b': (-0.5, 0.5)}

    def _assert_box(self, v):
        lo, hi = self.box
        assert np.all(v >= lo - 1e-8)
        assert np.all(v <= hi + 1e-8)

    def test_box_alone(self):
        v = project_weights(self.W, box=self.box)
        self._assert_box(v)

    def test_gross_alone(self):
        v = project_weights(self.W, box=self.box, gross_max=self.gross_max)
        self._assert_box(v)
        assert np.all(np.sum(np.abs(v), axis=1) <= self.gross_max + 1e-8)

    def test_net_alone(self):
        v = project_weights(self.W, box=self.box, net_range=self.net_range)
        self._assert_box(v)
        net = v.sum(axis=1)
        assert np.all(net >= self.net_range[0] - 1e-8)
        assert np.all(net <= self.net_range[1] + 1e-8)

    def test_groups_alone(self):
        v = project_weights(self.W, box=self.box, groups=self.groups, group_bounds=self.group_bounds)
        self._assert_box(v)
        for name, idx in self.groups.items():
            g_lo, g_hi = self.group_bounds[name]
            g_net = v[:, idx].sum(axis=1)
            assert np.all(g_net >= g_lo - 1e-8)
            assert np.all(g_net <= g_hi + 1e-8)

    def test_all_combined(self):
        v = project_weights(
            self.W,
            box=self.box,
            gross_max=self.gross_max,
            net_range=self.net_range,
            groups=self.groups,
            group_bounds=self.group_bounds,
        )
        self._assert_box(v)
        assert np.all(np.sum(np.abs(v), axis=1) <= self.gross_max + 1e-8)
        net = v.sum(axis=1)
        assert np.all(net >= self.net_range[0] - 1e-8)
        assert np.all(net <= self.net_range[1] + 1e-8)
        for name, idx in self.groups.items():
            g_lo, g_hi = self.group_bounds[name]
            g_net = v[:, idx].sum(axis=1)
            assert np.all(g_net >= g_lo - 1e-8)
            assert np.all(g_net <= g_hi + 1e-8)


# =========================================================================== #
#                          least-distance closed forms                       #
# =========================================================================== #


class TestClosedForms:
    """ method='exact' should reproduce known least-distance projections. """

    def test_box_only_equals_clip(self):
        rng = np.random.default_rng(2)
        w = rng.uniform(-3.0, 3.0, size=(20, 6))

        v = project_weights(w, box=(-1.0, 1.0), method='exact')

        assert np.allclose(v, np.clip(w, -1.0, 1.0), atol=1e-6)

    def test_net_range_projection(self):
        w = np.array([1.0, 1.0])

        v = project_weights(w, box=(-10.0, 10.0), net_range=(0.0, 1.0))

        assert np.allclose(v, [0.5, 0.5], atol=1e-6)


# =========================================================================== #
#                          fast path == exact path                           #
# =========================================================================== #


class TestFastEqualsExact:
    """ When only the box binds, the fast path must match the SLSQP path. """

    def test_fast_equals_exact(self):
        rng = np.random.default_rng(3)
        w = rng.uniform(-2.0, 2.0, size=(30, 5))
        box = (-1.0, 1.0)
        gross_max = 100.0  # large enough to never bind

        v_fast = project_weights(w, box=box, gross_max=gross_max, method='auto')
        v_exact = project_weights(w, box=box, gross_max=gross_max, method='exact')

        assert np.allclose(v_fast, v_exact, atol=1e-8)


# =========================================================================== #
#                              row-wise (T, N)                                #
# =========================================================================== #


class TestRowWise:
    """ (T, N) input must equal stacking independent (N,) calls. """

    def test_rowwise_matches_per_row_calls(self):
        rng = np.random.default_rng(4)
        W = rng.uniform(-2.0, 2.0, size=(10, 4))
        box = (-0.5, 0.5)
        gross_max = 1.5

        v_batch = project_weights(W, box=box, gross_max=gross_max)
        v_rows = np.stack(
            [project_weights(W[t], box=box, gross_max=gross_max) for t in range(W.shape[0])]
        )

        assert np.array_equal(v_batch, v_rows)


# =========================================================================== #
#                                   errors                                    #
# =========================================================================== #


class TestErrors:
    """ Error handling: bad group keys, infeasible sets, malformed box. """

    def test_unknown_group_bounds_key(self):
        w = np.zeros(3)

        with pytest.raises(ValueError, match="group_bounds"):
            project_weights(w, groups={'a': [0, 1]}, group_bounds={'b': (0.0, 1.0)})

    def test_infeasible_box_net_combo(self):
        w = np.zeros(10)

        with pytest.raises(ValueError, match="infeasible"):
            project_weights(w, box=(0.1, 1.0), net_range=(0.0, 0.5))

    def test_box_lo_greater_than_hi(self):
        w = np.zeros(3)

        with pytest.raises(ValueError):
            project_weights(w, box=(1.0, -1.0))
