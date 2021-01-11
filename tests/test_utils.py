# -*- coding: utf-8 -*-
# Copyright 2020 The PsiZ Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Module for testing utils.py.

Todo:
    - test compare_models

"""

import numpy as np
from psiz import utils
import psiz.models


def test_pairwise_matrix(rank_2g_mle_determ):
    """Test similarity matrix."""
    actual_simmat1 = np.array((
        (1., 0.35035481, 0.00776613),
        (0.35035481, 1., 0.0216217),
        (0.00776613, 0.0216217, 1.)
    ))
    actual_simmat2 = np.array((
        (1., 0.29685964, 0.00548485),
        (0.29685964, 1., 0.01814493),
        (0.00548485, 0.01814493, 1.)
    ))

    proxy = psiz.models.Proxy(model=rank_2g_mle_determ)

    computed_simmat0 = utils.pairwise_matrix(
        proxy.similarity, proxy.z[0])

    # Check explicit use of first set of attention weights.
    def similarity_func1(z_q, z_ref):
        sim_func = proxy.similarity(z_q, z_ref, group_id=0)
        return sim_func

    # Check without passing in explicit attention.
    computed_simmat1 = utils.pairwise_matrix(
        similarity_func1, proxy.z[0])

    # Check explicit use of second set of attention weights.
    def similarity_func2(z_q, z_ref):
        sim_func = proxy.similarity(z_q, z_ref, group_id=1)
        return sim_func

    computed_simmat2 = utils.pairwise_matrix(
        similarity_func2, proxy.z[0])

    np.testing.assert_array_almost_equal(actual_simmat1, computed_simmat0)
    np.testing.assert_array_almost_equal(actual_simmat1, computed_simmat1)
    np.testing.assert_array_almost_equal(actual_simmat2, computed_simmat2)


def test_matrix_comparison():
    """Test matrix correlation."""
    a = np.array((
        (1.0, .50, .90, .13),
        (.50, 1.0, .10, .80),
        (.90, .10, 1.0, .12),
        (.13, .80, .12, 1.0)
    ))

    b = np.array((
        (1.0, .45, .90, .11),
        (.45, 1.0, .20, .82),
        (.90, .20, 1.0, .02),
        (.11, .82, .02, 1.0)
    ))

    r2_score_1 = utils.matrix_comparison(a, b, score='r2')
    np.testing.assert_almost_equal(r2_score_1, 0.96723696)


def test_procrustean_solution():
    """Test procrustean solution for simple problem."""
    # Assemble rotation matrix (with scaling and reflection).
    s = np.array([[-2, 0], [0, 2]])
    r = psiz.utils.rotation_matrix(np.pi/4)
    r = np.matmul(s, r)

    # Assemble translation vector.
    t = np.array([-.8, 1])
    t = np.expand_dims(t, axis=0)

    # Create random set of points.
    x = np.array([
        [0.46472851, 0.09534286],
        [0.90612827, 0.21031482],
        [0.46595517, 0.92022067],
        [0.51457351, 0.88226988],
        [0.24506303, 0.75287697],
        [0.69773745, 0.25095083],
        [0.71550351, 0.14846334],
        [0.24825323, 0.96021703],
        [0.85497989, 0.9114596],
        [0.35982138, 0.85040905]
    ])

    # Apply affine transformation.
    y = np.matmul(x, r) + t

    # Attempt to recover original set of points.
    r_recov, t_recov = utils.procrustes_2d(x, y, n_restart=10, seed=252)
    x_recov = np.matmul(y, r_recov) + t_recov

    np.testing.assert_almost_equal(x, x_recov, decimal=2)


def test_choice_wo_replace():
    """Test choice_wo_replace."""
    n_trial = 10000
    n_reference = 8
    n_option = 20

    candidate_idx = np.arange(n_option)
    candidate_prob = np.array([
        0.04787656, 0.01988875, 0.08106771, 0.08468775, 0.07918673,
        0.05087084, 0.00922816, 0.08663405, 0.00707334, 0.02254985,
        0.01820681, 0.01532338, 0.07702897, 0.06774214, 0.09976408,
        0.05369049, 0.01056261, 0.07500489, 0.05508777, 0.03852514
    ])

    # Draw samples.
    np.random.seed(560897)
    drawn_idx = psiz.utils.choice_wo_replace(
        candidate_idx, (n_trial, n_reference), candidate_prob
    )
    bin_counts, bin_edges = np.histogram(drawn_idx.flatten(), bins=n_option)
    drawn_prob = bin_counts / np.sum(bin_counts)

    # Check that sampling was done without replacement for all trials.
    for i_trial in range(n_trial):
        assert len(np.unique(drawn_idx[i_trial])) == n_reference

    # Check that sampling distribution matches original probabilites.
    np.testing.assert_array_almost_equal(candidate_prob, drawn_prob, decimal=2)
