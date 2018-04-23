# -*- coding: utf-8 -*-
# Copyright 2018 The PsiZ Authors. All Rights Reserved.
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
# ==============================================================================

"""Module for testing trials.py.

Todo:
    - write test for subset and test config_id, it is important that
    the config_id and configuration is recomputed on initialization
    because down-stream code assumes that config_id is from [0,N[ and
    corresponds to indices of config_list
"""

import pytest
import numpy as np
import pandas as pd

from psiz.trials import UnjudgedTrials, JudgedTrials


@pytest.fixture(scope="module")
def setup_tasks_0():
    """
    """
    stimulus_set = np.array(((0, 1, 2, -1, -1, -1, -1, -1, -1),
                            (9, 12, 7, -1, -1, -1, -1, -1, -1),
                            (3, 4, 5, 6, 7, -1, -1, -1, -1),
                            (3, 4, 5, 6, 13, 14, 15, 16, 17)))
    n_trial = 4
    n_selected = np.array((1, 1, 1, 1))
    n_reference = np.array((2, 2, 4, 8))
    is_ranked = np.array((True, True, True, True))

    configurations = pd.DataFrame(
        {
            'n_reference': [2, 4, 8],
            'n_selected': [1, 1, 1],
            'is_ranked': [True, True, True]
        },
        index=[0, 2, 3])
    configuration_id = np.array((0, 0, 1, 2))

    tasks = UnjudgedTrials(stimulus_set)
    return {
        'n_trial': n_trial, 'stimulus_set': stimulus_set,
        'n_reference': n_reference, 'n_selected': n_selected,
        'is_ranked': is_ranked, 'tasks': tasks,
        'configurations': configurations,
        'configuration_id': configuration_id
        }


@pytest.fixture(scope="module")
def setup_tasks_1():
    """
    """
    stimulus_set = np.array(((0, 1, 2, -1, -1, -1, -1, -1, -1),
                            (9, 12, 7, -1, -1, -1, -1, -1, -1),
                            (3, 4, 5, 6, 7, -1, -1, -1, -1),
                            (3, 4, 5, 6, 13, 14, 15, 16, 17)))
    n_trial = 4
    n_selected = np.array((1, 1, 1, 2))
    n_reference = np.array((2, 2, 4, 8))
    is_ranked = np.array((True, True, True, True))

    configurations = pd.DataFrame(
        {
            'n_reference': [2, 4, 8],
            'n_selected': [1, 1, 2],
            'is_ranked': [True, True, True]
        },
        index=[0, 2, 3])
    configuration_id = np.array((0, 0, 1, 2))

    tasks = UnjudgedTrials(stimulus_set, n_selected=n_selected)
    return {
        'n_trial': n_trial, 'stimulus_set': stimulus_set,
        'n_reference': n_reference, 'n_selected': n_selected,
        'is_ranked': is_ranked, 'tasks': tasks,
        'configurations': configurations,
        'configuration_id': configuration_id
        }


@pytest.fixture(scope="module")
def setup_obs_0():
    """
    """
    stimulus_set = np.array(((0, 1, 2, -1, -1, -1, -1, -1, -1),
                            (9, 12, 7, -1, -1, -1, -1, -1, -1),
                            (3, 4, 5, 6, 7, -1, -1, -1, -1),
                            (3, 4, 5, 6, 13, 14, 15, 16, 17)))
    n_trial = 4
    n_selected = np.array((1, 1, 1, 2))
    n_reference = np.array((2, 2, 4, 8))
    is_ranked = np.array((True, True, True, True))

    configurations = pd.DataFrame(
        {
            'n_reference': [2, 4, 8],
            'n_selected': [1, 1, 2],
            'is_ranked': [True, True, True],
            'group_id': [0, 0, 0],
            'session_id': [0, 0, 0]
        },
        index=[0, 2, 3])
    configuration_id = np.array((0, 0, 1, 2))

    tasks = JudgedTrials(stimulus_set, n_selected=n_selected)
    return {
        'n_trial': n_trial, 'stimulus_set': stimulus_set,
        'n_reference': n_reference, 'n_selected': n_selected,
        'is_ranked': is_ranked, 'tasks': tasks,
        'configurations': configurations,
        'configuration_id': configuration_id
        }


@pytest.fixture(scope="module")
def setup_obs_1():
    """
    """
    stimulus_set = np.array(((0, 1, 2, -1, -1, -1, -1, -1, -1),
                            (9, 12, 7, -1, -1, -1, -1, -1, -1),
                            (3, 4, 5, 6, 7, -1, -1, -1, -1),
                            (3, 4, 5, 6, 13, 14, 15, 16, 17)))
    n_trial = 4
    n_selected = np.array((1, 1, 1, 2))
    n_reference = np.array((2, 2, 4, 8))
    is_ranked = np.array((True, True, True, True))
    group_id = np.array((0, 0, 1, 1))

    configurations = pd.DataFrame(
        {
            'n_reference': [2, 4, 8],
            'n_selected': [1, 1, 2],
            'is_ranked': [True, True, True],
            'group_id': [0, 1, 1],
            'session_id': [0, 0, 0]
        },
        index=[0, 2, 3])
    configuration_id = np.array((0, 0, 1, 2))

    tasks = JudgedTrials(stimulus_set, n_selected=n_selected,
                         group_id=group_id)
    return {
        'n_trial': n_trial, 'stimulus_set': stimulus_set,
        'n_reference': n_reference, 'n_selected': n_selected,
        'is_ranked': is_ranked, 'group_id': group_id, 'tasks': tasks,
        'configurations': configurations,
        'configuration_id': configuration_id
        }


class TestUnjudgedTrials1:

    def test_n_trial(self, setup_tasks_0):
        assert setup_tasks_0['n_trial'] == setup_tasks_0['tasks'].n_trial

    def test_stimulus_set(self, setup_tasks_0):
        np.testing.assert_array_equal(
            setup_tasks_0['stimulus_set'],
            setup_tasks_0['tasks'].stimulus_set)

    def test_n_reference(self, setup_tasks_0):
        np.testing.assert_array_equal(
            setup_tasks_0['n_reference'], setup_tasks_0['tasks'].n_reference)

    def test_n_selected(self, setup_tasks_0):
        np.testing.assert_array_equal(
            setup_tasks_0['n_selected'], setup_tasks_0['tasks'].n_selected)

    def test_is_ranked(self, setup_tasks_0):
        np.testing.assert_array_equal(
            setup_tasks_0['is_ranked'], setup_tasks_0['tasks'].is_ranked)

    def test_configurations(self, setup_tasks_0):
        pd.testing.assert_frame_equal(
            setup_tasks_0['configurations'],
            setup_tasks_0['tasks'].config_list)

    def test_configuration_id(self, setup_tasks_0):
        np.testing.assert_array_equal(
            setup_tasks_0['configuration_id'],
            setup_tasks_0['tasks'].config_id)


class TestUnjudgedTrials2:

    def test_n_trial(self, setup_tasks_1):
        assert setup_tasks_1['n_trial'] == setup_tasks_1['tasks'].n_trial

    def test_stimulus_set(self, setup_tasks_1):
        np.testing.assert_array_equal(
            setup_tasks_1['stimulus_set'], setup_tasks_1['tasks'].stimulus_set)

    def test_n_reference(self, setup_tasks_1):
        np.testing.assert_array_equal(
            setup_tasks_1['n_reference'], setup_tasks_1['tasks'].n_reference)

    def test_n_selected(self, setup_tasks_1):
        np.testing.assert_array_equal(
            setup_tasks_1['n_selected'], setup_tasks_1['tasks'].n_selected)

    def test_is_ranked(self, setup_tasks_1):
        np.testing.assert_array_equal(
            setup_tasks_1['is_ranked'], setup_tasks_1['tasks'].is_ranked)

    def test_configurations(self, setup_tasks_1):
        pd.testing.assert_frame_equal(
            setup_tasks_1['configurations'],
            setup_tasks_1['tasks'].config_list)

    def test_configuration_id(self, setup_tasks_1):
        np.testing.assert_array_equal(
            setup_tasks_1['configuration_id'],
            setup_tasks_1['tasks'].config_id)


class TestJudgedTrials1:

    def test_n_trial(self, setup_obs_0):
        assert setup_obs_0['n_trial'] == setup_obs_0['tasks'].n_trial

    def test_stimulus_set(self, setup_obs_0):
        np.testing.assert_array_equal(
            setup_obs_0['stimulus_set'], setup_obs_0['tasks'].stimulus_set)

    def test_n_reference(self, setup_obs_0):
        np.testing.assert_array_equal(
            setup_obs_0['n_reference'], setup_obs_0['tasks'].n_reference)

    def test_n_selected(self, setup_obs_0):
        np.testing.assert_array_equal(
            setup_obs_0['n_selected'], setup_obs_0['tasks'].n_selected)

    def test_is_ranked(self, setup_obs_0):
        np.testing.assert_array_equal(
            setup_obs_0['is_ranked'], setup_obs_0['tasks'].is_ranked)

    def test_configurations(self, setup_obs_0):
        pd.testing.assert_frame_equal(
            setup_obs_0['configurations'],
            setup_obs_0['tasks'].config_list)

    def test_configuration_id(self, setup_obs_0):
        np.testing.assert_array_equal(
            setup_obs_0['configuration_id'],
            setup_obs_0['tasks'].config_id)


class TestJudgedTrials2:

    def test_n_trial(self, setup_obs_1):
        assert setup_obs_1['n_trial'] == setup_obs_1['tasks'].n_trial

    def test_stimulus_set(self, setup_obs_1):
        np.testing.assert_array_equal(
            setup_obs_1['stimulus_set'], setup_obs_1['tasks'].stimulus_set)

    def test_n_reference(self, setup_obs_1):
        np.testing.assert_array_equal(
            setup_obs_1['n_reference'], setup_obs_1['tasks'].n_reference)

    def test_n_selected(self, setup_obs_1):
        np.testing.assert_array_equal(
            setup_obs_1['n_selected'], setup_obs_1['tasks'].n_selected)

    def test_is_ranked(self, setup_obs_1):
        np.testing.assert_array_equal(
            setup_obs_1['is_ranked'], setup_obs_1['tasks'].is_ranked)

    def test_configurations(self, setup_obs_1):
        pd.testing.assert_frame_equal(
            setup_obs_1['configurations'],
            setup_obs_1['tasks'].config_list)

    def test_configuration_id(self, setup_obs_1):
        np.testing.assert_array_equal(
            setup_obs_1['configuration_id'],
            setup_obs_1['tasks'].config_id)
