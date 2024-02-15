from unittest import TestCase
from MARLlib.marllib.marl.algos.core.IL.trafficppo import get_emergency_start_end
import numpy as np


class Test(TestCase):
    def test_get_emergency_start_end(self):
        # Test case 1
        emergency_obs_1 = np.array([[0., 0.], [0., 0.], [0.1, 0.1], [0.1, 0.1], [0, 0], [0.2, 0.2], [0.2, 0.2]])
        end_indices, start_indices, separate_results = get_emergency_start_end(emergency_obs_1)
        self.assertEqual(end_indices, [1, 3, 4, 6])
        self.assertEqual(start_indices, [0, 2, 4, 5])
        self.assertEqual(separate_results, [2, 4, 5, 6])

        # Test case 2
        emergency_obs_2 = np.array(
            [[0., 0.], [0., 0.], [0.1, 0.1], [0.1, 0.1], [0, 0], [0.2, 0.2], [0.2, 0.2], [0.3, 0.3]])
        end_indices, start_indices, separate_results = get_emergency_start_end(emergency_obs_2)
        self.assertEqual(end_indices, [1, 3, 4, 6])
        self.assertEqual(start_indices, [0, 2, 4, 5])
        self.assertEqual(separate_results, [2, 4, 5, 6])

        # Test case 3
        emergency_obs_3 = np.array(
            [[0.3, 0.3], [0.4, 0.4], [0.4, 0.4], [0.1, 0.1], [0.1, 0.1], [0.5, 0.5], [0.2, 0.2], [0.2, 0.2], [0.6, 0.6],
             [0.6, 0.6]])
        end_indices, start_indices, separate_results = get_emergency_start_end(emergency_obs_3)
        self.assertEqual(end_indices, [0, 2, 4, 5, 7, 9])
        self.assertEqual(start_indices, [0, 1, 3, 5, 6, 8])

        emergency_obs_4 = np.array([[0.3, 0.3], [0.3, 0.3], [0, 0], [0.4, 0.4], [0.4, 0.4], [0.1, 0.1],
                                    [0.1, 0.1], [0.5, 0.5], [0.2, 0.2], [0.2, 0.2], [0.6, 0.6], [0.6, 0.6]])
        end_indices, start_indices, separate_results = get_emergency_start_end(emergency_obs_4)
        self.assertEqual(end_indices, [1, 4, 6, 7, 9, 11])
        self.assertEqual(start_indices, [0, 3, 5, 7, 8, 10])

        emergency_obs_5 = np.array([[0.3, 0], [0.3, 0], [0, 0], [0, 0.4], [0, 0.4], [0, 0.1],
                                    [0, 0.1], [0.5, 0.5], [0.2, 0.2], [0.2, 0.2], [0.6, 0.6], [0.6, 0.6]])
        end_indices, start_indices, separate_results = get_emergency_start_end(emergency_obs_5)
        self.assertEqual(end_indices, [1, 4, 6, 7, 9, 11])
        self.assertEqual(start_indices, [0, 3, 5, 7, 8, 10])

        emergency_obs_6 = np.array([[0.1, 0.1], [0.1, 0.1], [0.2, 0.2, ], [0.2, 0.2, ],
                                    [0.3, 0.3], [0.3, 0.3], [0.4, 0.4], [0.4, 0.4]])
        end_indices, start_indices, separate_results = get_emergency_start_end(emergency_obs_6)
        self.assertEqual(end_indices, [1, 3, 5, 7])
        self.assertEqual(start_indices, [0, 2, 4, 6])
        self.assertEqual(separate_results, [0, 2, 4, 6])

        emergency_obs_6 = np.array([[0, 0], [0.1, 0.1], [0.2, 0.2, ], [0.2, 0.2, ],
                                    [0.3, 0.3], [0.3, 0.3], [0.4, 0.4], [0.4, 0.4]])
        end_indices, start_indices, separate_results = get_emergency_start_end(emergency_obs_6)
        self.assertEqual(end_indices, [1, 3, 5, 7])
        self.assertEqual(start_indices, [1, 2, 4, 6])
        self.assertEqual(separate_results, [1, 2, 4, 6])
