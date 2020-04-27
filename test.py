from shimmer4py import get_kurtosis, get_sd, get_mean
import unittest
import numpy as np
from scipy.stats import kurtosis as kt

class TestShimmerMethods(unittest.TestCase):

    def setUp(self):
        self.data = np.random.poisson(0.9, (10,))

        
    def test_sd(self):
        self.assertAlmostEqual(
            np.std(self.data, axis=0), get_sd(self.data, self.data.shape[0]),
            msg="STD not within bounds",
            delta=1e-6
        )

    def test_mean(self):
        self.assertAlmostEqual(
            np.mean(self.data, axis=0), get_mean(self.data),
            msg="Mean not within bounds",
            delta=1e-6
        )

    def test_kt(self):
        self.assertAlmostEqual(
            kt(self.data, fisher=False), get_kurtosis(self.data),
            msg="Kurtosis not within bounds",
            delta=1e-6
        )

if __name__=="__main__":
    unittest.main()
