from .context import SLRstats
import numpy as np

def test_create_distribution():
    Y = np.array([1.5, 1.0, 0.5])
    delta = 1.0
    D = SLRstats.Distribution(Y, delta, 0.0, 0.0)
    assert D.tmin == 0.0
    assert D.tmax == 3.0
    assert D.L == 3.0


def test_cdf():
    Y = np.array([1.0])
    delta = 1.0
    D = SLRstats.Distribution(Y, delta, 0.0, 0.1)
    print(D.I)
    print(D.E)
    
    assert D.L == 1.0
    assert D.cdf(0.0) == 0.0
