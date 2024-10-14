from time import time

from indicators.tests import IndcTester
from indicators.vol_distribution import VolDistribution
from indicators.zigzag import ZigZag

if __name__ == "__main__":
    tester = IndcTester(VolDistribution())
        
    t0 = time()
    for i, t in enumerate(range(2000, 3000)):
        tester.test_hvol(t)
    print((time()-t0)/(i+1))