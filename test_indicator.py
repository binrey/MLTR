from time import time

from indicators.testing import IndicatorTester
from indicators.vol_distribution import VolDistribution
from indicators.zigzag import ZigZag

if __name__ == "__main__":
    indicator = VolDistribution()
    tester = IndicatorTester(indicator)
        
    t0 = time()
    for i, t in enumerate(range(2000, 3000)):
        tester.sigle_run(t)
    print((time()-t0)/(i+1))