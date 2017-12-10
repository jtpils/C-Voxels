import sys

from utils.cloud import LasCloud
from utils.timemeasurement import time_func

import pyvoxels
import cvoxels
import cyvoxels

import timeit

def main():
    k = 0.2
    cloud = LasCloud(sys.argv[1])

    print("Cloud with {} points".format(len(cloud)))

    print("Python:", end=' ')
    time_func(pyvoxels.voxelize_cloud, cloud, k)
    print("C:", end=' ')
    time_func(cvoxels.voxelize_cloud, cloud, k)
    print("Cython:", end=' ')
    time_func(cyvoxels.voxelize_cloud, cloud, k)



if __name__ == '__main__':
    main()