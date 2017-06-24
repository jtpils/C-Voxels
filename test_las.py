import laspy
from PLAnalysis.common import pyvoxels, extract
from PLAnalysis.common.cloudtypes import LasCloud

import sys
import logging
import cvoxels
import numpy as np

from datetime import datetime

import time                                                

def timeit(method):

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        return result, (te-ts)

    return timed

@timeit
def exec_func(function, *args, **kwargs):
    return function(*args, **kwargs) 

def time_func(func, *args, **kwargs):
    res, exec_time = exec_func(func, *args, **kwargs)
    print "{} took {} secs".format(func.__name__, exec_time)
    return res

def main():

    cloud = LasCloud(sys.argv[1])
    k = 0.2

    coords = cloud.coords
    classification = cloud.classification
    mins = cloud.bb_min

    filter = [0]
    is_blacklist=False

    c_voxels = voxels = None
    print "Number of points: {}".format(len(classification))
    print ""

    print "Executing C functions:"
    if is_blacklist:
        c_voxels = time_func(cvoxels.voxelize_cloud, cloud, k, class_blacklist=filter)
    else:
        c_voxels = time_func(cvoxels.voxelize_cloud, cloud, k, class_whitelist=filter)

    c_neigh = time_func(cvoxels.neighbours_of_voxels, c_voxels)


    print ""
    print "Executing Python functions:"
    if is_blacklist:
        voxels = time_func(pyvoxels.voxelize_cloud, cloud, k, class_blacklist=filter)
    else:
        voxels = time_func(pyvoxels.voxelize_cloud, cloud, k, class_whitelist=filter)

    py_neigh = time_func(extract.adjacency_dict_of_voxels, voxels)


    print "Start Comparing {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # Check that voxels dict are the same
    print ""

    missing_voxels = 0
    missing_points = 0
    for key in voxels.keys():
        if key not in c_voxels:
            missing_voxels += 1

        else:
            for point in voxels[key]:
                s = set(c_voxels[key])
                if point not in s:
                    missing_points += 1

    print "c_voxels len: {},pyvoxels len: {}".format(len(c_voxels), len(voxels))
    print "{} missing voxels, {} missing points".format(missing_voxels, missing_points)



    # Check that neighbours dict are the same
    missing_neigh = 0
    for v in py_neigh:
        for neighbour in py_neigh[v]:
            if neighbour not in c_neigh[v]:
                missing_neigh += 1

    print "{} neighours that are un the py dict are not in the c dict".format(missing_neigh)

    missing_neigh = 0
    for v in c_neigh:
        for neighbour in c_neigh[v]:
            if neighbour not in py_neigh[v]:
                missing_neigh += 1

    print "{} neighours that are un the c dict are not in the py dict".format(missing_neigh)

if __name__ == '__main__':
    main()
