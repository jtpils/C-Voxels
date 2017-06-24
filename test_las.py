import laspy
from PLAnalysis.common import pyvoxels, extract
from PLAnalysis.common.cloudtypes import LasCloud

import sys
import logging
import cvoxels
import numpy as np

from datetime import datetime

def main():

    cloud = LasCloud(sys.argv[1])
    k = 0.2

    coords = cloud.coords
    classification = cloud.classification
    mins = cloud.bb_min

    filter = [7]
    is_blacklist=False

    c_voxels = voxels = None
    print len(classification)


    print "Start c-voxelization {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    if is_blacklist:
        c_voxels = cvoxels.voxelize_cloud(coords, classification, mins, k, class_blacklist=filter)
    else:
        c_voxels = cvoxels.voxelize_cloud(coords, classification, mins, k, class_whitelist=filter)
    print "End c-voxelization {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    print "Start c-neighbourization {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    c_neigh = cvoxels.neighbours_of_voxels(c_voxels)
    print "End c-neighbourization {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


    print "Start py-voxelization {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    if is_blacklist:
        voxels = pyvoxels.voxelize_cloud(cloud, k, class_blacklist=filter)
    else:
        voxels = pyvoxels.voxelize_cloud(cloud, k, class_whitelist=filter)
    print "End py-voxelization {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    print "Start py-neighbourization {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    py_neigh = extract.adjacency_dict_of_voxels(voxels)
    print "End py-neighbourization {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


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
