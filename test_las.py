import laspy
from PLAnalysis.common import voxel, extract
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

    # classification = [int(x) for x in classification]

    print len(classification)

    print "Start c-voxelization {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    c_voxels = cvoxels.voxelize_cloud(np.matrix(coords), classification, [0], mins, k)
    print "End c-voxelization {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    print "Start c-neighbourization {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    c_neigh = cvoxels.neighbours_of_voxels(c_voxels.keys())
    print "End c-neighbourization {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


    print "Start py-voxelization {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    voxels = voxel.voxelize_cloud(cloud, k, class_blacklist=[0])
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
