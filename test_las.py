import sys
from datetime import datetime

import cvoxels

from PLAnalysis.common import pyvoxels, extract
from PLAnalysis.common.cloudtypes import LasCloud
from PLAnalysis.common.timemeasurement import time_func


def main():
    cloud = LasCloud(sys.argv[1])
    k = 0.2

    filter = [2]
    is_blacklist = True

    c_voxels = voxels = None
    print("Number of points: {}".format(len(cloud)))
    print("")

    print("Executing C functions:")
    if is_blacklist:
        c_voxels = time_func(cvoxels.voxelize_cloud, cloud, k, class_blacklist=filter)
    else:
        c_voxels = time_func(cvoxels.voxelize_cloud, cloud, k, class_whitelist=filter)

    c_neigh = time_func(cvoxels.neighbours_of_voxels, c_voxels)

    print("Executing Python functions:")
    if is_blacklist:
        voxels = time_func(pyvoxels.voxelize_cloud, cloud, k, class_blacklist=filter)
    else:
        voxels = time_func(pyvoxels.voxelize_cloud, cloud, k, class_whitelist=filter)

    py_neigh = time_func(extract.adjacency_dict_of_voxels, voxels)

    print("Start Comparing {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    # Check that voxels dict are the same

    missing_voxels = 0
    missing_points = 0
    for key, pts in voxels.items():
        try:
            c_points = c_voxels[key]
            s = set(c_points)
            for p in pts:
                if p not in s:
                    missing_points += 1
        except KeyError:
            missing_voxels += 1

    print("c_voxels len: {},pyvoxels len: {}".format(len(c_voxels), len(voxels)))
    print("{} missing voxels, {} missing points".format(missing_voxels, missing_points))

    # Check that neighbours dict are the same
    missing_neigh = 0
    for v in py_neigh:
        for neighbour in py_neigh[v]:
            if neighbour not in c_neigh[v]:
                missing_neigh += 1

    print("{} neighours that are un the py dict are not in the c dict".format(missing_neigh))

    missing_neigh = 0
    for v in c_neigh:
        for neighbour in c_neigh[v]:
            if neighbour not in py_neigh[v]:
                missing_neigh += 1

    print("{} neighours that are un the c dict are not in the py dict".format(missing_neigh))


if __name__ == '__main__':
    main()
