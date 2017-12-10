from utils.cloud import LasCloud

import sys

import pyvoxels
import cvoxels
import cyvoxels


def main():
    cloud = LasCloud(sys.argv[1])
    k = 0.2

    filter = [2]
    is_blacklist = True

    print("Number of points: {}".format(len(cloud)))
    print("")


    voxels = exec_voxelization(filter, is_blacklist, pyvoxels.voxelize_cloud, cloud, k)
    c_voxels = exec_voxelization(filter, is_blacklist, cvoxels.voxelize_cloud, cloud, k)
    cy_voxels = exec_voxelization(filter, is_blacklist, cyvoxels.voxelize_cloud, cloud, k)

    compare_voxels_dict(voxels, c_voxels)
    compare_voxels_dict(voxels, cy_voxels)



def exec_voxelization(filter, is_blacklist, voxelize_func, *args):
    if is_blacklist:
        voxels = voxelize_func(*args, class_blacklist=filter)
    else:
        voxels = voxelize_func(*args, class_whitelist=filter)

    return voxels




def compare_voxels_dict(reference, d2):
    missing_voxels = 0
    missing_points = 0
    for key, pts in reference.items():
        try:
            c_points = d2[key]
            s = set(c_points)
            for p in pts:
                if p not in s:
                    missing_points += 1
        except KeyError:
            missing_voxels += 1

    print("Num voxels in referencen: {}, in other: {}".format(len(reference), len(d2)))
    print("{} missing voxels, {} missing points".format(missing_voxels, missing_points))

if __name__ == '__main__':
    main()
