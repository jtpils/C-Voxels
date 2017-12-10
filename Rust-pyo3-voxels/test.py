from PLAnalysis.common import cvoxels, cloudtypes, pyvoxels

import time
import sys

import rust2py

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
    print("{} took {} secs".format(func.__name__, exec_time))
    return res


def main():
    cloud = cloudtypes.LasCloud(sys.argv[1])
    k = 0.5

    c_voxels = time_func(pyvoxels.voxelize_cloud, cloud, k)
    print("go")
    voxels = time_func(rust2py.voxelize_cloud_2, cloud.coords, cloud.bb_min, k)
    print("\n\nmdr")
    voxels = time_func(rust2py.voxelize_cloud, cloud, k)


    missing_voxels, missing_points = 0, 0

    for key, pts in voxels.items():
        try:
            c_points = c_voxels[key]
        except KeyError:
            missing_voxels += 1


    print("c_voxels len: {},pyvoxels len: {}".format(len(c_voxels), len(voxels)))
    print("{} missing voxels, {} missing points".format(missing_voxels, missing_points))



if __name__ == '__main__':
    main()