import hello

from PLAnalysis.common import cvoxels, cloudtypes, pyvoxels

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
    print("{} took {} secs".format(func.__name__, exec_time))
    return res


cloud = cloudtypes.LasCloud(filename="C:/Users/Thomas/Documents/LasFiles//PLACE_DE_LA_BOURSE_F_1+000_1+050.las")
k = 0.5

c_voxels = time_func(cvoxels.voxelize_cloud, cloud, k)
voxels = time_func(hello.voxelize_cloud, cloud, k)


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



