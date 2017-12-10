import numpy as np

cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cdef tuple get_voxel_coordinates(double point[3], double k, double coords_min[3]):
    return ((point[0] - coords_min[0]) // k ,
           (point[1] - coords_min[1]) // k ,
           (point[2] - coords_min[2]) // k)

@cython.boundscheck(False)
@cython.wraparound(False)
def voxelize_cloud(object cloud, double k, list class_whitelist=None, list class_blacklist=None):
    cdef:
        int i
        dict voxels = {}
        double point[3]
        double coords_min[3]
        np.ndarray[double, ndim=2] coords = cloud.coords
        np.ndarray[np.npy_uint8, ndim=1] classification = cloud.classification
        np.ndarray[np.npy_uint8, ndim=1] class_filter = np.ones(256, dtype=np.uint8)
        tuple voxels_coord
        int num_points = coords.shape[0]

    if (class_blacklist is not None) and (class_whitelist is not None):
        raise ValueError("Provide either a blacklist or a whitelist not both")

    if class_whitelist is not None:
        class_filter[class_filter != class_whitelist] = 0
    elif class_blacklist is not None:
        class_filter[class_blacklist] = 0

    coords_min[0] = cloud.bb_min[0]
    coords_min[1] = cloud.bb_min[1]
    coords_min[2] = cloud.bb_min[2]

    for i in range(num_points):
        point[0] = coords[i, 0]
        point[1] = coords[i, 1]
        point[2] = coords[i, 2]

        if class_filter[classification[i]] == 1:
            voxel_coord = get_voxel_coordinates(point, k, coords_min)
            voxels.setdefault(voxel_coord, []).append(i)

    return voxels