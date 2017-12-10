from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

def get_voxel_coordinates(point, k, coords_min, rotation_mat=None):
    """ Get the coordinates of a 3D point in the voxel grid
    """
    if rotation_mat is not None:
        v = int((point[0] - coords_min[0]) / k), \
            int((point[1] - coords_min[1]) / k), \
            int((point[2] - coords_min[2]) / k)
        return tuple((int(c) for c in (np.dot(rotation_mat, v))))

    return int((point[0] - coords_min[0]) / k), int((point[1] - coords_min[1]) / k), int((point[2] - coords_min[2]) / k)

def voxelize_cloud(
        cloud, k, class_blacklist=None, class_whitelist=None, max_workers=4):
    if len(cloud) < 10 ** 6:
        max_workers = 1

    points_per_process = int(len(cloud) / max_workers)
    start_indexes = range(0, len(cloud) + 1 - points_per_process, points_per_process)
    stop_indexes = range(points_per_process, len(cloud) + 1, points_per_process)

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        for start, stop in zip(start_indexes, stop_indexes):
            sliced_cloud = cloud[start:stop]
            sliced_cloud.bb_min = cloud.bb_min
            results.append(
                pool.submit(__voxelize_cloud, sliced_cloud, k, class_blacklist, class_whitelist, start))

    voxels = defaultdict(list)
    for r in results:
        for voxel, points in r.result().items():
            voxels[voxel].extend(points)
    return dict(voxels)


def __voxelize_cloud(cloud, k, class_blacklist=None, class_whitelist=None, offset=0):
    """
    :param cloud: The point cloud to be voxelized
    :param k: Coefficient for the size of the voxels (in meter) Ex: k=0.2 => 20cm
    :param class_blacklist: list of class to ignore point from
    :param class_whitelist:  list of class to only consider point from
    :param offset: Used by the multiprocessing function, don't use this param

    :type cloud: PointCloud
    :type k: float
    :type class_blacklist: list

    :return: A dict: {(x,y,z): [list of points]}
    """

    if class_blacklist is not None:
        points_to_voxelize = ((i, point) for i, point in enumerate(cloud.coords)
                              if cloud.classification[i] not in class_blacklist)
    elif class_whitelist is not None:
        points_to_voxelize = ((i, point) for i, point in enumerate(cloud.coords)
                              if cloud.classification[i] in class_whitelist)
    else:
        points_to_voxelize = enumerate(cloud.coords)

    voxels = defaultdict(lambda: [])
    for i, point in points_to_voxelize:
        coordinates = get_voxel_coordinates(point, k, cloud.bb_min)
        voxels[coordinates].append(i + offset)
    return dict(voxels)