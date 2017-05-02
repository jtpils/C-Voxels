import laspy
from PLAnalysis.common import voxel,lascloud, extract
import sys
import logging
import cvoxels
import numpy as np
from collections import OrderedDict


logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s: %(module)s -> %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

def main():



    las_file = laspy.file.File(sys.argv[1])
    coords = np.vstack((las_file.x, las_file.y, las_file.z)).transpose()
    classification = las_file.get_classification()
    mins = las_file.header.min

    classification = [int(x) for x in classification]

    print len(classification)

    # print "min:", cloud.bb_min
    k = 0.2
    logger.info("Star Voxelization")
    c_voxels = OrderedDict(cvoxels.voxelize_cloud(np.matrix(coords), classification, [0], mins, k))
    logger.info("End Voxelization: {} Voxels".format(len(c_voxels)))

    logger.info("Start neighbourizing")
    # d = {(0,0,0):0, (2,2,2):2, (0,0,1):1, (0,1,0):0, (2,3,2):0}
    # d = OrderedDict(d)
    # print d
    keys = c_voxels.keys()
    # print keys
    # for i in range(len(keys)):
    #     print "{} has index: {}".format(keys[i], i)

    c_neigh = cvoxels.neighbours_of_voxels(keys)
    logger.info("Finish neighbourizing")
    cloud = lascloud.LasCloud(sys.argv[1])


    logger.info("Star Voxelization")
    voxels = voxel.voxelize_cloud(cloud, k, class_blacklist=[0])
    logger.info("End Voxelization: {} Voxels".format(len(voxels)))

    logger.info("Start neighbourizing")
    py_neigh = extract.neighbours_from_voxels_dict_multiprocess(keys)
    logger.info("Finish neighbourizing")


    logger.info("Start comparison")
    missing_neigh = 0
    neigh_len = 0
    for (py_voxel_neigh, c_voxel_neigh) in zip(py_neigh, c_neigh):
        # print py_voxel_neigh,"VS",c_voxel_neigh
        for neigh in py_voxel_neigh:
            if neigh not in c_voxel_neigh:
                # print neigh,"NOT FOUND IN"
                missing_neigh += 1
        neigh_len += len(c_voxel_neigh)

    missing_voxels = 0
    missing_points = 0
    for key in voxels.keys():
        if key not in c_voxels:
            missing_voxels += 1

        else:
            # print "{} VS {}".format(voxels[key], c_voxels[key])
            for point in voxels[key]:
                s = set(c_voxels[key])
                if point not in s:
                    missing_points += 1
    logger.info("{} missing voxels, {} missing points, {} missing neigh but {} are in".format(missing_voxels, missing_points, missing_neigh, neigh_len))


    # print "PYTHON NEIGH:\n {} \n\n C NEIGH:\n {}".format(py_neigh, c_neigh)




if __name__ == '__main__':
    main()
