from PLAnalysis.common import lascloud
import cvoxels
import logging
import sys
from collections import OrderedDict
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s: %(module)s -> %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


def main():
    cloud = lascloud.LasCloud(sys.argv[1])
    k = 0.2
    classification = [int(x) for x in cloud.classification]
    logger.info("Star Voxelization")
    c_voxels = OrderedDict(cvoxels.voxelize_cloud(np.matrix(cloud.coords), classification, [0], cloud.bb_min, k))
    logger.info("End Voxelization: {} Voxels".format(len(c_voxels)))

    logger.info("Start neighbourizing")
    keys = c_voxels.keys()
    c_neigh = cvoxels.neighbours_of_voxels(keys)
    logger.info("Finish neighbourizing")

if __name__ == '__main__':
    main()