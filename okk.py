from PLAnalysis.common import pyvoxels, cloudtypes
from scipy.spatial import cKDTree
import numpy as np
import sys

import cvoxels


def compute_voxels_normals(voxels):
    print "kdtree"
    voxels_coords = np.array(voxels.keys())
    tree = cKDTree(voxels_coords)

    clusters = tree.query(voxels_coords, 15)  # Need to find what's the best between query_ball_point or query.
    neighbours = clusters[1]

    print "start main loop"
    print neighbours.shape
    cvoxels.test2(voxels, neighbours, np.linalg.svd)




def main():
    print "voxelization"
    cloud = cloudtypes.LasCloud(sys.argv[1])
    cloud.reset_classification()
    voxels = cvoxels.voxelize_cloud(cloud, 0.2)
    print "{} voxels".format(len(voxels))


    compute_voxels_normals(voxels)

    cloud.close()

if __name__ == '__main__':
    main()