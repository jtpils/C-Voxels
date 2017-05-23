import laspy
from PLAnalysis.common import pyvoxels, extract
from PLAnalysis.common.cloudtypes import LasCloud
from PLAnalysis.common.RangeImage import RangeImage
from PLAnalysis.segmentation.vertical import facade_segmentation
import sys
import logging
import cvoxels
import numpy as np

from datetime import datetime

def main():

    cloud = LasCloud(sys.argv[1])
    cloud.reset_classification()
    k = 0.2

    coords = cloud.coords
    classification = np.zeros_like(cloud.classification)
    mins = cloud.bb_min

    print np.linalg.norm(cloud.classification)
    print np.linalg.norm(classification)

    ri = RangeImage(cloud.las_file, k)
    r_count, i_max = ri.ComputeAccumulationImageVoxels()
    mask_facades, mask_objects = facade_segmentation.segment_facades(r_count)

    starttime = datetime.now()
    print "go: {}".format(starttime)
    for i, point in enumerate(ri.coords):
        coordinates = ri.GetCoordinates(point)

        if mask_facades[coordinates] == 255:
            cloud.classification[i] = 6
            
    print "duration "+format(datetime.now()-starttime)

    print np.linalg.norm(cloud.classification)
    print np.linalg.norm(classification)

    starttime = datetime.now()
    print "go: {}".format(starttime)
    tset_arr = np.array([[255,255,255], [0,0,0,], [0,0,0]], dtype=np.uint8)
    cvoxels.project_to_3d(cloud.coords, mask_facades, classification, cloud.bb_min, k, 6)
    print "duration "+format(datetime.now()-starttime)

    print np.linalg.norm(cloud.classification)
    print np.linalg.norm(classification)

    cloud.classification = classification

    cloud.save_classification()
    cloud.close()

if __name__ == '__main__':
    main()
