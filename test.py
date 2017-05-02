import cvoxels
import numpy as np

# coords = np.vstack([[1.0, 2.0, 3.0, 3.0],
#                    [4.0, 5.0, 6.0, 6.0],
#                    [7.0, 8.0, 9.0, 9.0]]).transpose()

test_coords = np.vstack([[1.0, 0.0, 0.0, 1.0, 1.0],
                        [0.0, 1.0, 0.0, 1.0, 1.0],
                        [0.0, 0.0, 1.0, 1.0, 1.0]]).transpose()

test_coords = [[1.57485563e+06, 3.15746516e+06, 1.48502598e+02],
               [1.57485564e+06, 3.15746516e+06, 1.48492598e+02],
               [1.57485564e+06, 3.15746517e+06, 1.48489598e+02],
               [1.57485564e+06, 3.15746520e+06, 1.48499598e+02]]



test_classification = [1 for i in range(4)]
test_blacklist = [0]
test_coords_min = [np.min(test_coords[:, i]) for i in range(3)]

print cvoxels.version()
print "C: sum: {}".format(cvoxels.sum_coordinates(test_coords))
print "C voxelize: {} ".format(cvoxels.voxelize_cloud(test_coords, test_classification, [], test_coords_min, 0.2))
print "python sum: {}".format(sum(test_coords.ravel()))
# print test_coords_min
print "end"

