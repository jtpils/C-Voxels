#include <stdbool.h>
#include <Python.h>

#include "uthash.h"
#include "utlist.h"

#ifndef _C_VOXELS_H
#define _C_VOXELS_H

#define MAX_CLASS 256

struct PointCloud {
	unsigned num_points;
	double *coords;
	double *bb_min;
	unsigned char *classification;
};

struct Coordinates {
	int x;
	int y;
	int z;
};

struct Point {
	unsigned int index;
	struct Point *next;
};

struct Voxel {
	struct Coordinates coord; // This is the key
	struct Point *points; // list of points (provided py utlist)
	int num_points;
	int index;
	int label;
	bool visited;

	UT_hash_handle hh; // Needed by uthash
};



typedef enum Filter Filter;
enum Filter
{
	blacklisted = -1,
	whitelisted = 0,
};



PyDoc_STRVAR(voxelize_cloud_doc,
	"voxelize_cloud(cloud, k, class_blacklist=None, class_whitelist=None) -> dict\n\n"
	"Computes the voxels of the input cloud and the points contained into them\n"
	"Inputs:\n"
	"cloud: a object that has the following attributes:\n"
	"   coords: matrix of points coordinates (shape num_points x 3)\n"
	"   bb_min: the minimum coordinates along the x,y and z axis\n"
	"   classification: the class attribuated to a point\n"
	"k: coefficient that gives the size for thr voxels\n"
	"class_blacklist: Optionnal, if provided points with a classification code \n"
	"included in the blacklist will be ignored\n"
	"class_whitelist: Optionnal, if provided only points with a classification code\n"
	"included in the whitelist will be considered\n"
	"You can provide one of the 2 filter list (or none at all) at a time\n"
	"Returns a dict: \n"
	"   Keys are voxels coordinates (x,y,z)\n"
	"   Values are lists of points indices belonging to the voxel"
);
static PyObject* voxelize_cloud(PyObject *self, PyObject *args, PyObject *kwargs);

PyDoc_STRVAR(neighbours_of_voxels_doc,
	"neighbours_of_voxels(voxels) -> dict\n\n"
	"Computes the adjacency of voxels\n"
	"Inputs:\n"
	"voxels: a dict of voxels, see voxelize_cloud\n"
	"Return: a dict:\n"
	"   Keys are voxels coords\n"
	"   Values are lists of points "
);
static PyObject* neighbours_of_voxels(PyObject *self, PyObject *args);

PyDoc_STRVAR(project_to_3d_doc,
	"project_to_3d(coords, mask, classification, coords_min, k)\n\n"
	"Projects a classification contained within a mask image to a point cloud\n"
	"Inputs:\n"
	"  - points coordinates : numpy.array with shape (num_points, 3)\n"
	"  - mask: numpy.array (2D) of np.uint8\n"
	"  - classification : numpy.array whith shape (num_points, 1)\n"
	"  - coords_min : list of doubles that gives the minimum in x, y, z\n"
	"  - k: double, coefficient that gives the size of the pixel\n"
	"Returns nothing"
);
static PyObject* project_to_3d(PyObject *self, PyObject *args);

static PyObject* labelize_voxels(PyObject *self, PyObject *args);


/*
 * C function that computes the voxels
 *
 * returns a hash table (implemented in uthash)
 */
static struct Voxel *compute_voxels(const struct PointCloud cloud, double k, const Filter filter_list[MAX_CLASS]);


static struct Coordinates get_voxel_coordinates(double x, double y, double z, double k, const double *coords_min);



struct Voxel *new_voxel(struct Coordinates coords, int index);
struct Voxel new_voxel_stack(struct Coordinates coords, int index);
struct Point *new_point(int index);
struct Coordinates new_coordinates_from_py_tuple(PyObject *tuple);
PyObject *coordinates_to_py_tuple(struct Coordinates c);
#endif