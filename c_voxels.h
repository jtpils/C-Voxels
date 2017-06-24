#include <stdbool.h>
#include "uthash.h"
#include "utlist.h"

#ifndef _C_VOXELS_H
#define _C_VOXELS_H

#define MAX_CLASS 256

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


/* 
 * Voxelizes the cloud given in input
 *
 * Mandatory Arguments:
 *  - points coordinates : numpy.array with shape (num_points, 3)
 *  - classification : list of integers
 *  - coords_min : list of doubles that gives the minimum in x, y, z
 *  - k : double, coefficient that gives the voxel size in meter
 *
 * Optionnal Keyword Arguments:
 *  - class_blacklist: list of integers representing the classes to ignore
 *  - class_whitelist: list of integers representing the class to consider
 *
 * You must only give either a blacklist or a whitelist but never both at the same time.
 * If no list is provided all points will be considered
 *
 * returns a PyDict of voxels:
 * 	- keys: voxels coordinates (x, y, z)
 *  - values : list of index of the points contained in the voxel
 */

static PyObject* voxelize_cloud(PyObject *self, PyObject *args, PyObject *keywds);
/*
 * Builds a adjacency_dict from a voxels dict
 * 
 * Mandatory Argument:
 *  - voxels: dict of voxels
 *
 * returns a adjacency dict:
 *  - keys: voxels coordinates (x, y, z)
 *  - values: list of voxel coordinates which are neighbour to the key voxel
 */
static PyObject* neighbours_of_voxels(PyObject *self, PyObject *args);


/* Projects a classification contained within a mask image to a point cloud
 *  
 * Mandatory Arguments:
 *  - points coordinates : numpy.array with shape (num_points, 3)
 *  - mask: numpy.array (2D) of np.uint8
 *  - classification : numpy.array whith shape (num_points, 1)
 *  - coords_min : list of doubles that gives the minimum in x, y, z
 *  - k: double, coefficient that gives the size of the pixel
 *
 * returns nothing
 */
static PyObject* project_to_3d(PyObject *self, PyObject *args);

static PyObject* labelize_voxels(PyObject *self, PyObject *args);

static PyObject *version(PyObject *self);

/*
 * C function that computes the voxels
 *
 * returns a hash table (implemented in uthash)
 */
static struct Voxel *compute_voxels(const double * coords, const unsigned char *classification,
 const Filter filter_list[MAX_CLASS], 
									const double *coords_min, double k, unsigned int num_points);


static struct Coordinates get_voxel_coordinates(double x, double y, double z, double k, const double *coords_min);



struct Voxel *new_voxel(struct Coordinates coords, int index);
struct Voxel new_voxel_stack(struct Coordinates coords, int index);
struct Point *new_point(int index);
struct Coordinates new_coordinates_from_py_tuple(PyObject *tuple);
PyObject *coordinates_to_py_tuple(struct Coordinates c);
#endif