#include "uthash.h"
#include "utlist.h"

#ifndef _C_VOXELS_H
#define _C_VOXELS_H

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

    UT_hash_handle hh; // Needed by uthash
};


/* 
 * Python callable function that calls the C function "compute_voxels"
 * Does the conversion between Python args and C types, and vise-versa
 *
 * Expected arguments:
 *  - points coordinates : numpy.array with size (num_points, 3)
 *  - classification : a list of integers
 *  - class_blacklist : a list of integers which gives the classes that the
 * voxelization ignores
 *  - coords_min : a list of double that gives the minimum in x, y, z
 *  - k : a double coefficient that gives the voxel size in meter
 *
 * returns a PyDict of voxels:
 * 	- keys: voxels coordinates
 *  - values : list of index of the points contained in the voxel
 */
static PyObject* voxelize_cloud(PyObject *self, PyObject *args);
static PyObject *neighbours_of_voxels(PyObject *self, PyObject *args);


static PyObject *version(PyObject *self);

/*
 * C function that computes the voxels
 *
 * returns a hash table (implemented in uthash)
 */
static struct Voxel *compute_voxels(const double ** coords, const unsigned char * classification, const int black_list[256], 
									const double * coords_min, double k, unsigned int num_points);


static struct Coordinates get_voxel_coordinates(double x, double y, double z, double k, const double *coords_min);



struct Voxel *new_voxel(struct Coordinates coords, int index);
struct Point *new_point(int index);
struct Coordinates new_coordinates_from_py_tuple(PyObject *tuple);
PyObject *coordinates_to_py_tuple(struct Coordinates c);
#endif