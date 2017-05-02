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

struct black_list {
	int code; // The Key

	UT_hash_handle hh;
};

struct fifo_element {
    int node_index;

    struct fifo_element *prev;
    struct fifo_element *next;
};

struct fifo {
    struct fifo_element *head;
    struct fifo_element *tail;
    unsigned int size;
};





/*
 * Python callable function that calls the C function "compute_voxels"
 * Does the conversion between Python args and C types, and vise-versa
 *
 * Expected arguments:
 *  - points coordinates : numpy.matrix with size (num_points, 3)
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

static PyObject* sum_coordinates(PyObject *self, PyObject *args);
static PyObject *neighbours_of_voxels(PyObject *self, PyObject *args);


static PyObject *version(PyObject *self);

/*
 * C function that computes the voxels
 *
 * returns a hash table (implemented in uthash)
 */
static struct Voxel *compute_voxels(double **coords, int *classification, int *black_list, double *coords_min,
									double k, int num_points, int num_black_list);
static struct Coordinates get_voxel_coordinates(double x, double y, double z, double k, double *coords_min);

double **py_matrix_to_c_matrix(PyArrayObject *py_matrix);
double **pymatrix_to_Carrayptrs(PyArrayObject *arrayin);
double **ptrvector(long n);
int *py_int_list_to_c_array(PyObject *list);
double *py_double_list_to_c_array(PyObject *list);
void free_Carrayptrs(double **v);
int  not_doublematrix(PyArrayObject *mat);
#endif