#include <Python.h>
#include <numpy/arrayobject.h>

#include "c_voxels.h"
#include "conversion_utilities.h"
#include "uthash.h"

//#include "pyconfig.h"



//=====================================================================
// Functions callable from Python
//=====================================================================
static PyObject* voxelize_cloud(PyObject *self, PyObject *args) {
    PyArrayObject *py_coords;
    PyObject *py_classification, *py_class_black_list, *py_coords_min;
	

    double *c_coords_min;
	unsigned char *c_classification;
	unsigned int num_points;
    double k;

    // Parse args from python call
	if (!PyArg_ParseTuple(args, "O!OOOd", &PyArray_Type, &py_coords, &py_classification, &py_class_black_list,
                          &py_coords_min, &k))
        return NULL;

    // Check that we actually get lists for some of the args
    if (!PySequence_Check(py_classification) || !PySequence_Check(py_class_black_list) ||
        !PySequence_Check(py_coords_min)) {
        PyErr_SetString(PyExc_TypeError, "expected sequence");
        return NULL;
    }

	num_points = (unsigned int) py_coords->dimensions[0];

	double **c_coords = py_2d_array_to_c_2d_array(py_coords);
	if (c_coords == NULL) {
		free_2d_array(c_coords, num_points);
		return PyErr_NoMemory();
	}

	// Convert classification to a contiguous array that can be used in C
	PyArrayObject *classification_array;
	classification_array = (PyArrayObject *) PyArray_FROM_OTF(py_classification, NPY_UINT8, NPY_IN_ARRAY);

	if (classification_array->dimensions[0] != num_points) {
		PySys_WriteStdout("Error: Classification is not the same length as the number of points");
	}
	
	int c_class_black_list[256] = { 0 };
	for (Py_ssize_t i = 0; i < PyList_Size(py_class_black_list); ++i) {
		int code = PyInt_AsLong(PyList_GetItem(py_class_black_list, i));
		c_class_black_list[code] = 1;
	}

	c_classification = (unsigned char*) classification_array->data;
    c_coords_min = py_double_list_to_c_array(py_coords_min);
	
    struct Voxel *current_voxel, *tmp, *voxels = NULL;
    voxels = compute_voxels((const double **)c_coords, c_classification, c_class_black_list, c_coords_min, k, num_points);

	if (!voxels) {
		PyArray_XDECREF_ERR(classification_array);
		free(c_coords_min);
		free_2d_array(c_coords, num_points);
		return PyErr_NoMemory();
	}

    struct Point *current_point, *tmp_point;
    PyObject *voxels_dict = PyDict_New();

    HASH_ITER(hh, voxels, current_voxel, tmp) {
        int index = 0;
        PyObject *list_of_points = PyList_New(current_voxel->num_points);
        PyObject *key = Py_BuildValue("(i,i,i)", current_voxel->coord.x, current_voxel->coord.y, current_voxel->coord.z);

        LL_FOREACH_SAFE(current_voxel->points, current_point, tmp_point) {
            PyList_SetItem(list_of_points, index, Py_BuildValue("i", current_point->index));
            LL_DELETE(current_voxel->points, current_point);
            free(current_point);
            ++index;
        }
        PyDict_SetItem(voxels_dict, key, list_of_points);


        HASH_DEL(voxels, current_voxel);
        free(current_voxel);
    }


	PyArray_XDECREF_ERR(classification_array);
    free(c_coords_min);
	free_2d_array(c_coords, num_points);
	return voxels_dict;
}



static PyObject *neighbours_of_voxels(PyObject *self, PyObject *args) {
	PyObject *py_voxels, *py_keys;
    struct Voxel *c_voxels = NULL, *v;

	// Parse args from python call
	if (!PyArg_ParseTuple(args, "O", &py_voxels))
        return NULL;

	// Check that we got the good argument
	if (!PyDict_Check(py_voxels)) {
		PyErr_SetString(PyExc_TypeError, "Expected a dict");
	}

	py_keys = PyDict_Keys(py_voxels);
	int num_voxels = (int) PyList_Size(py_keys);

    if(!PySequence_Check(py_keys)) {
        PyErr_SetString(PyExc_TypeError, "Keys are not a sequence");
    }

    // Creating the PyObject that we will return
    PyObject *neighbours = PyDict_New();


    // Build the hash table of voxels
    // Caring only about the keys, not the values
	for (int i = 0; i < num_voxels; ++i) {
		PyObject *key = PyList_GetItem(py_keys, i);
		struct Coordinates c = new_coordinates_from_py_tuple(key);
		v = new_voxel(c, i);
        HASH_ADD(hh, c_voxels, coord, sizeof(struct Coordinates), v);
	}

    // Now loop over the voxels to find their neighbours
    struct Voxel *voxel, *p;
    for(voxel = c_voxels; voxel != NULL; voxel = voxel->hh.next) {

        struct Coordinates top = voxel->coord, bot = voxel->coord, left = voxel->coord,
                           right = voxel->coord, front = voxel->coord, back = voxel->coord;
        top.z += 1; bot.z -= 1; right.x += 1; left.x -= 1; front.y += 1; back.y -= 1;
        struct Coordinates potential_neighbours[6] = {top, bot, left, right, front, back};


        PyObject *neighbours_list = PyList_New(0);
        for (int k = 0; k < 6; ++k) {
            HASH_FIND(hh, c_voxels, &(potential_neighbours[k]), sizeof(struct Coordinates), p);

            if (p) {
                PyList_Append(neighbours_list, coordinates_to_py_tuple(potential_neighbours[k]));
            }
        }
        PyDict_SetItem(neighbours, coordinates_to_py_tuple(voxel->coord), neighbours_list);
    }


    HASH_ITER(hh, c_voxels, p, v) {
        HASH_DEL(c_voxels, p);
        free(p);
    }


    return neighbours;
}

static PyObject* version(PyObject* self)
{
    return Py_BuildValue("s", "Version 0.1");
}

//=====================================================================
// C functions
//=====================================================================
static struct Coordinates get_voxel_coordinates(double x, double y, double z, double k, const double *coords_min) {
    struct Coordinates voxel_coords;
    voxel_coords.x = (int)(((x - coords_min[0])) / k);
    voxel_coords.y = (int)(((y - coords_min[1])) / k);
    voxel_coords.z = (int)(((z - coords_min[2])) / k);
    return voxel_coords;
}

// TODO: Lets create a struct PointCloud !
static struct Voxel *compute_voxels(const double ** coords, const unsigned char * classification, const int black_list[256], 
									const double * coords_min, double k, unsigned int num_points) {

    struct Voxel *p = NULL, *voxels = NULL;

    for (unsigned int i = 0; i < num_points; ++i) {
       
        if (black_list[(int)classification[i]]) {
            continue;
        }
        struct Coordinates c = get_voxel_coordinates(coords[i][0], coords[i][1], coords[i][2], k, coords_min);
        struct Voxel *v = new_voxel(c, i);

		if (!v) {
			return NULL;
		}

        HASH_FIND(hh, voxels, &(v->coord), sizeof(struct Coordinates), p);

		struct Point *pp = new_point(i);

		if (!pp) {
			return NULL;
		}

        if (!p) { // Voxel not found in hash, add it
            HASH_ADD(hh, voxels, coord, sizeof(struct Coordinates), v);
            LL_PREPEND(v->points, pp);
            v->num_points += 1;
        }
        else {
            free(v); //Voxel already exist, delete the one we just created
            LL_PREPEND(p->points, pp);
            p->num_points += 1;

        }
    }
    return voxels;
}

//=====================================================================
// Other things
//=====================================================================
struct Voxel *new_voxel(struct Coordinates coords, int index) {
	struct Voxel *v = (struct Voxel*) malloc(sizeof(struct Voxel));
	
	if (v) {
		memset(v, 0, sizeof(struct Voxel));
		v->coord = coords;
		v->points = NULL;
		v->num_points = 0;
		v->index = index;
	}
	return v;
}

struct Point *new_point(int index) {
	struct Point *point = (struct Point*) malloc(sizeof(struct Point));

	if (point) {
		point->next = NULL;
		point->index = index;
	}
	return point;
}

struct Coordinates new_coordinates_from_py_tuple(PyObject *tuple) {
	struct Coordinates c = {
		PyInt_AsLong(PyTuple_GetItem(tuple, 0)),
		PyInt_AsLong(PyTuple_GetItem(tuple, 1)),
		PyInt_AsLong(PyTuple_GetItem(tuple, 2))
	};
	return c;
}

PyObject *coordinates_to_py_tuple(struct Coordinates c) {
	return Py_BuildValue("(i,i,i)", c.x, c.y, c.z);
}


//=====================================================================
// Python API function to register module
//=====================================================================

static PyMethodDef cvoxel_methods[] = {
    {"voxelize_cloud", voxelize_cloud, METH_VARARGS, "Voxelize the cloud"},
    {"version", (PyCFunction)version, METH_NOARGS, "Returns de module version"},
    {"neighbours_of_voxels", neighbours_of_voxels, METH_VARARGS, "ayy lmao"},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initcvoxels(void) {
    (void) Py_InitModule("cvoxels", cvoxel_methods);
    import_array();
}
