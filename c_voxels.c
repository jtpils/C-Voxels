#include <numpy/arrayobject.h>

#include "c_voxels.h"
#include "conversion_utilities.h"
#include "uthash.h"


//=====================================================================
// Functions callable from Python
//=====================================================================
static PyObject* voxelize_cloud(PyObject *self, PyObject *args, PyObject *kwargs) {

	PyObject *py_cloud, *py_class_black_list = NULL, *py_class_white_list = NULL;
	double k;

	// Parse args from python call
	static const char *kwlist[] = { "cloud", "k", "class_blacklist", "class_whitelist", NULL };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Od|OO", kwlist, &py_cloud, &k, &py_class_black_list, &py_class_white_list))
		return NULL;

	if (py_class_black_list && py_class_white_list) {
		PyErr_SetString(PyExc_ValueError, "Provide either a black list or white list, not both");
		return NULL;
	}

	// Get attributes from the cloud object
	PyObject *coords_attr = PyObject_GetAttrString(py_cloud, "coords");
	PyObject *bb_min_attr = PyObject_GetAttrString(py_cloud, "bb_min");
	PyObject *classification_attr = PyObject_GetAttrString(py_cloud, "classification");

	if (!coords_attr || !classification_attr || !bb_min_attr) {
		PyErr_SetString(PyExc_ValueError, "The cloud object must have the following attributes: coords, classification, bb_min");
		return NULL;
	}

	// Convert attributes to contiguous arrays
	PyArrayObject *coords_array = (PyArrayObject *) PyArray_FROM_OTF(coords_attr, NPY_DOUBLE, NPY_IN_ARRAY);
	PyArrayObject *bb_min_array = (PyArrayObject *) PyArray_FROM_OTF(bb_min_attr, NPY_DOUBLE, NPY_IN_ARRAY);
	PyArrayObject *classification_array = (PyArrayObject *) PyArray_FROM_OTF(classification_attr, NPY_UINT8, NPY_IN_ARRAY);

	// Get pointers to data from the arrays
	unsigned num_points = (unsigned) PyArray_DIM(coords_attr, 0);
	const double *coords = (double *) PyArray_DATA(coords_array);
	const double *bb_min = (double *) PyArray_DATA(bb_min_array);
	unsigned char *classification = (unsigned char *) PyArray_DATA(classification_array);

	// Init blacklist or whitelist
	Filter c_class_filter[MAX_CLASS] = { whitelisted };
	if (py_class_black_list) {
		for (Py_ssize_t i = 0; i < PyList_Size(py_class_black_list); ++i) {
			int code = PyInt_AsLong(PyList_GetItem(py_class_black_list, i));
			c_class_filter[code] = blacklisted;
		}
	}
	if (py_class_white_list) {
		for (size_t i = 0; i < MAX_CLASS; ++i) {
			c_class_filter[i] = blacklisted;
		}
		for (Py_ssize_t i = 0; i < PyList_Size(py_class_white_list); ++i) {
			int code = PyInt_AsLong(PyList_GetItem(py_class_white_list, i));
			c_class_filter[code] = whitelisted;
		}
	}

	struct PointCloud cloud = { num_points, coords, bb_min, classification };

	PyObject *voxels_dict = PyDict_New();
	struct Voxel *current_voxel, *tmp, *voxels = NULL;
	voxels = compute_voxels(cloud, k, c_class_filter);

	if (!voxels) {
		goto end;
	}

	struct Point *current_point, *tmp_point;

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

end:
	PyArray_XDECREF_ERR(classification_array);
	PyArray_XDECREF_ERR(bb_min_array);
	PyArray_XDECREF_ERR(coords_array);
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

	if (!PySequence_Check(py_keys)) {
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
	for (voxel = c_voxels; voxel != NULL; voxel = voxel->hh.next) {

		struct Coordinates top = voxel->coord, bot = voxel->coord, left = voxel->coord,
			right = voxel->coord, front = voxel->coord, back = voxel->coord;
		top.z += 1; bot.z -= 1; right.x += 1; left.x -= 1; front.y += 1; back.y -= 1;
		struct Coordinates potential_neighbours[6] = { top, bot, left, right, front, back };


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

static PyObject* labelize_voxels(PyObject *self, PyObject *args) {
	PyObject *adjacency;

	if (!PyArg_ParseTuple(args, "O", &adjacency))
		return NULL;

	size_t num_voxels = PyObject_Length(adjacency);
	bool *visited = malloc(sizeof(bool) * num_voxels);


	free(visited);
	return Py_None;
}


static PyObject* project_to_3d(PyObject *self, PyObject *args) {
	PyArrayObject *py_mask;
	PyObject *py_cloud;
	double k;
	int code;

	// Parse args from python call
	if (!PyArg_ParseTuple(args, "OO!di", &py_cloud, &PyArray_Type, &py_mask, &k, &code))
		return NULL;

	// Get attributes from the cloud object
	PyObject *coords_attr = PyObject_GetAttrString(py_cloud, "coords");
	PyObject *bb_min_attr = PyObject_GetAttrString(py_cloud, "bb_min");
	PyObject *classification_attr = PyObject_GetAttrString(py_cloud, "classification");

	// Convert attributes to contiguous arrays
	PyArrayObject *coords_array = (PyArrayObject *) PyArray_FROM_OTF(coords_attr, NPY_DOUBLE, NPY_IN_ARRAY);
	PyArrayObject *bb_min_array = (PyArrayObject *) PyArray_FROM_OTF(bb_min_attr, NPY_DOUBLE, NPY_IN_ARRAY);
	PyArrayObject *classification_array = (PyArrayObject *) PyArray_FROM_OTF(classification_attr, NPY_UINT8, NPY_IN_ARRAY);

	// Get pointers to data from the arrays
	unsigned num_points = (unsigned) PyArray_DIM(coords_attr, 0);
	const double *bb_min = (double *) PyArray_DATA(bb_min_array);
	unsigned char *classification = (unsigned char *) PyArray_DATA(classification_array);

	for (unsigned i = 0; i < num_points; ++i) {
		double x = (double) *(double*) PyArray_GETPTR2(coords_array, i, 0);
		double y = (double) *(double*) PyArray_GETPTR2(coords_array, i, 1);
		double z = (double) *(double*) PyArray_GETPTR2(coords_array, i, 2);

		struct Coordinates c = get_voxel_coordinates(x, y, z, k, bb_min);
		if ((int) *(unsigned char*) PyArray_GETPTR2(py_mask, c.x, c.y) == 255) {
			classification[i] = (unsigned char) code;
		}
	}

	PyArray_XDECREF_ERR(classification_array);
	PyArray_XDECREF_ERR(bb_min_array);
	PyArray_XDECREF_ERR(coords_array);
	return Py_None;
}

//=====================================================================
// C functions
//=====================================================================
static struct Coordinates get_voxel_coordinates(double x, double y, double z, double k, const double *coords_min) {
	struct Coordinates voxel_coords;
	voxel_coords.x = (int) (((x - coords_min[0])) / k);
	voxel_coords.y = (int) (((y - coords_min[1])) / k);
	voxel_coords.z = (int) (((z - coords_min[2])) / k);
	return voxel_coords;
}

static struct Voxel *compute_voxels(const struct PointCloud cloud, double k, const Filter filter_list[MAX_CLASS]) {

	struct Voxel *p = NULL, *voxels = NULL;
	unsigned int i;
	for (i = 0; i < cloud.num_points; ++i) {

		if (filter_list[(int) cloud.classification[i]] != whitelisted) {
			continue;
		}
		struct Coordinates c = get_voxel_coordinates(cloud.coords[(3 * i) + 0], cloud.coords[(3 * i) + 1], cloud.coords[(3 * i) + 2], k, cloud.bb_min);
		struct Voxel v = new_voxel_stack(c, i);

		HASH_FIND(hh, voxels, &(v.coord), sizeof(struct Coordinates), p);

		struct Point *pp = new_point(i);

		if (!pp) {
			return NULL;
		}

		if (!p) { // Voxel not found in hash, add it
			struct Voxel *vox = new_voxel(c, i);
			HASH_ADD(hh, voxels, coord, sizeof(struct Coordinates), vox);
			LL_PREPEND(vox->points, pp);
			vox->num_points += 1;
		}
		else { // Voxel already exists, add current point
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

struct Voxel new_voxel_stack(struct Coordinates coords, int index) {
	struct Voxel v;
	memset(&v, 0, sizeof(struct Voxel));
	v.coord = coords;
	v.points = NULL;
	v.num_points = 0;
	v.index = index;

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
	{"voxelize_cloud", (PyCFunction) voxelize_cloud, METH_VARARGS | METH_KEYWORDS, voxelize_cloud_doc},
	{"neighbours_of_voxels", neighbours_of_voxels, METH_VARARGS, neighbours_of_voxels_doc},
	{"project_to_3d", project_to_3d, METH_VARARGS, project_to_3d_doc},
	{NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initcvoxels(void) {
	(void) Py_InitModule("cvoxels", cvoxel_methods);
	import_array();
}
