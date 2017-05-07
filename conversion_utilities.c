#include "conversion_utilities.h"


double ** py_2d_array_to_c_2d_array(PyArrayObject *array)
{
	
	unsigned int n_rows = (unsigned int) array->dimensions[0];
	unsigned int n_cols = (unsigned int) array->dimensions[1];

	double **out = malloc(sizeof(double*) * n_rows);
	if (!out) {
		return NULL;
	}

	for (unsigned int i = 0; i < n_rows; ++i) {
		out[i] = malloc(sizeof(double*) * n_cols);

		if (!out) { // Can't malloc anymore, free what was created 
			free_2d_array(out, i);
			return NULL;
		}

		for (unsigned int j = 0; j < n_cols; ++j) {
			out[i][j] = *(double*)(array->data + i * array->strides[0] + j * array->strides[1]);
		}
	}
	
	return out;
}

int *py_int_list_to_c_array(PyObject *list) {
	int *array;
	Py_ssize_t array_size;
	PyObject *item;

	array_size = PyObject_Length(list);
	array = malloc(array_size * sizeof * array);

	if (!array) {
		return NULL;
	}

	for (Py_ssize_t i = 0; i < array_size; i++) {
		item = PySequence_GetItem(list, i);

		if (item == NULL) {
			PyErr_SetString(PyExc_TypeError, "item not accessible");
			free(array);
			return NULL;
		}

		if (!PyInt_Check(item)) {
			free(array);  /* free up the memory before leaving */ 
			PyErr_SetString(PyExc_TypeError, "expected sequence of integers");
			return NULL;
		}
		array[i] = PyInt_AsLong(item);
	}
	return array;
}


double *py_double_list_to_c_array(PyObject *list) {
	double *array;
	Py_ssize_t array_size;
	PyObject *item;

	array_size = PyObject_Length(list);
	array = malloc(array_size * sizeof * array);

	if (!array) {
		return NULL;
	}

	for (Py_ssize_t i = 0; i < array_size; ++i) {
		item = PySequence_GetItem(list, i);

		if (item == NULL) {
			PyErr_SetString(PyExc_TypeError, "item not accessible");
			free(array);
			return NULL;
		}

		if (!PyFloat_Check(item)) {
			free(array);  /* free up the memory before leaving */
			PyErr_SetString(PyExc_TypeError, "expected sequence of Float");
			return NULL;
		}
		/* assign to the C array */
		array[i] = PyFloat_AsDouble(item);
	}
	return array;
}





double **numpy_matrix_to_c_array(PyArrayObject *arrayin) {
	double **c, *a;
	int i, n, m;

	n = (int) arrayin->dimensions[0];
	m = (int) arrayin->dimensions[1];
	c = ptrvector(n);
	a = (double *)arrayin->data;  /* pointer to arrayin data as double */
	for (i = 0; i<n; i++) {
		c[i] = a + i*m;
	}
	return c;
}



/* ==== Allocate a double *vector (vec of pointers) ======================
Memory is Allocated!  See void free_Carray(double ** )                  */
double **ptrvector(long n) {
	double **v;
	v = (double **)malloc((size_t)(n * sizeof(double)));
	if (!v) {
		printf("In **ptrvector. Allocation of memory for double array failed.");
		exit(0);
	}
	return v;
}


/* ==== Free a double *vector (vec of pointers) ========================== */
void free_c_array(double **v) {
	free((char*)v);
}

void free_2d_array(double **array, unsigned int n_rows) {
	for (unsigned int i = 0; i < n_rows; ++i) {
		free(array[i]);
	}
	free(array);
}


int not_doublematrix(PyArrayObject *mat) {
	if (mat->descr->type_num != NPY_DOUBLE || mat->nd != 2) {
		PyErr_SetString(PyExc_ValueError,
			"In not_doublematrix: array must be of type Float and 2 dimensional (n x m).");
		return 1;
	}
	return 0;
}