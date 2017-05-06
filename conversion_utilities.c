#include "conversion_utilities.h"

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





double **py_matrix_to_c_array(PyArrayObject *arrayin) {
	double **c, *a;
	int i, n, m;

	n = (int*) arrayin->dimensions[0];
	m = (int*) arrayin->dimensions[1];
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


int not_doublematrix(PyArrayObject *mat) {
	if (mat->descr->type_num != NPY_DOUBLE || mat->nd != 2) {
		PyErr_SetString(PyExc_ValueError,
			"In not_doublematrix: array must be of type Float and 2 dimensional (n x m).");
		return 1;
	}
	return 0;
}