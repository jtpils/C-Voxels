#ifndef _CONVERSION_UTILITIES_H
#define _CONVERSION_UTILITIES_H

#include <Python.h>
#include <numpy/arrayobject.h>

int *py_int_list_to_c_array(PyObject *list);
double *py_double_list_to_c_array(PyObject *list);


double **numpy_matrix_to_c_array(PyArrayObject *array);
double **py_2d_array_to_c_2d_array(PyArrayObject *array);
int **py_2d_array_to_c_2d_array_int(PyArrayObject *array);
unsigned char **py_2d_array_to_c_2d_array_unsigned_char(PyArrayObject *array);

double **ptrvector(long n);
void free_c_array(double **v);
void free_2d_array(void **array, unsigned int n_rows);
int  not_doublematrix(PyArrayObject *mat);
#endif