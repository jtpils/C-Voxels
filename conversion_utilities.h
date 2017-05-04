#ifndef _CONVERSION_UTILITIES_H
#define _CONVERSION_UTILITIES_H

#include <Python.h>
#include <numpy/arrayobject.h>

int *py_int_list_to_c_array(PyObject *list);
double *py_double_list_to_c_array(PyObject *list);


double **py_matrix_to_c_array(PyArrayObject *arrayin);
double **ptrvector(long n);
void free_c_array(double **v);
int  not_doublematrix(PyArrayObject *mat);
#endif