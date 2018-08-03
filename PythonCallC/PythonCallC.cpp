
#define PY_ARRAY_UNIQUE_SYMBOL NUMBUF_ARRAY_API

#include <Python.h>
#include "segment-image.h"
#include "numpy/arrayobject.h"

const double e = 2.7182818284590452353602874713527;

double sinh_impl(double x) {
	return (1 - pow(e, (-2 * x))) / (2 * pow(e, -x));
}

double cosh_impl(double x) {
	return (1 + pow(e, (-2 * x))) / (2 * pow(e, -x));
}

double tanh_impl(double x) {
	return sinh_impl(x) / cosh_impl(x);
}

PyObject* tanh_impl(PyObject *, PyObject* o) {
	double x = PyFloat_AsDouble(o);
	double tanh_x = sinh_impl(x) / cosh_impl(x);
	return PyFloat_FromDouble(tanh_x);
}


double GetItemDoubleValue(PyArrayObject *arr , int pos0, int pos1)
{
	void* pPosition = PyArray_GETPTR2(arr, pos0, pos1);
	PyObject *v = PyArray_GETITEM(arr, pPosition);
	double dv = PyFloat_AsDouble(v);
	return dv;
}

PyObject* segmentGraphEdge_impl(PyObject *self, PyObject* args) {

	double maxID, numEdge, k, minSz;
	PyArrayObject * panoedge;
	PyArrayObject * type;
	/* Parse Python args to C args */
	if (!PyArg_ParseTuple(args, "ddO!dd", &maxID, &numEdge, &PyArray_Type, &panoedge, &k, &minSz))
	{
		return NULL;
	}



	edge *edges = new edge[numEdge];
	for (int i = 0; i < numEdge; i++)
	{
		edges[i].a = GetItemDoubleValue(panoedge, 0, i);
		edges[i].b = GetItemDoubleValue(panoedge, 1, i);
		edges[i].w = GetItemDoubleValue(panoedge, 2, i);


	}

	printf("a: %d, b: %d, w: %f\n", edges[0].a, edges[0].b, edges[0].w);
	printf("Loading finished!\n");
	universe *u = segment_graph(maxID, numEdge, edges, k);

	printf("get out of segment_graph\n");
	// post process
	for (int i = 0; i < numEdge; i++)
	{
		int a = u->find(edges[i].a);
		int b = u->find(edges[i].b);
		if ((a != b) && ((u->size(a) < minSz) || (u->size(b) < minSz)))
			u->join(a, b);
	}

	printf("finish post process\n");
	// pass result to python
	int dimensions[1];
	dimensions[0] = maxID;
	PyArrayObject * result = (PyArrayObject *)PyArray_FromDims(1, dimensions, PyArray_DOUBLE);
	
	for (int i = 0; i < maxID; i++)
	{
		double v = (double)(u->find(i));
		PyObject* pv = PyFloat_FromDouble(v);

		void* pPosition = PyArray_GETPTR1(result, i);
		PyArray_SETITEM(result,pPosition,pv);
	}

	printf("packed up output\n");
	delete[] edges;
	printf("delete edges\n");
	//delete u;
	printf("memory released\n");
	return PyArray_Return(result);
}



static PyMethodDef p2c_methods[] = {
	// The first property is the name exposed to Python, fast_tanh, the second is the C++
	// function name that contains the implementation.
	{ "fast_tanh", (PyCFunction)tanh_impl, METH_O, nullptr },
{ "segmentGraphEdge", (PyCFunction)segmentGraphEdge_impl, METH_O, nullptr },

// Terminate the array with an object containing nulls.
{ nullptr, nullptr, 0, nullptr }
};

static PyModuleDef p2c_module = {
	PyModuleDef_HEAD_INIT,
	"PythonCallC",                        // Module name to use with Python import statements
	"Provides some functions, but faster",  // Module description
	0,
	p2c_methods                   // Structure that defines the methods of the module
};

PyMODINIT_FUNC PyInit_PythonCallC() {

	import_array();
	return PyModule_Create(&p2c_module);
}
