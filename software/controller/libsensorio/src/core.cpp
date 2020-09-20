#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "Python.h"

#include <iostream>
#include <cstring>
#include <chrono>
#include <vector>

#include "numpy/arrayobject.h"

int DEV_FD;
fd_set DEV_FDS;
std::chrono::high_resolution_clock CLOCK;
std::vector<std::chrono::time_point, double> DATA;
struct timeval TIMEOUT = { 10, 0 }; // 10 Seconds

void record_data() {

}


static PyObject* connect(PyObject* self, PyObject* args, PyObject* kwargs) {
    char* device;

    if (!PyArg_ParseTuple(args, "s:device", &device)) {
        return NULL;
    }

    DEV_FD = open(device, O_RDWR | O_NOCTTY | O_NDELAY);
    FD_ZERO(&DEV_FDS);
    FD_SET(DEV_FD, &DEV_FDS);
    fputs("I/O TEST FROM MASTER\n", DEV_FD);
    int ret = select(DEV_FD+1, &DEV_FDS, NULL, NULL, &TIMEOUT);
    PyObject* pyret;
    switch (ret) {
        case -1: {
            pyret = Py_BuildValue("s", "Error reading from device.");
            return pyret;
        }
        case 0: {
            pyret = Py_BuildValue("s", "Timeout reading from device.");
            return pyret;
        }
        case 1: {
            char recvstr[30];
            fgets(recvstr, 30, DEV_FD);
            if (strcmp(recvstr, "I/O RESPONSE FROM HARDWARE")) {
                pyret = Py_BuildValue("s", "Device successfully connected.");
            } else {
                pyret = Py_BuildValue("s", "Not receiving proper data from device.");
            }
            return pyret;
        }
    }
}

static PyObject* start(PyObject* self, PyObject* args, PyObject* kwargs) {

}

static PyObject* end(PyObject* self, PyObject* args, PyObject* kwargs) {

}


#define pyargflag METH_VARARGS | METH_KEYWORDS
static PyMethodDef CoreMethods[] = {
        {"connect", (PyCFunction) start, pyargflag, "Connect to hardware."},
        {"record", (PyCFunction) start, pyargflag, "Initialize hardware polling."},
        {"end", (PyCFunction) end, pyargflag, "Return poll data."},
        {NULL, NULL, 0, NULL}
};

static struct PyModuleDef cModPyDem = {
        PyModuleDef_HEAD_INIT,
        "core", "doc todo",
        -1,
        CoreMethods
};

PyMODINIT_FUNC
PyInit_core(void) {
    //import_array();
    return PyModule_Create(&cModPyDem);
};
