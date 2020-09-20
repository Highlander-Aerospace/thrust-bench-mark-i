#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "Python.h"

#include <iostream>
#include <cstring>
#include <chrono>
#include <vector>
#include <fcntl.h>
#include <termios.h>
#include <thread>

#include "numpy/arrayobject.h"


int DEV_FD;
fd_set DEV_FDS;
FILE* DEV_FILE;
std::vector<std::pair<float, float>> DATA;
struct timeval TIMEOUT = { 10, 0 }; // 10 Seconds
const speed_t DEV_BAUD = B115200;
bool RECORDING;
std::thread RECORDING_THREAD;


int await_from_serial() {
    return select(DEV_FD+1, &DEV_FDS, NULL, NULL, &TIMEOUT);
}

void send_to_serial(const char* msg) {
    fputs(msg, DEV_FILE);
}

void read_from_serial(char* dest, int len) {
    fgets(dest, len, DEV_FILE);
}

float read_float_from_serial() {
    float r;
    fscanf(DEV_FILE, "%f\n", &r);
    return r;
}

static PyObject* connect(PyObject* self, PyObject* args, PyObject* kwargs) {
    char* device;

    if (!PyArg_ParseTuple(args, "s:device", &device)) {
        return NULL;
    }

    // dwbi
    DEV_FD = open(device, O_RDWR | O_NOCTTY | O_NDELAY);
    DEV_FILE = fdopen(DEV_FD, "rw");
    FD_ZERO(&DEV_FDS);
    FD_SET(DEV_FD, &DEV_FDS);
    struct termios settings;
    tcgetattr(DEV_FD, &settings);
    cfsetospeed(&settings, DEV_BAUD);
    settings.c_oflag &= ~OPOST;
    tcsetattr(DEV_FD, TCSANOW, &settings);
    tcflush(DEV_FD, TCOFLUSH);

    send_to_serial("I/O TEST FROM MASTER\n");
    PyObject* pyret;
    switch (await_from_serial()) {
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
            read_from_serial(recvstr, 30);
            if (strcmp(recvstr, "I/O RESPONSE FROM HARDWARE") == 0) {
                pyret = Py_BuildValue("s", "Device successfully connected.");
            } else {
                pyret = Py_BuildValue("s", "Not receiving proper data from device.");
            }
            return pyret;
        }
    }
}

static PyObject* tare(PyObject* self, PyObject* args, PyObject* kwargs) {
    send_to_serial("T\n");
    PyObject* pyret;
    switch (await_from_serial()) {
        case -1: {
            pyret = Py_BuildValue("s", "Error getting confirmation from device.");
            return pyret;
        }
        case 0: {
            pyret = Py_BuildValue("s", "Timeout getting confirmation from device.");
            return pyret;
        }
        case 1: {
            char recvstr[2];
            read_from_serial(recvstr, 2);
            if (strcmp(recvstr, "T") == 0) {
                pyret = Py_BuildValue("s", "Device successfully tared.");
            } else {
                pyret = Py_BuildValue("s", "Not receiving proper data from device.");
            }
            return pyret;
        }
    }
}

void record_data() {
    auto start = std::chrono::high_resolution_clock::now();
    float val;
    std::chrono::duration<float, std::nano> t{};
    while (RECORDING) {
        if (await_from_serial() == 1) {
            val = read_float_from_serial();
            t = std::chrono::high_resolution_clock::now()-start;
            DATA.emplace_back(t.count(), val);
        }
    }
}

static PyObject* start(PyObject* self, PyObject* args, PyObject* kwargs) {
    send_to_serial("R\n");
    PyObject* pyret;
    switch (await_from_serial()) {
        case -1: {
            pyret = Py_BuildValue("s", "Error getting confirmation from device.");
            return pyret;
        }
        case 0: {
            pyret = Py_BuildValue("s", "Timeout getting confirmation from device.");
            return pyret;
        }
        case 1: {
            char recvstr[2];
            read_from_serial(recvstr, 2);
            if (strcmp(recvstr, "R") == 0) {
                pyret = Py_BuildValue("s", "Device recording started.");
                DATA.clear();
                RECORDING = true;
                RECORDING_THREAD = std::thread(&record_data);
            } else {
                pyret = Py_BuildValue("s", "Not receiving proper data from device.");
            }
            return pyret;
        }
    }
}

static PyObject* end(PyObject* self, PyObject* args, PyObject* kwargs) {
    send_to_serial("S\n");
    PyObject* pyret;
    switch (await_from_serial()) {
        case -1: {
            pyret = Py_BuildValue("s", "Error getting confirmation from device.");
            return pyret;
        }
        case 0: {
            pyret = Py_BuildValue("s", "Timeout getting confirmation from device.");
            return pyret;
        }
        case 1: {
            char recvstr[2];
            read_from_serial(recvstr, 2);
            if (strcmp(recvstr, "S") == 0) {
                pyret = Py_BuildValue("s", "Device recording stopped.");
                RECORDING = false;
                RECORDING_THREAD.join();
                npy_intp dims[2] = {2, static_cast<npy_intp>(DATA.size())};
                PyObject* ret = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, DATA.data());
                PyArray_ENABLEFLAGS((PyArrayObject *) ret, NPY_ARRAY_OWNDATA);
                return ret;
            } else {
                pyret = Py_BuildValue("s", "Not receiving proper data from device.");
            }
            return pyret;
        }
    }
}


#define pyargflag METH_VARARGS | METH_KEYWORDS
static PyMethodDef CoreMethods[] = {
        {"connect", (PyCFunction) connect, pyargflag, "Connect to hardware."},
        {"tare", (PyCFunction) tare, pyargflag, "Tare cell."},
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
    import_array();
    return PyModule_Create(&cModPyDem);
};
