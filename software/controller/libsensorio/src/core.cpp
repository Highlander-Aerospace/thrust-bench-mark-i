#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "Python.h"

#include <iostream>
#include <cstring>
#include <chrono>
#include <vector>
#include <thread>

#include "numpy/arrayobject.h"
#include "arduino-serial-lib.c"

int SERIAL_PORT;
std::vector<std::pair<float, float>> DATA;
const int DEV_BAUD = 9600;
bool RECORDING;
std::thread RECORDING_THREAD;

static PyObject* connect(PyObject* self, PyObject* args, PyObject* kwargs) {
    char* device;

    if (!PyArg_ParseTuple(args, "s:device", &device)) {
        return NULL;
    }

    SERIAL_PORT = serialport_init(device, DEV_BAUD);
    serialport_flush(SERIAL_PORT);
    int r = serialport_write(SERIAL_PORT, "I/O TEST FROM MASTER\n");
    PyObject* pyret;
    if (r == -1) {
        std::string msg = "Could not write to device: "+std::string(device);
        pyret = Py_BuildValue("s", msg.c_str());
        return pyret;
    }


    std::string recvstr(100, ' ');
    switch (serialport_read_until(SERIAL_PORT, &recvstr[0], '\n', 100, 5000)) {
        case -1: {
            pyret = Py_BuildValue("s", "Error reading from device.");
            return pyret;
        }
        case -2: {
            pyret = Py_BuildValue("s", "Timeout reading from device.");
            return pyret;
        }
        case 1: {
            if (recvstr == "I/O RESPONSE FROM HARDWARE") {
                pyret = Py_BuildValue("s", "Device successfully connected.");
            } else {
                std::string msg = "Not receiving proper data from device. Received: ";
                msg = msg+recvstr;
                pyret = Py_BuildValue("s", msg.c_str());
            }
            return pyret;
        }
    }
}

static PyObject* serial_close(PyObject* self, PyObject* args, PyObject* kwargs) {
    serialport_close(SERIAL_PORT);
    PyObject* pyret = Py_BuildValue("s", "Closed connection to device.");
    return pyret;
}

/**
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
**/

#define pyargflag METH_VARARGS | METH_KEYWORDS
static PyMethodDef CoreMethods[] = {
        {"connect", (PyCFunction) connect, pyargflag, "Connect to hardware."},
        {"close", (PyCFunction) serial_close, pyargflag, "Close connection to hardware."},
        //{"tare", (PyCFunction) tare, pyargflag, "Tare cell."},
        //{"record", (PyCFunction) start, pyargflag, "Initialize hardware polling."},
        //{"end", (PyCFunction) end, pyargflag, "Return poll data."},
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
