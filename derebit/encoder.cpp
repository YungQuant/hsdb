#include "encoder.h"
#include <Python.h>

#include <string>
#include <sstream>
#include <vector>

encoder::encoder(std::string apikey, std::string apisecret)
{
    key = apikey;
    secret = apisecret;
}

std::string encoder::ws_sign()
{
    Py_Initialize();
    PyObject* sysPath = PySys_GetObject((char*)"path");
    PyObject* curDir = PyString_FromString(".");
    PyList_Append(sysPath, curDir);
    Py_DECREF(curDir);

    PyObject* name = PyString_FromString("encryption");
    PyObject* plugMod = PyImport_Import(name);
    Py_DECREF(name);

    PyObject* sign = PyObject_GetAttrString(plugMod, "authenticate");
    Py_DECREF(plugMod);


    PyObject* pkey = PyString_FromString(key.c_str());
    PyObject* psec = PyString_FromString(secret.c_str());

    PyObject* args = PyTuple_New(2);
    PyTuple_SetItem(args, 0, pkey);
    PyTuple_SetItem(args, 1, psec);

    PyObject* resultObj = PyObject_CallObject(sign, args);
    Py_DECREF(sign);
    Py_DECREF(args);

    const char* resultStr = PyString_AsString(resultObj);
    std::string result = resultStr;

    Py_DECREF(resultObj);

    Py_Finalize();

    return result;
}


