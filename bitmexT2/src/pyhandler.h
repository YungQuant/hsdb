#ifndef PYHANDLER_H
#define PYHANDLER_H

#include <Python.h>
#include <string>
#include <iostream>


namespace pyhandler {

    namespace pyclient {


        struct _interpreter {
            PyObject * tensorflow;

            static _interpreter& get() {
                static _interpreter ctx;
                return ctx;
            }

            private:

                _interpreter() {
                    Py_Initialize();
                    PyObject* sysPath = PySys_GetObject((char*)"path");
                    PyObject* curDir = PyString_FromString(".");
                    PyList_Append(sysPath, curDir);

                    PyObject * tensorflow_name, scikit_name;
                    tensorflow_name = PyString_FromString("tensorflow");

                    tensorflow = PyImport_Import(tensorflow_name);
                    Py_DECREF(tensorflow_name);


                    
                }

                ~_interpreter() {
                    Py_Finalize();
                }
        };
    }

    void __init__(){

    }


}

#endif
