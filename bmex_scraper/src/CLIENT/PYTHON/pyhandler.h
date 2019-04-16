#ifndef PYHANDLER_H
#define PYHANDLER_H

#include <vector>
#include <map>
#include <unordered_map>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <cstdint> // <cstdint> requires c++11 support
#include <functional>

#include <Python.h>
#ifndef WITHOUT_NUMPY
#  define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#  include <numpy/arrayobject.h>
#endif // WITHOUT_NUMPY

#include <iostream>



namespace pyhandler {

    namespace pyclient {

        struct _interpreter {

            PyObject * keras;
            PyObject * keras_models;
            PyObject * load_model;
            PyObject * keras_models_predict;

            static _interpreter& get() {
                static _interpreter ctx;
                return ctx;
            }

            private:
                #ifndef WITHOUT_NUMPY
                #  if PY_MAJOR_VERSION >= 3

                    void *import_numpy() {
                        import_array(); // initialize C-API
                        return NULL;
                    }

                #  else

                    void import_numpy() {
                        import_array(); // initialize C-API
                    }

                #  endif
                #endif
                _interpreter() {
                    Py_Initialize();
                    #ifndef WITHOUT_NUMPY
                            import_numpy(); // initialize numpy C-API
                    #endif
                    //PyObject* sysPath = PySys_GetObject((char*)"path");
                    //PyObject* curDir = PyString_FromString(".");
                    //PyList_Append(sysPath, curDir);

                    PyObject * keras_name = PyString_FromString("keras");
                    PyObject * keras_mod_name = PyString_FromString("keras.models");
                    //PyObject * keras_mod_pred_name = PyString_FromString("keras.models.predict");

                    keras = PyImport_Import(keras_name);
                    keras_models = PyImport_Import(keras_mod_name);
                    //keras_models_predict = PyImport_Import(keras_mod_pred_name);

                    if(!keras){
                        std::cout << "Fail on loading keras" << std::endl;
                    }
                    if(!keras_models){
                        std::cout << "Fail on loading keras.models" << std::endl;
                    }
                    //if(!keras_models_predict){
                    //    std::cout << "Fail on loading keras.models.predict" << std::endl;
                    //}
                    Py_DECREF(keras_name);
                    Py_DECREF(keras_mod_name);
                    //Py_DECREF(keras_mod_pred_name);

                    load_model = PyObject_GetAttrString(keras_models, "load_model");

                    if(!load_model){
                        std::cout << "Load model function did not load" << std::endl;
                    }

                }

                ~_interpreter() {
                    Py_Finalize();
                }
        };
    }


    void __init__(){
        std::cout << "Python Handler Initialized" << std::endl;

        if(!pyclient::_interpreter::get().keras){
            std::cout << "tensorflow did not load" << std::endl;
        } else {
            std::cout << "tensorflow is good to go" << std::endl;
        }

    }


}

#endif
