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

namespace cplot {

  namespace detail {

    static std::string s_backend;

    struct _interpreter {
      PyObject * matplotlib;
      PyObject * mpl_toolkits;
      PyObject * axis3d;
      PyObject * pyplot;
      PyObject * cm;

      PyObject * func_figure;
      PyObject * func_show;
      PyObject * func_fig;
      PyObject * clear_plot;
      PyObject * func_pause;
      PyObject * func_tight_layout;
      PyObject * empty_tuple;

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

	 // Defining names of libraries needed
          PyObject* matplotlibname = PyString_FromString("matplotlib");
          PyObject* pyplotname = PyString_FromString("matplotlib.pyplot");
          PyObject* cmname = PyString_FromString("matplotlib.cm");
          PyObject* mpl_toolkitsname = PyString_FromString("mpl_toolkits");
          PyObject* axis3dname = PyString_FromString("mpl_toolkits.mplot3d");


          matplotlib = PyImport_Import(matplotlibname);
          PyObject_CallMethod(matplotlib, const_cast<char*>("use"), const_cast<char*>("s"), s_backend.c_str());

          mpl_toolkits = PyImport_Import(mpl_toolkitsname);
          axis3d = PyImport_Import(axis3dname);
          pyplot = PyImport_Import(pyplotname);
          cm = PyImport_Import(cmname);

          empty_tuple = PyTuple_New(0);

          func_show = PyObject_GetAttrString(pyplot, "show");
          func_figure = PyObject_GetAttrString(pyplot,"figure");
          func_fig = PyObject_CallObject(func_figure, empty_tuple);
          clear_plot = PyObject_GetAttrString(pyplot, "clf");
          func_pause = PyObject_GetAttrString(pyplot, "pause");
          func_tight_layout = PyObject_GetAttrString(pyplot, "tight_layout");

        }
        ~_interpreter() {
          Py_Finalize();
        }
    };
  }

  inline void backend(const std::string& name)
  {
    detail::s_backend = name;
  }

  inline void show()
  {
    bool block = true;
    PyObject * res;
    if(block) {
      res = PyObject_CallObject(detail::_interpreter::get().func_show,
                                detail::_interpreter::get().empty_tuple);
    } else {
      PyObject *kwargs = PyDict_New();
      PyDict_SetItemString(kwargs, "block", Py_False);
      res = PyObject_Call(detail::_interpreter::get().func_show, detail::_interpreter::get().empty_tuple, kwargs);
      Py_DECREF(kwargs);
      Py_DECREF(res);
    }


  }

  #ifndef WITHOUT_NUMPY
  // Type selector for numpy array conversion
  template <typename T> struct select_npy_type { const static NPY_TYPES type = NPY_NOTYPE; }; //Default
  template <> struct select_npy_type<double> { const static NPY_TYPES type = NPY_DOUBLE; };
  template <> struct select_npy_type<float> { const static NPY_TYPES type = NPY_FLOAT; };
  template <> struct select_npy_type<bool> { const static NPY_TYPES type = NPY_BOOL; };
  template <> struct select_npy_type<int8_t> { const static NPY_TYPES type = NPY_INT8; };
  template <> struct select_npy_type<int16_t> { const static NPY_TYPES type = NPY_SHORT; };
  template <> struct select_npy_type<int32_t> { const static NPY_TYPES type = NPY_INT; };
  template <> struct select_npy_type<int64_t> { const static NPY_TYPES type = NPY_INT64; };
  template <> struct select_npy_type<uint8_t> { const static NPY_TYPES type = NPY_UINT8; };
  template <> struct select_npy_type<uint16_t> { const static NPY_TYPES type = NPY_USHORT; };
  template <> struct select_npy_type<uint32_t> { const static NPY_TYPES type = NPY_ULONG; };
  template <> struct select_npy_type<uint64_t> { const static NPY_TYPES type = NPY_UINT64; };

  template<typename Numeric>
  PyObject* array2D(const std::vector<std::vector<Numeric>>& v)
  {
    detail::_interpreter::get();

    npy_intp dims[2] = {static_cast<npy_intp>(v.size()),
                        static_cast<npy_intp>(v[0].size())};

    PyArrayObject *vink = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_DOUBLE);

    double *x_begin = static_cast<double *>(PyArray_DATA(vink));

    for(const ::std::vector<Numeric> & vr : v){
        std::copy(vr.begin(), vr.end(), x_begin);
        x_begin += dims[1];
    }

    return reinterpret_cast<PyObject *>(vink);
  }

  template<typename Numeric>
  PyObject* get_array(const std::vector<Numeric>& v)
  {
      PyObject* list = PyList_New(v.size());
      for(size_t i = 0; i < v.size(); ++i) {
          PyList_SetItem(list, i, PyFloat_FromDouble(v.at(i)));
      }
      return list;
  }

  #endif


  template<typename Numeric>
  void plot_surface(const std::vector<::std::vector<Numeric>> & x, const std::vector<::std::vector<Numeric>> & y,const std::vector<::std::vector<Numeric>> & z, std::string cmap, std::string color)
  {

    detail::_interpreter::get();

    PyObject *xx = array2D(x);
    PyObject *yy = array2D(y);
    PyObject *zz = array2D(z);

    PyObject *args = PyTuple_New(3);
    PyTuple_SetItem(args, 0, xx);
    PyTuple_SetItem(args, 1, yy);
    PyTuple_SetItem(args, 2, zz);

    PyObject *kwargs = PyDict_New();
    PyDict_SetItemString(kwargs, "rstride", PyInt_FromLong(1));
    PyDict_SetItemString(kwargs, "cstride", PyInt_FromLong(1));

    if(cmap != ""){
      PyObject * cmdot = PyObject_GetAttrString(detail::_interpreter::get().cm, cmap.c_str());
      PyDict_SetItemString(kwargs, "cmap", cmdot);
    }

    PyObject *gca_kwargs = PyDict_New();
    PyDict_SetItemString(gca_kwargs, "projection", PyString_FromString("3d"));

    PyObject *gca = PyObject_GetAttrString(detail::_interpreter::get().func_fig, "gca");
    Py_INCREF(gca);

    PyObject *axis = PyObject_Call(gca, detail::_interpreter::get().empty_tuple, gca_kwargs);
    Py_INCREF(axis);

    Py_DECREF(gca);
    Py_DECREF(gca_kwargs);

    PyObject * plot_surfacex = PyObject_GetAttrString(axis, "plot_surface");
    PyObject * res = PyObject_Call(plot_surfacex, args, kwargs);

    Py_DECREF(axis);
    Py_DECREF(args);
    Py_DECREF(kwargs);
    Py_DECREF(res);
  }

  template<typename Numeric>
  void plot_scatter(const std::vector<::std::vector<Numeric>> & x, const std::vector<::std::vector<Numeric>> & y,const std::vector<::std::vector<Numeric>> & z, std::string color)
  {

    detail::_interpreter::get();

    PyObject *xx = array2D(x);
    PyObject *yy = array2D(y);
    PyObject *zz = array2D(z);

    PyObject *args = PyTuple_New(3);
    PyTuple_SetItem(args, 0, xx);
    PyTuple_SetItem(args, 1, yy);
    PyTuple_SetItem(args, 2, zz);

    PyObject *kwargs = PyDict_New();

    PyObject *gca_kwargs = PyDict_New();
    PyDict_SetItemString(gca_kwargs, "projection", PyString_FromString("3d"));

    PyObject *gca = PyObject_GetAttrString(detail::_interpreter::get().func_fig, "gca");
    Py_INCREF(gca);

    PyObject *axis = PyObject_Call(gca, detail::_interpreter::get().empty_tuple, gca_kwargs);
    Py_INCREF(axis);

    Py_DECREF(gca);
    Py_DECREF(gca_kwargs);

    PyObject * plot_surfacex = PyObject_GetAttrString(axis, "scatter");
    PyObject * res = PyObject_Call(plot_surfacex, args, kwargs);

    Py_DECREF(axis);
    Py_DECREF(args);
    Py_DECREF(kwargs);
    Py_DECREF(res);
  }




  template<typename Numeric>
  void set_yticklabels(const std::vector<Numeric> & labels)
  {

    detail::_interpreter::get();

    PyObject *xx = get_array(labels);

    PyObject *args = PyTuple_New(1);
    PyTuple_SetItem(args, 0, xx);

    PyObject *kwargs = PyDict_New();


    PyObject *gca_kwargs = PyDict_New();
    PyDict_SetItemString(gca_kwargs, "projection", PyString_FromString("3d"));

    PyObject *gca = PyObject_GetAttrString(detail::_interpreter::get().func_fig, "gca");
    Py_INCREF(gca);

    PyObject *axis = PyObject_Call(gca, detail::_interpreter::get().empty_tuple, gca_kwargs);
    Py_INCREF(axis);

    Py_DECREF(gca);
    Py_DECREF(gca_kwargs);

    PyObject * plot_surfacex = PyObject_GetAttrString(axis, "set_yticklabels");
    PyObject * res = PyObject_Call(plot_surfacex, args, kwargs);

    Py_DECREF(axis);
    Py_DECREF(args);
    Py_DECREF(kwargs);
    Py_DECREF(res);
  }

  void view_init(double def_angle, double angle) 
  {
    detail::_interpreter::get();

    PyObject *args = PyTuple_New(2);
    PyTuple_SetItem(args, 0, PyFloat_FromDouble(def_angle));
    PyTuple_SetItem(args, 1, PyFloat_FromDouble(angle));

    PyObject *kwargs = PyDict_New();
    PyObject *gca_kwargs = PyDict_New();
    PyDict_SetItemString(gca_kwargs, "projection", PyString_FromString("3d"));

    PyObject *gca = PyObject_GetAttrString(detail::_interpreter::get().func_fig, "gca");
    Py_INCREF(gca);

    PyObject *axis = PyObject_Call(gca, detail::_interpreter::get().empty_tuple, gca_kwargs);
    Py_INCREF(axis);

    Py_DECREF(gca);
    Py_DECREF(gca_kwargs);

    PyObject * plot_surfacex = PyObject_GetAttrString(axis, "view_init");
    PyObject * res = PyObject_Call(plot_surfacex, args, kwargs);

    Py_DECREF(axis);
    Py_DECREF(args);
    Py_DECREF(kwargs);
    Py_DECREF(res);
    

  }

  inline void clf() {
      PyObject *res = PyObject_CallObject(
          detail::_interpreter::get().clear_plot,
          detail::_interpreter::get().empty_tuple);


      Py_DECREF(res);
  }
  template<typename Numeric>
  inline void pause(Numeric interval)
  {
      PyObject* args = PyTuple_New(1);
      PyTuple_SetItem(args, 0, PyFloat_FromDouble(interval));

      PyObject* res = PyObject_CallObject(detail::_interpreter::get().func_pause, args);

      Py_DECREF(args);
      Py_DECREF(res);
  }
  inline void tight_layout() {
      PyObject *res = PyObject_CallObject(
          detail::_interpreter::get().func_tight_layout,
          detail::_interpreter::get().empty_tuple);

      Py_DECREF(res);
  }

  inline void grid(bool flag)
  {
      PyObject* pyflag = flag ? Py_True : Py_False;
      Py_INCREF(pyflag);

      PyObject* args = PyTuple_New(1);
      PyTuple_SetItem(args, 0, pyflag);

      PyObject *kwargs = PyDict_New();
      PyObject *gca_kwargs = PyDict_New();
      PyDict_SetItemString(gca_kwargs, "projection", PyString_FromString("3d"));

      PyObject *gca = PyObject_GetAttrString(detail::_interpreter::get().func_fig, "gca");
      Py_INCREF(gca);

      PyObject *axis = PyObject_Call(gca, detail::_interpreter::get().empty_tuple, gca_kwargs);
      Py_INCREF(axis);

      Py_DECREF(gca);
      Py_DECREF(gca_kwargs);

      PyObject * plot_surfacex = PyObject_GetAttrString(axis, "grid");
      PyObject * res = PyObject_Call(plot_surfacex, args, kwargs);

      Py_DECREF(args);
      Py_DECREF(res);
  }

}
