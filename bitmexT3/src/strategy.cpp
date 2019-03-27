#include <iostream>
#include "feed.h"
#include "pyhandler.h"

#include <time.h>


void Strategy(feed * self){
    pyhandler::__init__();
    int counter = 0;
    while(true){

        if(self->quant.sync == true){
            self->quant.book("XBTUSD");

            if(self->quant.obook_sync == true){

                std::vector<double> data;

                for(auto & t : self->quant.OBOOK["BidPrice"]){
                    data.push_back(std::atof(t.c_str()));
                }
                for(auto & t : self->quant.OBOOK["AskPrice"]){
                    data.push_back(std::atof(t.c_str()));
                }
                for(auto & t : self->quant.OBOOK["BidVolume"]){
                    data.push_back(std::atof(t.c_str()));
                }
                for(auto & t : self->quant.OBOOK["AskVolume"]){
                    data.push_back(std::atof(t.c_str()));
                }

                for(auto & t : data){ std::cout << t << "\t";}
                std::cout << "\n" << std::endl;

                /*
                PyObject *lModelP = pyhandler::pyclient::load_model;
                PyObject *args = PyTuple_New(1);
                PyTuple_SetItem(args, 0, PyString_FromString("model_name_version.h5"));


                PyObject *model = PyObject_Call(lModelP, args, PyDict_New());
                if(!model){
                    std::cout << "Model dont work" << std::endl;
                }


                std::cout << std::endl;
                */
                counter += 1;
                data.clear();
                sleep(10)
            }
        } else {
            std::cout << "Loading Data" << std::endl;
        }


    }
}


















































 //std::cout << "Strategy Bid Size: " << self->quant.bids["XBTUSD"].size() << "\tAsks Size: " << self->quant.asks["XBTUSD"].size() << std::endl;

            /*
            for(auto kv : self->quant.bids["XBTUSD"]){
                std::string key = kv.first;
                for(auto & t : self->quant.bids["XBTUSD"][key]){
                    std::cout << t << "\t";
                }
                std::cout << std::endl;
            }
            */
