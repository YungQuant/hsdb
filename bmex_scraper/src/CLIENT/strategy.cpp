#include <iostream>

#include <time.h>
#include <fstream>

#include "./feed.h"

void TestStrategy(feed * self) {
    std::cout << "Position: " << self->trader.auth.nonce() << std::endl;
}


void Strategy(feed * self, std::ofstream & writer){
    while(self->sync == true){
        if(self->quant.sync == true && self->quant.quant_sync == true){
            try{
                (*self).quant.prep_book(writer, "XBTUSD");
                usleep(1000000);
                self->quant.updates++;
                std::cout << "Update: " << self->quant.updates << "\n";
            } catch (...) {std::cout << "Restarting @ Strategy on Update: " << self->quant.updates << "\n";}
        }
    }
}



void TStrategy(feed * self, std::ofstream & writer){

	double t0, t1;
	t0 = self->trader.auth.curr_timestamp();
	t1 = t0;

    while(self->sync == true){

        if(self->quant.sync == true){
            
            if(self->quant.quant_sync == true){
                self->quant.book("XBTUSD");

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

                for(auto & bb : data){
                    writer << bb << ",";
                }
                writer << "\n";
				std::cout << "Bids Size: " << self->quant.bids["XBTUSD"].size() << "\tSeconds Passed: " << t1 - t0 << std::endl;

                data.clear();
				t1 = self->trader.auth.curr_timestamp();
            } else {
                std::cout << "Depositing in book" << std::endl;
            }
        } else {
            std::cout << "Loading Data" << std::endl;
        }


    }

    std::cout << "STRATEGY HAS CLOSED AT: " << self->trader.auth.nonce() << std::endl;
    //writer << "\nSocket has to reboot buddy, now the time is: " << self->trader.auth.nonce() << std::endl;
    
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
