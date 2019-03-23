#include <iostream>
#include "feed.h"
#include <time.h>


void Strategy(feed * self){
    while(true){
        if(self->sync == true){
            
            self->quant.book("XBTUSD");

            for(auto & t : self->quant.OBOOK["BidPrice"]){
                std::cout << t << "\t";
            }
            std::cout << std::endl;

        } 

        sleep(10);
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