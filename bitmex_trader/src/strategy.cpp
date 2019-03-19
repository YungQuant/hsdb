#include <iostream>
#include "feed.h"

using namespace std;

void Strategy(feed * self){
    if(self->quant.sync == true){
        std::cout << "Strategy Bid Size: " << self->quant.bids["XBTUSD"].size() << "\tAsks Size: " << self->quant.bids["XBTUSD"].size() << std::endl;
    
      
    } else {
        std::cout << "Loading book" << std::endl;
    }
}
