#include "../shared.hpp"

#ifndef _STRATEGY_H_
#define _STRATEGY_H_ 1

using namespace std;

void Strategy() {
    if (hsdb::data::sync == true){
        std::cout
            << "Strategy Bid Size: "
            << hsdb::data::bids["XBTUSD"].size()
            << "\tAsks Size: "
            << hsdb::data::bids["XBTUSD"].size()
            << std::endl;
    } else {
        std::cout
            << "Loading book"
            << std::endl;
    }
}

#endif
