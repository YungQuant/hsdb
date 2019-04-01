#include "./feed.h"
#include "./strategy.cpp"
#include "./PYTHON/pyhandler.h"

#include <cpprest/ws_client.h>
#include <boost/property_tree/json_parser.hpp>
#include <boost/foreach.hpp>
#include "/usr/include/boost/variant.hpp"
#include <boost/property_tree/ptree.hpp>
#include <boost/algorithm/string.hpp>

#include <future>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>
#include <iostream>
#include <map>

#include <fstream>


using namespace web;
using namespace web::websockets::client;

feed::feed(std::string key, std::string secret, bool ON_SWITCH):trader(key, secret),
                                                                quant(30)
{
    feed_url = "wss://www.bitmex.com/realtime";
    is_on = ON_SWITCH;
    sync = true;
    pyhandler::__init__();
}

std::vector<std::string> subscriptions(std::vector<std::string> subs){
    std::vector<std::string> res;
    for(auto & t : subs){
        res.push_back("{\"op\":\"subscribe\",\"args\":[\"" + t + "\"]}");
    }
    subs.clear();
    return res;
}

void __call__(feed * self, std::string message)
{
    self->quant.cyclone(self->quant.json(message));
}

void init_socket(feed * self, std::string sign, std::vector<std::string> items)
{
    (*self).sync = true;
    double t0, t1;
    t0 = (*self).trader.auth.curr_timestamp();
    t1 = t0;

    websocket_client client;
    client.connect((*self).feed_url).wait();

    websocket_outgoing_message ms;
    ms.set_utf8_message(sign);
    client.send(ms).wait();

    for(auto & z : subscriptions(items)){
        ms.set_utf8_message(z);
        client.send(ms).wait();
    }

    while(true){
        client.receive().then([](websocket_incoming_message in_msg)
        {
            return in_msg.extract_string();
        }).then([&](std::string msg){
            __call__(self, msg);
            t0 = (*self).trader.auth.curr_timestamp();
        }).wait();

        if(t1 - t0 > 5){ 
            break; // breaks if no message for 5 seconds
        }

        t1 = (*self).trader.auth.curr_timestamp();   
    }

    client.close().wait();
    
    (*self).sync = false;
}

std::vector<std::future<void>> feed::start(std::ofstream & writer)
{ 
    // Deleted 'trader' and 'instrument' calls
    std::vector<std::string> items;
    items.push_back("orderBookL2");
    items.push_back("margin");
    items.push_back("position");

    std::vector<std::future<void>> conn;
    if(is_on == true){
        writer << "MODE: LIVE\n";
        conn.push_back(std::async(init_socket, this, trader.auth.__ws__(), items));
        conn.push_back(std::async(Strategy, this, std::ref(writer)));
    } else {
        writer << "MODE: TEST\n" << std::endl;
        conn.push_back(std::async(TestStrategy, this));
    }
    return conn;
}
