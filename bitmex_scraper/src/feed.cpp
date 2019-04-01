#include "feed.h"

#include "strategy.cpp"

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

feed::feed(std::string key, std::string secret):trader(key, secret),
                                                quant(30)
{
    feed_url = "wss://www.bitmex.com/realtime";
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

void recorder_socket(feed * self, std::string sign, std::vector<std::string> items)
{
    std::ofstream writer;

    writer.open("output_log.txt");

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
            writer << msg << "\n";
        }).wait();
    }

    client.close().wait();
    writer.close();
}


void init_socket(feed * self, std::string sign, std::vector<std::string> items)
{
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
        }).wait();
    }

    client.close().wait();


}

std::vector<std::future<void>> feed::start()
{
    std::vector<std::string> items;
    items.push_back("orderBookL2");
    //items.push_back("trade");
    //items.push_back("instrument");
    //items.push_back("margin");
    //items.push_back("position");

    std::vector<std::future<void>> conn;
    conn.push_back(std::async(init_socket, this, trader.auth.__ws__(), items));
    conn.push_back(std::async(Strategy, this));
    return conn;
}
