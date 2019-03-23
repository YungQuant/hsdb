#include "../shared.hpp"
#include "data.hpp"
#include "strategy.hpp"

#ifndef _FEED_H_
#define _FEED_H_ 1

using namespace web;
using namespace web::websockets::client;

namespace hsdb { namespace feed {
    std::string feed_url = "wss://www.bitmex.com/realtime";
    std::string socket_msg;

    void feed(std::string key, std::string secret) {
        bitmex::bitmex(key, secret);
        hsdb::data::data(60);
    }

    void __call__(std::string message) {
        hsdb::data::cyclone(hsdb::data::json(message));
    }

    std::vector<std::string> subscriptions(std::vector<std::string> subs){
        std::vector<std::string> res;
        for(auto & t : subs){
            res.push_back("{\"op\":\"subscribe\",\"args\":[\"" + t + "\"]}");
        }
        subs.clear();
        return res;
    }

    void recorder_socket(std::string sign, std::vector<std::string> items) {
        std::ofstream writer;

        writer.open("output_log.txt");

        websocket_client client;
        client.connect(feed_url).wait();

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
                writer << msg << "\n\n";
            }).wait();
        }

        client.close().wait();
        writer.close();
    }

    void init_socket(std::string sign, std::vector<std::string> items) {
        std::vector<std::future<void>> *tasks;

        websocket_client client;
        client.connect(feed_url).wait();

        websocket_outgoing_message ms;
        ms.set_utf8_message(sign);
        client.send(ms).wait();

        for(auto & z : subscriptions(items)) {
            ms.set_utf8_message(z);
            client.send(ms).wait();
        }

        while(true) {
            client.receive()
            .then([=](websocket_incoming_message in_msg) {
                return in_msg.extract_string();
            })
            .then([=](std::string msg) {
                tasks->push_back(std::async(__call__, msg));
                tasks->push_back(std::async(Strategy));
            })
            .wait();
        }

        client.close().wait();

        for(auto &e : *tasks){
            e.get();
        }
    }

    std::vector<std::future<void>> start() {
        std::vector<std::string> items;
        items.push_back("orderBookL2");
        //items.push_back("trade");
        //items.push_back("instrument");
        items.push_back("margin");
        items.push_back("position");

        std::vector<std::future<void>> conn;
        conn.push_back(std::async(init_socket, bitmex::auth.__ws__(), items));
        //conn.push_back(std::async(Strategy, this));
        return conn;
    }

} }

#endif