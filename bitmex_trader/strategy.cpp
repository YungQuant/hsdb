#include <iostream>
#include "feed.h"
#include <unistd.h>
#include <fstream>
#include <cpprest.h>

using namespace web;
using namespace std;

//not sure if this needs to be in a header file, if so pls fix <3
std::string oldMsgStore;
std::string oldMsgStrat;
json::value::element_vector sells;
json::value::element_vector buys;
json::value buyIds;
json::value sellIds;
//not sure if this needs to be in a header file, if so pls fix <3

void Store(feed * self){
    while(true){
        std::string msg = (*self).socket_msg;
        if(msg != oldMsgStore){
            cout << "Received & Storing Socket Message " << "\n";
            std::ofstream out("log.txt");
            std::ofstream outBack(".backup_log.txt");
    
            out << msg;
            outBack << msg;

            out.close();
            outBack.close();

            oldMsgStore = msg;
        }
    }   
}


void Strategy(feed * self){
    while(true){
        std::string msg = (*self).socket_msg;
        if(msg != oldMsgStrat){
            cout << "Received Socket Message " << "\n";
            try{
                message = json::value::parse(msg);
                if(message[U("action")] == "partial"){
                    auto dataArray = message.at(U("data")).as_array();
                    for(int i=0; i<dataArray.size(); i++;){
			            json::value datum = dataArray[i]; // maybe not computationally optimal 
                        if(datum[U("side")] == "Sell"){
                            sellIds[datum[U("id")]] = json::value::array({datum[U("price")], datum[U("Size")]});
                        }else{
                            buyIds[datum[U("id")]] = json::value::array({datum[U("price")], datum[U("Size")]});
                        }
                    }
                }else if(message[U("action")] == "update"){
                    auto dataArray = message.at(U("data")).as_array();
                    for(int i=0; i<dataArray.size(); i++;){
			            json::value datum = dataArray[i]; // maybe not computationally optimal
                        if(datum[U("side")] == "Sell"){
                            json::value::array arr = sellIds.at(datum[U("id")]);
                            sellIds.at(datum[U("id")]) = {arr[0], datum.at(U("size")).as_number()};
                        else{
			                json::value::array arr = buyIds.at(datum[U("id")]);
                            buyIds.at(datum[U("id")]) = {arr[0], datum.at(U("size")).as_number()};
                        }
                    }
                }else if(message[U("action")] == "insert"){
                    auto dataArray = message.at(U("data")).as_array();
                    for(int i=0; i<dataArray.size(); i++;){
			            json::value datum = dataArray[i]; // maybe not computationally optimal
                        if(datum[U("side")] == "Sell"){
                            sellIds[datum[U("id")]] = json::value::array({datum[U("price")], datum[U("Size")]});       
                        }else{
                            buyIds[datum[U("id")]] = json::value::array({datum[U("price")], datum[U("Size")]});
                        }
                    }
                }else if(message[U("action")] == "delete"){
                    auto dataArray = message.at(U("data")).as_array();
                    for(int i=0; i<dataArray.size(); i++;){
			            json::value datum = dataArray[i]; // maybe not computationally optimal
                        if(datum[U("side")] == "Sell"){
                            json::array::erase(sellIds.at(datum[U("id")]));
                        }else{
                            json::array::erase(buyIds.at(datum[U("id")]));
                        }
                    }
                }
                cout << "Indexed: " << message[U("data")] << "\n :) :) :) :) :) \n";
            }catch{
                cout << "Parsing Failed :()" << "\n";
            }

            oldMsgStrat = msg;
        }
    
    }
}



