#include <iostream>
#include "feed.h"
#include <unistd.h>
#include <fstream>
#include <cpprest/json.h>

using namespace web;
using namespace std;

//not sure if this needs to be in a header file, if so pls fix <3
std::string oldMsgStore;
std::string oldMsgStrat;
//json::value::element_vector sells;
//json::value::element_vector buys;
json::value buyIds;
json::value sellIds;
//not sure if this needs to be in a header file, if so pls fix <3

json::value constructor(feed * self){
    std::string msg = (*self).socket_msg;
	if(msg != oldMsgStrat){
            cout << "Received Socket Message " << "\n";
            try{
                json::value message = json::value::parse(msg);
                if((message[U("action")]).as_string() == "partial"){
                    auto dataArray = message.at(U("data")).as_array();
                    for(int i=0; i<dataArray.size(); i++){
			            json::value datum = dataArray[i]; // maybe not computationally optimal 
                        if((datum[U("side")]).as_string() == "Sell"){
                            sellIds[U(datum[U("id")])] = json::value::array({datum[U("price")], datum[U("Size")]});
                        }else{
                            buyIds[U(datum[U("id")])] = json::value::array({datum[U("price")], datum[U("Size")]});
                        }
                    }
                }else if((message[U("action")]).as_string() == "update"){
                    auto dataArray = message.at(U("data")).as_array();
                    for(int i=0; i<dataArray.size(); i++){
			            json::value datum = dataArray[i]; // maybe not computationally optimal
                        if((datum[U("side")]).as_string() == "Sell"){
                            json::value::array arr = sellIds.at(U(datum[U("id")]));
                            sellIds.at(U(datum[U("id")])) = {arr[0], datum.at(U("size")).as_number()};
                        }else{
			                json::value::array arr = buyIds.at(U(datum[U("id")]));
                            buyIds.at(U(datum[U("id")])) = {arr[0], datum.at(U("size")).as_number()};
                        }
                    }
                }else if((message[U("action")]).as_string() == "insert"){
                    auto dataArray = message.at(U("data")).as_array();
                    for(int i=0; i<dataArray.size(); i++){
			            json::value datum = dataArray[i]; // maybe not computationally optimal
                        if((datum[U("side")]).as_string() == "Sell"){
                            sellIds[U(datum[U("id")])] = json::value::array({datum[U("price")], datum[U("Size")]});       
                        }else{
                            buyIds[U(datum[U("id")])] = json::value::array({datum[U("price")], datum[U("Size")]});
                        }
                    }
                }else if((message[U("action")]).as_string() == "delete"){
                    auto dataArray = message.at(U("data")).as_array();
                    for(int i=0; i<dataArray.size(); i++){
			            json::value datum = dataArray[i]; // maybe not computationally optimal
                        if((datum[U("side")]).as_string() == "Sell"){
                            json::array::erase(sellIds.at(U(datum[U("id")])));
                        }else{
                            json::array::erase(buyIds.at(U(datum[U("id")])));
                        }
                    }
                }
                cout << "Indexed: " << message[U("data")] << "\n :) :) :) :) :) \n";
            }catch(const std::exception& e){
                cout << "Parsing Failed :()" << "\n";
            }

            oldMsgStrat = msg.as_string();
        }

}

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
        constructor(&msg);
	    cout << "buyIds: " << buyIds << "\n";
    
    }
}



