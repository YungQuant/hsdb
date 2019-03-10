#include "wsocket.h"
#include "database.h"

#include <cpprest/ws_client.h>
#include <future>
#include <sstream>

#include <iostream>


using namespace web;
using namespace web::websockets::client;

wsocket::wsocket(std::string apikey, std::string apisec):auth(apikey, apisec),data()
{
	key = apikey;
	secret = apisec;
	
	ws_url = "wss://www.deribit.com/ws/api/v1/";
	
}

void websocketx(database data, std::string &socket_msg, std::string url, std::string server_msg)
{
	std::string result = "Socket";
	websocket_client client; 
	client.connect(url).wait();
	
	websocket_outgoing_message send_msg;
	send_msg.set_utf8_message(server_msg);
	client.send(send_msg).wait();
	
	std::future<void> dthd;
	
	while(true){
		client.receive().then([](websocket_incoming_message in_msg)
		{
			return in_msg.extract_string();
		}).then([&](std::string socket_msgs)
		{
			socket_msg = socket_msgs;
		}).wait();
		
		if(data.check_msg(socket_msg) == true){
			dthd = data.push_msg(socket_msg);
			dthd.get();
		}
	}
	
	
	client.close().wait();	

}


std::future<void> wsocket::start()
{
	std::string final_sig = auth.ws_sign();
	std::future<void> result(std::async(websocketx, data, std::ref(socket_msg), ws_url, final_sig));
	return result;
}