#include "bmxsocket.h"

#include <iostream>
#include <string>
#include <vector>
#include <future>
#include <thread>
#include <cpprest/ws_client.h>
#include "encrypt++/encoder.h"

using namespace web;
using namespace web::websockets::client;


bmxsocket::bmxsocket(std::string key, std::string secret)
{
	apikey = key;
	apisec = secret;
}


void sock_init(std::string k, std::string s)
{
	encoder auth(k, s);

	std::string nonce = auth.nonce();
	
	std::string body = "GET/realtime" + nonce;
	
	std::string signature = auth.sha256_hmac(body);
	std::string url = "wss://www.bitmex.com/realtime?api-expires=" + nonce + "&api-signature=" + signature + "&api-key=" + k;

	websocket_client client;
	client.connect(url).wait();	

	client.receive().then([](websocket_incoming_message in_msg)
	{
		return in_msg.extract_string();
	}).then([](std::string bodyy)
	{
		std::cout << bodyy << std::endl << std::endl;
	}).wait();

	std::string msg = "{\"op\":\"subscribe\",\"args\":\"position\"}";

	std::cout << msg << std::endl << std::endl;

	websocket_outgoing_message out_msg;
	out_msg.set_utf8_message(msg);
	client.send(out_msg).wait();
	
	msg = "{\"op\":\"subscribe\",\"args\":\"quote:XBTUSD\"}";

	out_msg.set_utf8_message(msg);
	client.send(out_msg).wait();

	while(true){
		client.receive().then([](websocket_incoming_message in_msg)
		{
			return in_msg.extract_string();
		}).then([](std::string bodys)
		{
			std::cout << bodys << std::endl << std::endl;
		}).wait();
	}

	client.close().wait();

}



std::future<void> bmxsocket::WebSocketApp()
{
	std::future<void> result(std::async(sock_init, apikey, apisec));

	std::cout << "FAO sucks dude" << std::endl;

	return result;
}
