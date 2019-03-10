#ifndef WSOCKET_H
#define WSOCKET_H

#include "encoder.h"
#include "database.h"

#include <string>
#include <future>

class wsocket{
	
	private:
		std::string key;
		std::string secret;
	
		std::string ws_url;
	
		encoder auth;
		
	
	public:
		wsocket(std::string apikey, std::string apisec);
		database data;
	
		std::string socket_msg;
	
		std::future<void> start();
};



#endif