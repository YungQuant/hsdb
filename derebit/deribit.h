#ifndef DERIBIT_H
#define DERIBIT_H

#include "wsocket.h"
#include "encoder.h"
#include "greeks/greeks.h"

#include <curl/curl.h>
#include <string>
#include <vector>
#include <future>


class deribit{

	private:
		std::string apikey;
		std::string apisecret;
		
		std::vector<std::vector<std::string>> blank_vec;
	
		std::string rest_url;
	
		encoder auth;
		
	public:
		deribit(std::string apikey, std::string apisec);
		
		greeks option;
		wsocket feed;

        CURL *curl = curl_easy_init();
	    CURLcode res;
		static size_t writecallback(void *contents, size_t size, size_t nmemb, void *userp);

		std::string Request(std::string VERB, std::string path, std::string body, std::vector<std::vector<std::string>> query);
	
	
	
};

#endif