#ifndef BITMEX_H
#define BITMEX_H

#include "./AUTH/encoder.h"

#include <curl/curl.h>
#include <string>
#include <vector>
#include <future>


class bitmex
{
	private:
		std::string url;

	public:
		bitmex(std::string key, std::string secret);

		encoder auth;

        CURL *curl = curl_easy_init();
	    CURLcode res;
		static size_t writecallback(void *contents, size_t size, size_t nmemb, void *userp);

		std::string Request(std::string VERB, std::string path, std::string body, std::string query);

		std::string limit_buy(std::string symbol, double price, int quantity);
		std::string limit_sell(std::string symbol, double price, int quantity);
		std::string bulk_order(std::string symbol, std::vector<double> prices, std::vector<int> qty, std::string side);

		std::string leverage(std::string symbol, double leverage);

		std::string cancel_order(std::string orderId);
		std::string cancel_all(std::string symbol, std::string side);
	
		std::string chat(std::string message);
		std::string account(std::string symbol);
		std::string position(std::string symbol);
		std::string withdraw(std::string symbol, std::string address, double amount);


};


#endif
