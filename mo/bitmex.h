#ifndef BITMEX_H
#define BITMEX_H

#include "encrypt++/encoder.h"
#include <curl/curl.h>
#include <string>
#include <vector>
#include <future>


class bitmex
{
	private:
		encoder auth;		
		
		std::string apikey;
		std::string apisecret;
		std::string url;
		std::string sock_url;

		int twin;

		std::vector<std::vector<std::string>> blank_vec;

	public:
		bitmex(std::string key, std::string secret, int window);

        CURL *curl = curl_easy_init();
	    CURLcode res;
		static size_t writecallback(void *contents, size_t size, size_t nmemb, void *userp);

		std::string socket_msg;
		
		double bid;
		double ask;
		double last;
		double rsi;
		bool price_sync;
		bool rsi_sync;
		std::vector<double> store_last;

		std::string urlComp(std::string base, std::vector<std::vector<std::string>> params);
		std::string Request(std::string VERB, std::string path, std::string body, std::vector<std::vector<std::string>> query);

		std::string limit_buy(std::string symbol, double price, int quantity);
		std::string limit_sell(std::string symbol, double price, int quantity);
		std::string bulk_order(std::string symbol, std::vector<double> prices, std::vector<int> qty, std::string side);
		
		std::string leverage(std::string symbol, double leverage);

		std::string cancel_order(std::string orderId);
		std::string cancel_all(std::string symbol, std::string side);
		
		std::string chat(std::string message);
		std::string account(std::string symbol);
		std::string withdraw(std::string symbol, std::string address, double amount);

		std::future<void> socket(std::string symbol);		
		static void sock_init(std::string &socket_msg, std::string symbol, std::string url);


};


#endif
