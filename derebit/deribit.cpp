#include "deribit.h"

#include <curl/curl.h>

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>

#include <iostream>

deribit::deribit(std::string key, std::string secret):feed(key, secret),auth(key, secret),option(300, 0.0000001)
{
	apikey = key;	
	apisecret = secret;
	
	blank_vec = {{""}};
}

std::string deribit::Request(std::string VERB, std::string path, std::string body, std::vector<std::vector<std::string>> query)
{
	std::string result;
	/*
	std::string timestamp = auth.nonce();

	if(query != blank_vec)
	{
		//path = urlComp(path, query);
	}

	std::string payload = VERB + "/api/v1" + path + timestamp + body;
	//std::string signature = auth.sha256HMAC(payload); Does not work
	
	
	
	struct curl_slist *httpHeaders = NULL;
	httpHeaders = curl_slist_append(httpHeaders, ("api-nonce: " + timestamp).c_str());
	httpHeaders = curl_slist_append(httpHeaders, ("api-key: " + apikey).c_str());
	httpHeaders = curl_slist_append(httpHeaders, ("api-signature: " + signature).c_str());
	httpHeaders = curl_slist_append(httpHeaders, "User-Agent:api");
	httpHeaders = curl_slist_append(httpHeaders, "Content-Type: Application/JSON");
	httpHeaders = curl_slist_append(httpHeaders, ("Content-Length: " + intstr(body.size())).c_str());

	std::string send_url = url + path;

	if(VERB=="DELETE"){
		curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "DELETE");
		curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body.c_str());
	}

	curl_easy_setopt(curl, CURLOPT_URL, send_url.c_str());

	if(VERB=="POST")
	{
		curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, NULL);
		curl_easy_setopt(curl, CURLOPT_POST, 1L);
		curl_easy_setopt(curl, CURLOPT_HTTPGET, 0);
		curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body.c_str());
	}
	else if (VERB=="GET")
	{
		curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, NULL);
		curl_easy_setopt(curl, CURLOPT_HTTPGET, 1L);
		curl_easy_setopt(curl, CURLOPT_POST, 0);
	}

	curl_easy_setopt(curl, CURLOPT_HTTPHEADER, httpHeaders);
	curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writecallback);
	curl_easy_setopt(curl, CURLOPT_WRITEDATA, &result);

	res = curl_easy_perform(curl);
	curl_slist_free_all(httpHeaders);
	
	*/
	
	return result;
}