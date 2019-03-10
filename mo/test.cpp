#include "bitmex.h"
#include "compute.h"
#include <iostream>
#include <string>
#include <future>


/*
	Compiling Instructions 

		> g++ test.cpp bitmex.cpp compute.cpp encrypt++/encoder.cpp -l:libcryptopp.a -lcurl -lpthread -lcpprest

*/

int main()
{
	std::string key = "cWP7GiLCEYL6yyJrseMToiIe";
	std::string secret = "bj_7gzbxwPJAowDJm5MmVZJQ3XItF7kgiUu6vJjRS1g_XtCv";


	bitmex client(key, secret, 20);
	
	std::future<void> hold = client.socket("XBTUSD");

	
	
	hold.get();

	return 0;
}



