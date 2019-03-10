#include "deribit.h"
#include "encoder.h"

#include <iostream>
#include <vector>
#include <string>
#include <future>
#include <unordered_map>


auto printv = [](std::vector<std::string> data){
	for(auto & t : data)
	{
		std::cout << t << "\t";	
	}
};


int main()
{
	std::string Key = "4KVqcSPakgsq2";
	std::string Secret = "4TCG3TZBS2CZD24ZNOTEE43MW326XFG2";
	
	deribit client(Key, Secret);
	
	std::future<void> hold = client.feed.start();
	
	std::cout << "Socket has opened" << std::endl;
	
	hold.get();
	
	return 0;	
}
