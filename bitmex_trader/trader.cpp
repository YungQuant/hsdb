#include "feed.h"
#include <iostream>
#include <string>
#include <vector>
#include <future>


int main()
{
    std::string key = "l_OGpLlK63FsA__81_232atG";
	std::string secret = "wuBQ-cFVsEp6zgOcRHegeXC7LC_mV4v7K0zhZ1NeqRhfAw5d";
    feed client(key, secret);

    std::vector<std::future<void>> tasks = client.socket();

    for(auto & t : tasks){
        t.get();
    }

    return 0;
}
