
#include "feed.h"
#include <iostream>
#include <string>
#include <vector>



int main()
{
    std::string key = "P10jnGlMWgOur_ZlOoiH8_Uk";
	std::string secret = "p0dA-oQ4JuqrER77J7GReEoE3dBm7UmMtR08asXa1UVYqpSP";
    feed bitmex(key, secret);

    for(auto & t : bitmex.start()){
        t.get();
    }


    return 0;
}
