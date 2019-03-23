#include "shared.hpp"

int main() {
    std::cout << "Starting bitmex trader...\n";

    std::string key = "P10jnGlMWgOur_ZlOoiH8_Uk";
	std::string secret = "p0dA-oQ4JuqrER77J7GReEoE3dBm7UmMtR08asXa1UVYqpSP";

    hsdb::feed::feed(key, secret);

    for(auto & t : hsdb::feed::start()){
        t.get();
    }


    return 0;
}
