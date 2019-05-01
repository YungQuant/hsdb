
#include "./CLIENT/feed.h"

#include <iostream>
#include <string>
#include <vector>
#include <future>

#include <unistd.h>
#include <fstream>


int main()
{
    bool on = true;

    std::string key = "1qO8kPEtCpEReAhbGsdgmapK";
	std::string secret = "ia0apGZ3tevgVB1p9JaenBkovOHffgdviZOwRds0lOTVUpPc";
    
    

    std::cout << "Bitmex Trading System" << std::endl << std::endl;
    
    

    while(true){
        try {
            std::vector<std::future<void>> tasks;

            std::ofstream writer;

            feed bitmex(key, secret, on);
            std::cout << "Started: " << bitmex.trader.auth.nonce() << std::endl;
            //writer << "START: " << bitmex.trader.auth.nonce() << "\n";
            tasks = bitmex.start(writer);
            for(auto & t : tasks){
                t.get();
            }
            std::cout << "Restarting: " << bitmex.trader.auth.nonce() << std::endl;
            //writer << "RESTARTING: " << bitmex.trader.auth.nonce() << "\n";
            tasks.clear();
            sleep(15);
        } catch (...) {std::cout << "Restarting @ Main" << "\n";}

        sleep(15);
    } 

    
    return 0;
}
