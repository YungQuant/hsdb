
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

    std::vector<std::future<void>> tasks;

    std::cout << "Bitmex Trading System" << std::endl << std::endl;

    std::ofstream writer;
    writer.open("log.txt");

    while(true){
        feed bitmex(key, secret, on, 20, 60);
        std::cout << "Started: " << bitmex.trader.auth.nonce() << std::endl;
        writer << "START: " << bitmex.trader.auth.nonce() << "\n";
        tasks = bitmex.start(writer);
        for(auto & t : tasks){
            t.get();
        }
        std::cout << "Restarting: " << bitmex.trader.auth.nonce() << std::endl;
        writer << "RESTARTING: " << bitmex.trader.auth.nonce() << "\n";
        tasks.clear();
        sleep(10);
    }

    writer.close();
    return 0;
}