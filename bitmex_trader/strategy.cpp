#include <iostream>
#include "feed.h"
#include <unistd.h>

void Strategy(feed * self, std::string message){
    std::cout << "Halleleluyeigh: " << message << std::endl;
    sleep(10);
}
