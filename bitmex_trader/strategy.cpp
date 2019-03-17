#include <iostream>
#include "feed.h"
#include <unistd.h>

using namespace std;

void Strategy(feed * self){
    while(true){
        cout << "Socket Message: " << (*self).socket_msg << endl << endl;
    }
}
