#ifndef FEED_H
#define FEED_H

#include "bitmex.h"
#include "data.h"
#include <string>
#include <vector>
#include <future>

class feed {

    public:
        feed(std::string key, std::string secret);

        bitmex trader;
        data quant;

        bool sync;

        std::string socket_msg;

        std::string feed_url;
        std::vector<std::future<void>> start();

};

#endif
