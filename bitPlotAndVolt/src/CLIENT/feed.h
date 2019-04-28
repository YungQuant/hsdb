#ifndef FEED_H
#define FEED_H

#include "./REST/bitmex.h"
#include "./REST/DATA/data.h"
#include <string>
#include <vector>
#include <future>
#include <fstream>


class feed {

    private:
        bool is_on;

    public:
        feed(std::string key, std::string secret, bool ON_SWITCH, int depth, int limit);

        bitmex trader;
        data quant;

        bool sync;

        std::string socket_msg;
        std::string feed_url;
        std::vector<std::future<void>> start(std::ofstream & writer);

};

#endif
