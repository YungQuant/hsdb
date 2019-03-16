#ifndef FEED_H
#define FEED_H

#include <string>
#include <vector>
#include <future>


class feed {

    private:
        std::string key;
        std::string secret;

    public:
        feed(std::string k, std::string s);

        std::string feed_url;
        std::vector<std::future<void>> socket();

};

#endif
