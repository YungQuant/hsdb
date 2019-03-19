#ifndef ENCODER_H
#define ENCODER_H

#include <string>
#include <vector>

class encoder
{
    private:
        std::string key;
        std::string secret;

    public:
        encoder(std::string key, std::string secret);

        std::string intstr(int x);
        std::string nonce();
        std::string HMACSHA256(std::string post);
		std::string url_encode(const std::string &value);
		std::string sock_auth(std::string body);

        std::string __ws__();
        std::vector<std::string> rest_auth(std::string verb, std::string path, std::string body);

};


#endif
