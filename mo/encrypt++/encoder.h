#ifndef ENCODER_H
#define ENCODER_H

#include <string>


class encoder
{
    private:
        std::string key;
        std::string secret; 

    public:
        encoder(std::string key, std::string secret);

        std::string nonce();
        std::string sha256_hmac(std::string post);
		std::string url_encode(const std::string &value);
		std::string sock_auth(std::string body);

};


#endif
