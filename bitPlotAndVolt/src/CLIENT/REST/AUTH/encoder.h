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
        std::string numstr(double x);
        std::string wrap_query(const std::string &value);
        std::string build_msg(std::vector<std::vector<std::string>> params);
        
        std::string nonce();
        double curr_timestamp();
        
        std::string HMACSHA256(std::string post);
		std::string sock_auth(std::string body);

        std::string __ws__();
        std::vector<std::string> rest_auth(std::string verb, std::string path, std::string body);

};


#endif
