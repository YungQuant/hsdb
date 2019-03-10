#ifndef ENCODER_H
#define ENCODER_H

#include <Python.h>
#include <string>
#include <vector>

class encoder
{
    private:
        std::string key;
        std::string secret;  

    public:
        encoder(std::string key, std::string secret);

        std::string ws_sign();
        
};


#endif