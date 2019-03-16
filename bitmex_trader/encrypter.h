#ifndef ENCRYPTER_H
#define ENCRYPTER_H

#include <cryptopp/hmac.h>
#include <cryptopp/osrng.h>
#include <cryptopp/hex.h>
#include <cryptopp/base64.h>

#include <string>
#include <sstream>
#include <iomanip>
#include <vector>
#include <time.h>
#include <cctype>

namespace encrypter {

    namespace signature {

        long nonce(){
            time_t t1;
            time(&t1);
            t1 *= 1000;
            return t1;
        }

        std::string num2str(long t1)
        {
            std::stringstream ss;
            ss << t1;
            return ss.str();
        }

        std::string HMACSHA256(std::string secret, std::string nonce)
        {
        	std::string mac;
        	std::string result;
            std::string post = "GET/realtime" + nonce;

        	CryptoPP::SecByteBlock byteKey((const byte*)secret.c_str(), secret.size());
         	CryptoPP::HMAC<CryptoPP::SHA256> hmac(byteKey, byteKey.size());
        	CryptoPP::StringSource ss1(post, true, new CryptoPP::HashFilter(hmac, new CryptoPP::StringSink(mac)));
        	CryptoPP::StringSource ss2(mac, true, new CryptoPP::HexEncoder(new CryptoPP::StringSink(result)));
        	std::transform(result.begin(), result.end(), result.begin(), ::tolower);

        	return result;
        }

    }

    std::string __ws__(std::string k, std::string s){
        std::string nonce = signature::num2str(signature::nonce());
        std::string res = "{\"op\": \"authKeyExpires\", \"args\": [\"" + k + "\"," + nonce + ",\"" + signature::HMACSHA256(s, nonce) + "\"]}";
        return res;
    }





}


#endif
