#include "encoder.h"

#include <string>
#include <sstream>
#include <iomanip>
#include <vector>
#include <time.h>
#include <cctype>


#include <cryptopp/hmac.h>
#include <cryptopp/osrng.h>
#include <cryptopp/hex.h>
#include <cryptopp/base64.h>


encoder::encoder(std::string apikey, std::string apisecret)
{
    key = apikey;
    secret = apisecret;
}

std::string encoder::nonce()
{
	time_t time1;
	time(&time1);
	time1 *= 1000;
    std::stringstream ss;
	ss << time1;
	std::string timestamp = ss.str();
	return timestamp;
}

std::string encoder::sha256_hmac(std::string post)
{
	std::string mac;
	std::string result;

	CryptoPP::SecByteBlock byteKey((const byte*)secret.c_str(), secret.size());
 	CryptoPP::HMAC<CryptoPP::SHA256> hmac(byteKey, byteKey.size());
	CryptoPP::StringSource ss1(post, true, new CryptoPP::HashFilter(hmac, new CryptoPP::StringSink(mac)));
	CryptoPP::StringSource ss2(mac, true, new CryptoPP::HexEncoder(new CryptoPP::StringSink(result)));
	std::transform(result.begin(), result.end(), result.begin(), ::tolower);

	return result;
}

std::string encoder::url_encode(const std::string &value) {
    std::ostringstream escaped;
    escaped.fill('0');
    escaped << std::hex;

    for (std::string::const_iterator i = value.begin(), n = value.end(); i != n; ++i) {
        std::string::value_type c = (*i);

        if (isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
            escaped << c;
            continue;
        }

        escaped << std::uppercase;
        escaped << '%' << std::setw(2) << int((unsigned char) c);
        escaped << std::nouppercase;
    }

    return escaped.str();
}

std::string encoder::sock_auth(std::string body)
{
	std::string timestamp = nonce();
	body += timestamp;
	std::string sig = sha256_hmac(body);
	std::string result = "?api-expires=" + timestamp + "&api-signature=" + sig + "&api-key=" + key;
	return result;
}
