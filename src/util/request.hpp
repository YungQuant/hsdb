#ifndef _REQUEST_H_
#define _REQUEST_H_ 1

#include <cpprest/http_client.h>
#include <cpprest/json.h>

namespace utility {

        web::http::client::http_client_config requestConfig(web::json::value configParams) {
            web::http::client::http_client_config config;

            if (configParams["credentials"].is_object()) {
                web::http::client::credentials credentials;
                std::string username = configParams["credentials"]["username"].as_string();
                std::string password = configParams["credentials"]["password"].as_string();

                credentials = web::http::client::credentials(username, password);
                config.set_credentials(credentials);
            }

            if (configParams["timeout"].is_integer()) {
                std::chrono::seconds timeout(configParams["timeout"].as_integer());
                config.set_timeout(timeout);
            }

            return config;
        }

        pplx::task<web::http::http_response> request(web::http::method method, std::string url, web::json::value body, web::json::value configParams) {
            return pplx::task<web::http::http_response>([=] {
                try {
                    web::http::client::http_client_config config = requestConfig(configParams);
                    web::http::client::http_client client(url, config);

                    web::http::http_response response = client.request(method).get();
                    return response;
                } catch (std::exception& e) {
                    std::cout << e.what() << std::endl;

                    web::http::http_response response;
                    response.set_status_code(500);
                    response.set_reason_phrase("Something went wrong creating the request");

                    return response;
                }
            });
        }

        pplx::task<web::http::http_response> request(web::http::method method, std::string url, web::json::value body) {
            return pplx::task<web::http::http_response>([=] {
                try {
                    web::http::client::http_client client(url);
                    web::http::http_response response = client.request(method).get();
                    return response;
                } catch (std::exception& e) {
                    std::cout << e.what() << std::endl;

                    web::http::http_response response;
                    response.set_status_code(500);
                    response.set_reason_phrase("Something went wrong creating the request");

                    return response;
                }
            });
        }

        pplx::task<web::http::http_response> request(web::http::method method, std::string url) {
            return pplx::task<web::http::http_response>([=] {
                try {
                    web::http::client::http_client client(url);
                    web::http::http_response response = client.request(method).get();
                    return response;
                } catch (std::exception& e) {
                    std::cout << e.what() << std::endl;

                    web::http::http_response response;
                    response.set_status_code(500);
                    response.set_reason_phrase("Something went wrong creating the request");

                    return response;
                }
            });
        }

}

#endif
