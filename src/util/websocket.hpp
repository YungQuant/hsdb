#ifndef _WEBSOCKET_H_
#define _WEBSOCKET_H_ 1

#include <cpprest/ws_client.h>
#include <cpprest/json.h>

namespace utility {

    pplx::task<web::websockets::client::websocket_callback_client> websocket(std::string socketUri) {
        return pplx::task<web::websockets::client::websocket_callback_client>([=] {
            web::websockets::client::websocket_callback_client newSocket;

            try {
                newSocket.connect(socketUri).wait();
            } catch (std::exception& e) {
                std::cout << e.what() << std::endl;
            }

            return newSocket;
        });
    }

}

#endif
