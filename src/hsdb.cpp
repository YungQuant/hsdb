#include <string>
#include <fstream>
#include "./util/request.hpp"
#include "./util/websocket.hpp"

//using namespace utility;

int main(int argc, char** argv) {
    std::cout << "Hello world, ready to have your markets made?\n";
    
    pplx::task<web::websockets::client::websocket_callback_client> sock = websocket("www.bitmex.com/realtime");


    websocket_outgoing_message msg;
    msg = "{\"op\": \"subscribe\", \"args\": [\"orderBookL2_25:XBTUSD\"]}";
    std:cout << "Sending:" << msg << "\n";
    msg.set_utf8_message(msg);
    client.send(msg);

	client.set_message_handler([](websocket_incoming_message msg)
	{
		std:cout << "Received: " << msg.extract_string() << "\n";
		ofstream fP;
		fP.open("log.txt");
		fP << msg.extract_string() << "\n";
		fP.close(); 
		
	});
	
	
}
