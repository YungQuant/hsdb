#include "client_wss.hpp"
#include "server_wss.hpp"
#include "json/json.h"

using namespace std;

using WssClient = SimpleWeb::SocketClient<SimpleWeb::WSS>;

int i = 0;

int main() {
  // Example 4: Client communication with server
  // Second Client() parameter set to false: no certificate verification
  // Possible output:
  //   Server: Opened connection 0x7fcf21600380
  //   Client: Opened connection
  //   Client: Sending message: "Hello"
  //   Server: Message received: "Hello" from 0x7fcf21600380
  //   Server: Sending message "Hello" to 0x7fcf21600380
  //   Client: Message received: "Hello"
  //   Client: Sending close connection
  //   Server: Closed connection 0x7fcf21600380 with status code 1000
  //   Client: Closed connection with status code 1000

    //WssClient client("wss:99.84.216.17/realtime", false);
    WssClient client("www.bitmex.com/realtime/", false); 

  client.on_message = [](shared_ptr<WssClient::Connection> connection, shared_ptr<WssClient::Message> message) {
    cout << "Client: Message " << i << " received: \"" << message->string().c_str() << "\"\ni\n" << endl;

    Json::CharReaderBuilder builder;
    Json::CharReader* prs = builder.newCharReader();
    Json::Value msg;
    std::string errs;
    bool prs_success = prs->parse(message->string().c_str(), message->string().c_str() + message->string().size(), &msg, &errs);
    delete prs;
    if(prs_success){
        cout << "Client: Message " << i << " parsed. DATA: \"" << msg["data"] << "\"\ni\n" << endl;
        cout << errs << "PRS_SUCCESS\n" << endl;
        }
    else {cout << errs << "PRS_FAIL\n" << endl;}
    
    i++;
    if(i > 100){
        cout << "Client: Sending close connection\n" << endl;
        connection->send_close(1000);
		//delete i;
        }
    };

  client.on_open = [](shared_ptr<WssClient::Connection> connection) {
    cout << "Client: Opened connection\n" << endl;
 	auto send_stream = make_shared<WssClient::SendStream>();
 
    //string message = "{\"message\": \"Help\"}";
    string message = "\"help\"";
    cout << "Client: Sending message: \"" << message << "\"\n" << endl;
    *send_stream << message;
    connection->send(send_stream);  

	
	message = "{\"op\": \"subscribe\", \"args\": [\"orderBookL2_25:XBTUSD\"]}";
    //string message = "\"help\"";
    cout << "Client: Sending message: \"" << message << "\"\n" << endl;
    *send_stream << message;
    connection->send(send_stream);  
	
	
	/*string message = "{\"message\": \"Help\"}";
    string message = "\"help\"";
    cout << "Client: Sending message: \"" << message << "\"" << endl;
    *send_stream << message;
    connection->send(send_stream);
    */
	

  };

  client.on_close = [](shared_ptr<WssClient::Connection> /*connection*/, int status, const string & /*reason*/) {
    cout << "Client: Closed connection with status code " << status << endl;
  };

  // See http://www.boost.org/doc/libs/1_55_0/doc/html/boost_asio/reference.html, Error Codes for error code meanings
  client.on_error = [](shared_ptr<WssClient::Connection> /*connection*/, const SimpleWeb::error_code &ec) {
    cout << "Client: Error: " << ec << ", error message: " << ec.message() << endl;
  };

  client.start();
}
