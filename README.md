# HSDB

## Requirements

1. [C++ Rest SDK](https://github.com/Microsoft/cpprestsdk)

2. [CMake and Make](https://cmake.org/) (>=3.7)

## Installing Requirements

Just run the shell script in the root directory

```bash
sh prereqs.sh
```

## Branching

Make sure to create a branch when developing:

```bash
git checkout -b @name/feature
```

Then make a pull request when you want to merge to main dev.

## Building and Running

**1. Make the build directory**

```bash
mkdir build
cd build
```

**2. Run cmake and make:**

```bash
cmake ..
make
```

**3. Run the app:**

```bash
./hsdb
```

## Example Usage

### Make a HTTP(S) Request:

```c++
#include "./util/request.hpp"

web::http::http_response response = utility::request("GET", "http://url.com").get();
web::json::value value = response.extract_json().get();
```

### Listen to a WS(S) Server:

```c++
#include "./util/websocket.hpp"

web::websockets::client::websocket_callback_client wss = utility::websocket("wss://url.com).get();

wss.set_message_handler([=](web::websockets::client::websocket_incoming_message message) {
    std::string messageStr = message.extract_string().get();
});
```
