# HSDB

## Requirements

1. [C++ Rest SDK](https://github.com/Microsoft/cpprestsdk)

2. [CMake and Make](https://cmake.org/) (>=3.7)

3. [Crypto++](https://github.com/weidai11/cryptopp)

4. [Curl](https://github.com/curl/curl)

## Installing Requirements

If you do not have C++ Rest SDK or Crypto++ installed right now.

```bash
# Install C++ Rest SDK
sh bin/install.cpprestsdk.sh

# Install Crypto++
sh bin/install.cryptopp.sh

# Install curl
sh bin/install.curl.sh
```

*Please note you might need to instal `libtool`, `autoconf` and `automake` first in order to install curl.*

## Dockerized Envrionments

**TBD: Will implement if problems with environments still occurring.**

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
# Your binary should be the project you're working on based on below configurable cmake builds
./your-binary
```

## Custom CMAKE builds

**To run `/src/bitmex`:**

```bash
cmake .. -DBITMEX=ON
make
# Run bitmex binary
./bitmex
```

**To run `/src/examples`:**

```bash
cmake .. -DEXAMPLES=ON
make
# Run examples binary
./hsdb
```
