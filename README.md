# 5G data streaming tool
A simple tool for sending a variety of data types over a wireless channel

## Capabilites
- throughput measurement
- normal / poisson distribution generated data streams
- serve many ports
- concurrent data streams
- DoS simulation
- UDP and TCP traffic
- ZMQ transmission
- UHD transmission
- generative AI

## dependencies
CMake
OpenCV
Boost

## Installing binary
```bash
sudo dpkg -i <(curl -sSL https://github.com/cueltschey/wl-stream/releases/latest)
```

## Compiling from source
```bash
mkdir -p build
cd build
cmake ..
make -j$(nproc)
sudo make install
```
## Example Usage
