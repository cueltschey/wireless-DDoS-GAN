#ifndef IP_MANAGER_H
#define IP_MANAGER_H

#include <iostream>
#include <vector>
#include <cstring>
#include <string>
#include <optional>
#include <boost/asio.hpp>

namespace io_lib {

typedef struct {
  std::string ip_addr;
  uint16_t port;
  bool is_server;
} ip_manager_params;

class ip_manager {
    public:
        ip_manager(const ip_manager_params& p);

        size_t send(const std::vector<uint8_t>& buffer);

        std::optional<std::vector<uint8_t>> recv();

    private:
        ip_manager_params params;

        // Boost IP
        boost::asio::io_context io_context;
        boost::asio::ip::tcp::socket socket;
        boost::asio::ip::tcp::endpoint ip_endpoint;
        std::unique_ptr<boost::asio::ip::tcp::acceptor> acceptor;
    };
}

#endif // !IP_MANAGER_H

