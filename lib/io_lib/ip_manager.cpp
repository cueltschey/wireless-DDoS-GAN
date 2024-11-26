#include "io_lib/ip_manager.h"

namespace io_lib {

ip_manager::ip_manager(const ip_manager_params& p) :
      params(p), io_context(), socket(io_context),
      ip_endpoint(boost::asio::ip::make_address(p.ip_addr), p.port) {

  // TODO: add custom errors
  if (params.is_server) {
      std::cout << "ip_manager: Server Started... " << std::endl;
      acceptor = std::make_unique<boost::asio::ip::tcp::acceptor>(
          io_context, ip_endpoint);
      acceptor->accept(socket);
      std::cout << "Connection Received" << std::endl;
  } else {
      std::cout << "ip_manager: Client Started... " << std::endl;
      socket.connect(ip_endpoint);
      std::cout << "Connection Received" << std::endl;
  }
}

size_t ip_manager::send(const std::vector<uint8_t>& buffer) {
    uint32_t data_size = buffer.size();
    uint32_t data_size_net = htonl(data_size);

    // Send the data size
    boost::asio::write(socket, boost::asio::buffer(&data_size_net, sizeof(data_size_net)));

    // Send the entire buffer at once
    boost::asio::write(socket, boost::asio::buffer(buffer));

    return buffer.size();
}


std::optional<std::vector<uint8_t>> ip_manager::recv() {
    try {
      uint32_t data_size_net = 0;

      // Receive the size of the incoming data
      boost::asio::read(socket, boost::asio::buffer(&data_size_net, sizeof(data_size_net)));
      uint32_t data_size = ntohl(data_size_net);

      // Allocate buffer for the entire data
      std::vector<uint8_t> buffer(data_size);

      // Receive the entire data at once
      boost::asio::read(socket, boost::asio::buffer(buffer));
      return buffer;
    }
    catch (const boost::wrapexcept<boost::system::system_error>&) {
      return std::nullopt;
    }
}

}
