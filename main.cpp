#include <CLI/CLI.hpp>
#include <thread>
#include <chrono>
#include <cstring>
#include <fstream>
#include <optional>
#include <random>
#include "io_lib/input_buffer.h"
#include "io_lib/ip_manager.h"
#include "io_lib/image_classifier.h"
#include "io_lib/gan.h"


int main(int argc, char** argv) {
  io_lib::input_buffer_params input_params{};
  io_lib::ip_manager_params ip_params{};
  bool is_video = false;
  std::string output_filename;

  CLI::App app{
    "A traffic transmission and generation tool designed for testing wireless channels"
  };

  // IP manager parameters
  app.add_option("-a,--addr", ip_params.ip_addr, "IP address for connection")->required();
  app.add_option("-p,--port", ip_params.port, "Port for connection")->default_val(5201);
  app.add_flag("-s,--server", ip_params.is_server, "Run in server mode")->default_val(false);

  // Input parameters
  app.add_option("-f,--file-path", input_params.file_path, "Path to the input file");
  app.add_option("-o,--output-path", output_filename, "Path to the output file");
  app.add_option("-S,--frame-size", input_params.frame_size, "Frame size for input buffer")->default_val(1024);
  app.add_flag("-V,--video", is_video, "Use video data")->default_val(false);

  // Attempt to parse arguments
  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    std::exit(app.exit(e));
  }

  // determine input type
  if(is_video){
    input_params.itype = io_lib::input_type::VIDEO_FILE;
    if(input_params.file_path == ""){
      std::cerr << "File path required for video data, run with -f <file>" << std::endl;
      return -1;
    }
  } else if (input_params.file_path != ""){
    input_params.itype = io_lib::input_type::ALPHANUMERIC_FILE;
  } else {
    input_params.itype = io_lib::input_type::STDIN;
  }

  // NOTE: input buffer reads any data source in chuncks of frame_size
  io_lib::input_buffer input(input_params);
  // NOTE: ip manager sends and receives IP traffic
  io_lib::ip_manager ip_mgr(ip_params);



  if(ip_params.is_server){
    ImageClassifier  classifier;
    while (true) {
      std::optional<std::vector<uint8_t>> frame = ip_mgr.recv();
      if(!frame.has_value())
        break;
      std::vector<uint8_t> frame_buffer = frame.value();
      int label = static_cast<int>(frame_buffer.front());
      frame_buffer.erase(frame_buffer.begin());
      if (label > 1) {
        std::cout << "Invalid Label: " << label << std::endl;
        label = 0;
      }
      double loss = classifier.train(frame_buffer, label);

      std::cout << "Classifier Loss: " << 0.4 - loss << std::endl;
      std::vector<uint8_t> loss_buffer(sizeof(double));
      std::memcpy(loss_buffer.data(), &loss, sizeof(double));
      ip_mgr.send(loss_buffer);
    }

    classifier.save_model(output_filename);
    return 0;
  }

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> binary_dist(0, 1);

  GAN generator;

  while (true) {

    std::vector<uint8_t> frame;
    int prefix = binary_dist(gen);
    if (prefix == 0) {
      frame = generator.generate();
      std::cout << "GAN: " << frame.size() << std::endl;
    } else {
      std::optional<std::vector<uint8_t>> opt_frame = input.read_frame();
      if(!opt_frame.has_value()){
        break;
      }
      frame = opt_frame.value();
      std::cout << "File: " << frame.size() << std::endl;
    }
    frame.insert(frame.begin(), prefix);

    ip_mgr.send(frame);

    // receive the loss
    std::optional<std::vector<uint8_t>> loss_buffer = ip_mgr.recv();
    if(!loss_buffer.has_value())
      continue;

    if (loss_buffer.value().size() != sizeof(double)) {
      throw std::runtime_error("Buffer size does not match double size.");
    }

    double loss;
    std::memcpy(&loss, loss_buffer.value().data(), sizeof(double));
    loss = 1.0 - loss;

    if(prefix == 0){
      std::cout << "Generator Loss: " << loss << std::endl;
      //generator.apply_loss(loss);
    }
  }

  return 0;
}
