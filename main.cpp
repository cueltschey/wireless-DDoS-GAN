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
    std::vector<std::pair<std::vector<uint8_t>, int>> dataset;
    int frame_index = 0;
    while (true) {
      frame_index++;
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
      dataset.push_back(std::pair<std::vector<uint8_t>, int>(frame_buffer, label));
      if (frame_index % 100 == 0) {
        std::cout << "Epoch: " << frame_index / 100 << std::endl;
        classifier.train(dataset);
        dataset.clear();
      }
    }

    classifier.save_model(output_filename);
    return 0;
  }

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> binary_dist(0, 1);

  while (true) {

    std::vector<uint8_t> frame;
    int prefix = binary_dist(gen);
    if (prefix == 0) {
      std::cout << "Sending random buffer" << std::endl;
      frame = io_lib::generate_random_buffer(200000); // Example size: 10 bytes
    } else {
      std::optional<std::vector<uint8_t>> opt_frame = input.read_frame();
      if(!opt_frame.has_value()){
        break;
      }
      frame = opt_frame.value();
      std::cout << "Sending image data: " << frame.size() << std::endl;
    }
    frame.insert(frame.begin(), prefix);

    ip_mgr.send(frame);
  }

  return 0;
}
