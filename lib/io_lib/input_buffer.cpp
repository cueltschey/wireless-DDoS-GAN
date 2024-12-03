#include "io_lib/input_buffer.h"
#include <random>

namespace io_lib {

std::vector<uint8_t> generate_random_buffer(size_t size) {
  std::vector<uint8_t> buffer(size);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint8_t> dis(0, 255);

  std::generate(buffer.begin(), buffer.end(), [&]() { return dis(gen); });
  return buffer;
}

input_buffer::input_buffer(const input_buffer_params& p): params(p) {
  switch (params.itype) {
    case STDIN:
      break;
    case VIDEO_FILE:
      video_capture.open(params.file_path);
      if (!video_capture.isOpened()) {
          throw std::runtime_error("input_buffer: Failed to open video file: " + params.file_path);
      }
      break;
    case ALPHANUMERIC_FILE:
      file_stream.open(params.file_path, std::ios::binary);
      if (!file_stream.is_open()) {
          throw std::runtime_error("input_buffer: Failed to open alphanumeric file: " + params.file_path);
      }
      break;
    default:
      throw std::invalid_argument("input_buffer: Unsupported input type");  }
}

std::optional<std::vector<uint8_t>> input_buffer::read_frame() {
  switch (params.itype) {
  case STDIN: {
    buffer.resize(params.frame_size);
    std::cin.read(reinterpret_cast<char*>(buffer.data()), params.frame_size);
    if (std::cin.gcount() == 0) {
        return std::nullopt; // End of input
    }
    buffer.resize(std::cin.gcount()); // Adjust size to read bytes
    return buffer;
  }
  case VIDEO_FILE: {
    cv::Mat frame;
    if (!video_capture.read(frame)) {
        video_capture.set(cv::CAP_PROP_POS_FRAMES, 0);  // Reset the video to the first frame
        video_capture.read(frame);
    }
    cv::imencode(".jpg", frame, buffer);
    return buffer;
  }
  case ALPHANUMERIC_FILE: {
      buffer.resize(params.frame_size);
      file_stream.read(reinterpret_cast<char*>(buffer.data()), params.frame_size);
      if (file_stream.gcount() == 0) {
          return std::nullopt; // End of file
      }
      buffer.resize(file_stream.gcount()); // Adjust size to read bytes
      return buffer;
    }
    default:
      throw std::logic_error("Unsupported input type in read_frame");
  }
}

void input_buffer::reset_buffer() {
  buffer.clear();
  if (params.itype == VIDEO_FILE) {
    video_capture.set(cv::CAP_PROP_POS_FRAMES, 0); // Reset video to first frame
  } else if (params.itype == ALPHANUMERIC_FILE) {
    file_stream.clear(); // Clear EOF or error flags
    file_stream.seekg(0, std::ios::beg); // Reset file stream to beginning
  }
}

input_buffer::~input_buffer() {
  if (file_stream.is_open()) {
      file_stream.close();
  }
  if (video_capture.isOpened()) {
      video_capture.release();
  }
}

}
