#include <vector>
#include <cstdint>
#include <optional>
#include <fstream>
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

namespace io_lib {

typedef enum {
  STDIN,
  VIDEO_FILE,
  ALPHANUMERIC_FILE,
} input_type;

typedef struct {
  input_type itype;
  uint16_t frame_size;
  std::string file_path;
} input_buffer_params;

  
class input_buffer {
public:
  input_buffer(const input_buffer_params& p);
  ~input_buffer();
  std::optional<std::vector<uint8_t>> read_frame();
  void reset_buffer();

private:
  input_buffer_params params;
  std::ifstream file_stream;
  cv::VideoCapture video_capture;
  std::vector<uint8_t> buffer;
};

}
