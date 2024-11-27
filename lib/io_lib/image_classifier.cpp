#include "io_lib/image_classifier.h"

// Constructor
ImageClassifier::ImageClassifier(const std::string& model_path, int img_width, int img_height)
    : img_width_(img_width), img_height_(img_height), model_(nullptr) {
    if (!model_path.empty()) {
        load_model(model_path);
    } else {
        initialize_model();
    }
}

// Preprocess a frame for inference
torch::Tensor ImageClassifier::preprocess(std::vector<uint8_t>& frame) {
    cv::Mat image;

    // Try to decode the image
    image = cv::imdecode(frame, cv::IMREAD_COLOR);

    // If decoding fails, create an image from the raw buffer
    if (image.empty()) {
        int required_size = img_height_ * img_width_ * 3;
        if (frame.size() < required_size) {
            throw std::runtime_error("Insufficient bytes for raw image creation");
        }

        // Create a CV_8UC3 Mat from the buffer
        image = cv::Mat(img_height_, img_width_, CV_8UC3, frame.data()).clone();
    } else {
        // Resize the decoded image to the target size
        cv::resize(image, image, cv::Size(img_width_, img_height_));
    }

    // Convert the OpenCV Mat to a PyTorch Tensor
    auto tensor = torch::from_blob(image.data, {1, img_height_, img_width_, 3}, torch::kUInt8)
                      .permute({0, 3, 1, 2}) // Convert from HWC to CHW
                      .to(torch::kFloat32)
                      .div_(255.0);         // Normalize to [0, 1]

    // NOTE: to display images
    // cv::imshow("Recv", image);
    // cv::waitKey(0);
    // cv::destroyWindow("Recv");

    return tensor;
}

// Train the model
double ImageClassifier::train(std::vector<uint8_t> frame, int label, double lr) {
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    model_->to(device);
    torch::optim::SGD optimizer(model_->parameters(), lr);

    model_->train();

    auto input_tensor = preprocess(frame).to(device);
    size_t expected_size = img_width_ * img_height_ * 3;
    if (frame.size() < expected_size) {
      return 1.0;
    }
    auto label_tensor = torch::tensor({label}, torch::kInt64).to(device);

    optimizer.zero_grad();
    auto output = model_->forward(input_tensor);
    auto loss = torch::nll_loss(output, label_tensor);

    loss.backward();
    optimizer.step();
    return loss.item<double>();
}

// Classify a single frame
bool ImageClassifier::classify(std::vector<uint8_t>& frame) {
    torch::NoGradGuard no_grad;
    model_->eval();

    auto input_tensor = preprocess(frame).to(torch::kCUDA);
    auto output = model_->forward(input_tensor);
    auto prediction = output.argmax(1).item<int>();
    return prediction == 1;
}

// Save the trained model
void ImageClassifier::save_model(const std::string& path) {
    torch::save(model_, path);
    std::cout << "Model saved to " << path << std::endl;
}

// Load a saved model
void ImageClassifier::load_model(const std::string& path) {
    model_ = Classifier();
    torch::load(model_, path);
    std::cout << "Model loaded from " << path << std::endl;
}

// Initialize the model
void ImageClassifier::initialize_model() {
    model_ = Classifier();
}

// Internal CNN model definition
ImageClassifier::ClassifierImpl::ClassifierImpl() {
    conv1 = register_module("conv1", torch::nn::Conv2d(3, 16, 3));
    conv2 = register_module("conv2", torch::nn::Conv2d(16, 32, 3));
    fc1 = register_module("fc1", torch::nn::Linear(32 * 54 * 54, 128));
    fc2 = register_module("fc2", torch::nn::Linear(128, 2));
}

torch::Tensor ImageClassifier::ClassifierImpl::forward(torch::Tensor x) {
  x = torch::relu(conv1->forward(x));
  x = torch::max_pool2d(x, 2);
  x = torch::relu(conv2->forward(x));
  x = torch::max_pool2d(x, 2);
  x = x.contiguous();  // Ensure the tensor is contiguous in memory
  x = x.view({x.size(0), -1});  // Flatten the tensor
  x = torch::relu(fc1->forward(x));
  x = fc2->forward(x);
  return torch::log_softmax(x, 1);
}

