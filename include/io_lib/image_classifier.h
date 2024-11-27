#ifndef IMAGE_CLASSIFIER_H
#define IMAGE_CLASSIFIER_H

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <stdexcept>

class ImageClassifier {
public:
    // Constructor
    ImageClassifier(const std::string& model_path = "", int img_width = 224, int img_height = 224);

    // Preprocess a frame for inference
    torch::Tensor preprocess(std::vector<uint8_t>& frame);

    // Train the model
    void train(std::vector<std::pair<std::vector<uint8_t>, int>>& dataset, int epochs = 10, double lr = 0.01);

    // Classify a single frame
    bool classify(std::vector<uint8_t>& frame);

    // Save the trained model
    void save_model(const std::string& path);

    // Load a saved model
    void load_model(const std::string& path);

private:
    // Initialize the model
    void initialize_model();

    // Internal CNN model definition
    struct ClassifierImpl : torch::nn::Module {
        torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
        torch::nn::Linear fc1{nullptr}, fc2{nullptr};

        ClassifierImpl();
        torch::Tensor forward(torch::Tensor x);
    };
    TORCH_MODULE(Classifier);

    int img_width_, img_height_;
    Classifier model_;
};

#endif // IMAGE_CLASSIFIER_H

