#include <opencv2/opencv.hpp>
#include "io_lib/gan.h"

// Learning Rate
const double kLr = 2e-4;

// Beta1
const double kBeta1 = 0.5;

// Beta2
const double kBeta2 = 0.999;


GAN::GAN(int noise_dim, int img_width, int img_height)
    : generator_(Generator()), 
      generator_optimizer_(generator_->parameters(), torch::optim::AdamOptions(kLr).betas(std::make_tuple(kBeta1, kBeta2))),
      noise_dim_(noise_dim),
      img_width_(img_width),
      img_height_(img_height) {
      torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
      generator_->to(device);
      torch::load(generator_, "../generator-checkpoint.pt");
      torch::load(generator_optimizer_, "../generator-optimizer-checkpoint.pt");
}

std::vector<uint8_t> GAN::generate() {
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    //generator_->eval();

    auto noise = torch::randn({ 64, noise_dim_, 1, 1 }, device);

    // NOTE: noise works
    //auto tensor_cpu = torch::randn({ 1, 3, 64, 64 }, device).to(device).to(torch::kUInt8).contiguous();

    auto fake_image = generator_->forward(noise);
    auto tensor_cpu = fake_image.to(device).to(torch::kUInt8).contiguous();

    std::vector<uint8_t> image_data(tensor_cpu.numel());
    std::memcpy(image_data.data(), tensor_cpu.data_ptr<uint8_t>(), tensor_cpu.numel());

    return image_data;
}

double GAN::apply_loss(double loss_value) {
    generator_->train();

    // Convert the provided double loss value into a scalar tensor
    torch::Tensor loss_tensor = torch::tensor(loss_value, torch::requires_grad(true));
    torch::Tensor fake_labels = torch::zeros(1, torch::kCPU).fill_(1);

    torch::Tensor g_loss = torch::binary_cross_entropy(loss_tensor,fake_labels);

    // Backpropagation and optimization
    //generator_->zero_grad();
    g_loss.backward();
    generator_optimizer_.step();

    return loss_tensor.item<double>();
}
