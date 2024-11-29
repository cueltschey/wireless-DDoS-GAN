#include "io_lib/gan.h"

GAN::GAN(int noise_dim, int img_width, int img_height)
    : generator_(Generator()), 
      generator_optimizer_(generator_->parameters(), torch::optim::AdamOptions(0.0002)),
      noise_dim_(noise_dim),
      img_width_(img_width),
      img_height_(img_height) {
}

std::vector<uint8_t> GAN::generate() {
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    generator_->to(device);
    generator_->eval();

    // Generate random noise
    auto noise = torch::randn({ 224, noise_dim_, 1, 1 }, device);

    // Forward pass through the generator
    auto fake_image_tensor = generator_->forward(noise);

    // Rescale the output to [0, 255] and clamp
    fake_image_tensor = fake_image_tensor.clamp(0, 1).mul(255).to(device).to(torch::kUInt8);

    // Convert the tensor to std::vector<uint8_t>
    auto fake_image = fake_image_tensor.permute({0, 2, 3, 1}).contiguous(); // CHW -> HWC
    std::vector<uint8_t> image_data(fake_image.data_ptr<uint8_t>(),
                                    fake_image.data_ptr<uint8_t>() + fake_image.numel());

    return image_data;
}

double GAN::apply_loss(double loss_value) {
    generator_->train();

    // Convert the provided double loss value into a scalar tensor
    torch::Tensor loss = torch::tensor(loss_value, torch::requires_grad(true));

    // Backpropagation and optimization
    generator_optimizer_.zero_grad();
    loss.backward(); // Backpropagate the scalar tensor loss
    generator_optimizer_.step();

    return loss.item<double>();
}
