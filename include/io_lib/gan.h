#ifndef GAN_H
#define GAN_H

#include <torch/torch.h>
#include <vector>
#include <iostream>

class GAN {
public:
    GAN(int noise_dim = 100, int img_width = 224, int img_height = 224);
    std::vector<uint8_t> generate();
    double apply_loss(const torch::Tensor& loss);

private:
    struct GeneratorImpl : torch::nn::Module {
        torch::nn::ConvTranspose2d deconv1, deconv2;
        torch::nn::Linear fc;

        GeneratorImpl()
            : deconv1(torch::nn::ConvTranspose2dOptions(100, 64, 4).stride(1).padding(0)),
              deconv2(torch::nn::ConvTranspose2dOptions(64, 3, 4).stride(2).padding(1)),
              fc(torch::nn::Linear(256, 128)) {
            register_module("deconv1", deconv1);
            register_module("deconv2", deconv2);
            register_module("fc", fc);
        }

        torch::Tensor forward(torch::Tensor x) {
            x = torch::relu(fc->forward(x));
            x = x.view({x.size(0), 100, 1, 1});
            x = torch::relu(deconv1->forward(x));
            x = torch::tanh(deconv2->forward(x)); // Range [-1, 1]
            return x;
        }
    };
    TORCH_MODULE(Generator);

    Generator generator_;
    torch::optim::Adam generator_optimizer_;
    int noise_dim_;
    int img_width_;
    int img_height_;
};


#endif
