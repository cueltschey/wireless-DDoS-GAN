#ifndef GAN_H
#define GAN_H

#include <torch/torch.h>
#include <vector>
#include <iostream>

class GAN {
public:
    GAN(int noise_dim = 256, int img_width = 224, int img_height = 224);
    std::vector<uint8_t> generate();
    double apply_loss(double loss_value);

private:
    struct GeneratorImpl : torch::nn::Module {
        GeneratorImpl()
            : conv1(torch::nn::ConvTranspose2dOptions(256, 512, 4)
                .stride(1)
                .padding(0)
                .bias(false)),
            batch_norm1(512),
            conv2(torch::nn::ConvTranspose2dOptions(512, 256, 4)
                .stride(2)
                .padding(1)
                .bias(false)),
            batch_norm2(256),
            conv3(torch::nn::ConvTranspose2dOptions(256, 128, 4)
                .stride(2)
                .padding(1)
                .bias(false)),
            batch_norm3(128),
            conv4(torch::nn::ConvTranspose2dOptions(128, 64, 4)
                .stride(2)
                .padding(1)
                .bias(false)),
            batch_norm4(64),
            conv5(torch::nn::ConvTranspose2dOptions(64, 3, 4)
                .stride(2)
                .padding(1)
                .bias(false))
        {
            register_module("conv1", conv1);
            register_module("conv2", conv2);
            register_module("conv3", conv3);
            register_module("conv4", conv4);
            register_module("conv5", conv5);
            register_module("batch_norm1", batch_norm1);
            register_module("batch_norm2", batch_norm2);
            register_module("batch_norm3", batch_norm3);
            register_module("batch_norm4", batch_norm4);
        }

        torch::Tensor forward(torch::Tensor x) {
            x = relu(batch_norm1(conv1(x)));
            x = relu(batch_norm2(conv2(x)));
            x = relu(batch_norm3(conv3(x)));
            x = relu(batch_norm4(conv4(x)));
            x = tanh(conv5(x));
            return x;
        }

        torch::nn::ConvTranspose2d conv1, conv2, conv3, conv4, conv5;
        torch::nn::BatchNorm2d batch_norm1, batch_norm2, batch_norm3, batch_norm4;
    };
    TORCH_MODULE(Generator);

    Generator generator_;
    torch::optim::Adam generator_optimizer_;
    int noise_dim_;
    int img_width_;
    int img_height_;
};


#endif
