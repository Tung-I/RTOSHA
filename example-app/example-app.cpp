#include <torch/script.h> 
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
 
#include <chrono>
using namespace std::chrono;

torch::Device device(torch::kCUDA);

int main(int argc, const char* argv[]) {
    if (argc != 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
    }

    // Load the TorchScript model
    torch::jit::script::Module module;
    try {
    module = torch::jit::load(argv[1]);
    }
    catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
    }
    std::cout << "Model loaded successfully\n";

    // Load a sample image
    std::string img_path = "/home/ubuntu/RTOSHA/example-app/frame_0000.png";

    cv::Mat img;
    img = cv::imread(img_path, cv::IMREAD_COLOR);
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    img.convertTo(img, CV_32FC3, 1.0f / 255.0f);
    img = (img - 0.5) / 0.5;


    // Create a vector to hold the image.
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::from_blob(img.data, {1, img.rows, img.cols, 3}).permute({0, 3, 1, 2}).to(device));
    std::cout << "Input tensor shape: " << inputs[0].toTensor().sizes() << std::endl;

    
    // Run the model
    auto start = high_resolution_clock::now();
    auto stop = high_resolution_clock::now();
    // Warm up the model
    for(int i = 0; i < 20; i++){
        start = high_resolution_clock::now();
        at::Tensor output = module.forward(inputs).toTensor();
        stop = high_resolution_clock::now();
        std::cout << "Inference time: " << duration_cast<milliseconds>(stop - start).count() << "ms" << std::endl;
    }
    // Run the model
    at::Tensor output = module.forward(inputs).toTensor();

    // Ensure the tensor is contiguous and moved to CPU
    auto output_cpu = output.to(torch::kCPU).contiguous();
    // Print the shape of output tensor
    std::cout << "Output tensor shape: " << output_cpu.sizes() << std::endl;
    // Print the min and max value of the output tensor
    std::cout << "Min value: " << output_cpu.min().item<float>() << std::endl;
    std::cout << "Max value: " << output_cpu.max().item<float>() << std::endl;
    // Make output tensor to be binary
    output_cpu = (output_cpu > 0.5).to(torch::kFloat32);

    // Extract the output tensor to a OpenCV Mat, where the image is in gray scale with a shape (1, 1, H, W)
    cv::Mat output_img(cv::Size(output_cpu.size(3), output_cpu.size(2)), CV_32FC1, output_cpu.data_ptr<float>());

    // Normalize and convert back to OpenCV
    output_img = output_img * 255.0;
    output_img.convertTo(output_img, CV_8UC1);
    cv::imwrite("/home/ubuntu/RTOSHA/example-app/output.png", output_img);


}