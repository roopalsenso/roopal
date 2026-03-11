#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

// ================= CONFIG =================
const std::string MODEL_PATH =
"/root/old-data/home/qualviz-microservice/media/users/vignesh/projects/semsons__ed5a22e8-521c-4b7d-8188-cc6f7c85b44e/steps/S1/models/mode_512__0ebd2f0b-2e3d-4d14-9dd7-e113a9287015/best_model/512_2_semsons_mode_512_color.onnx";

const std::string IMAGE_PATH =
"/root/old-data/home/roopal/datasets/test_offline_dataset/images/2.jpg";

const std::string SAVE_DIR =
"/root/old-data/home/roopal/inference/results/cpp_onnx_opencv";

const int INPUT_H = 512;
const int INPUT_W = 512;
const float CONF_THRESHOLD = 0.5;
// ==========================================

int main() {
    // Load image
    cv::Mat img = cv::imread(IMAGE_PATH);
    if (img.empty()) {
        std::cerr << "ERROR: Cannot load image: " << IMAGE_PATH << "\n";
        return -1;
    }

    cv::Mat original = img.clone();

    // Load ONNX model
    cv::dnn::Net net = cv::dnn::readNetFromONNX(MODEL_PATH);

    // Enable CUDA if available
     // Print OpenCV build information
     std::cout << cv::getBuildInformation() << std::endl;
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);

    std::cout << "Model loaded successfully!\n";

    // Preprocess: blob from image
    cv::Mat input_blob = cv::dnn::blobFromImage(img, 1.0/255.0, cv::Size(INPUT_W, INPUT_H), cv::Scalar(), true, false);
    net.setInput(input_blob);
    cv::Mat output = net.forward();

    // For single class mask
    cv::Mat mask_mat(INPUT_H, INPUT_W, CV_32F, output.ptr<float>());
    cv::resize(mask_mat, mask_mat, img.size(), 0, 0, cv::INTER_NEAREST);

    // Forward pass
    output = net.forward();

    // Output shape: [1, 1, H, W] for single-class
    // Reshape to 2D: H x W
    cv::Mat mask = output.reshape(1, INPUT_H);  // single channel

    // Resize mask to original image size
    cv::resize(mask, mask, original.size(), 0, 0, cv::INTER_NEAREST);

    // Threshold to get binary mask
    cv::Mat mask_bin;
    cv::threshold(mask, mask_bin, CONF_THRESHOLD, 255, cv::THRESH_BINARY);
    mask_bin.convertTo(mask_bin, CV_8U);

    // Find contours and overlay
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask_bin, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Mat overlay = original.clone();
    cv::drawContours(overlay, contours, -1, cv::Scalar(0, 0, 255), 2);

    // Create save directory
    fs::create_directories(SAVE_DIR);
    std::string base = fs::path(IMAGE_PATH).stem().string();

    // Save mask and overlay
    cv::imwrite(SAVE_DIR + "/" + base + "_mask.jpg", mask_bin);
    cv::imwrite(SAVE_DIR + "/" + base + "_overlay.jpg", overlay);

    std::cout << "Saved mask to: " << SAVE_DIR + "/" + base + "_mask.jpg\n";
    std::cout << "Saved overlay to: " << SAVE_DIR + "/" + base + "_overlay.jpg\n";

    return 0;
}