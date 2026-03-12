#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <filesystem>
#include <vector>
#include <numeric>
#include <chrono>

namespace fs = std::filesystem;

// ================= CONFIG =================
const std::string MODEL_PATH =
"/root/old-data/home/roopal/models/onnx/512_1_seg_semsons_mode_512_color.onnx";

const std::string IMAGE_PATH =
"/root/old-data/home/roopal/datasets/test_offline_dataset/images/2.jpg";

const std::string SAVE_DIR =
"/root/old-data/home/roopal/inference/results/cpp_onnx_opencv";

const int INPUT_H = 512;
const int INPUT_W = 512;
const float CONF_THRESHOLD = 0.5;
const int WARMUP_RUNS  = 5;
const int MEASURE_RUNS = 50;
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
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);

    std::cout << "Model loaded successfully!\n";

    // Warmup
    cv::Mat output;
    std::cout << "Running warmup...\n";
    for (int i = 0; i < WARMUP_RUNS; i++) {
        cv::Mat input_blob = cv::dnn::blobFromImage(img, 1.0/255.0, cv::Size(INPUT_W, INPUT_H));
        net.setInput(input_blob);
        output = net.forward();
    }

    // Timing vectors
    std::vector<double> gpu_times;
    std::vector<double> full_times;

    for (int i = 0; i < MEASURE_RUNS; i++) {
        // Full pipeline timing
        auto t_start = std::chrono::high_resolution_clock::now();

        cv::Mat input_blob = cv::dnn::blobFromImage(img, 1.0/255.0, cv::Size(INPUT_W, INPUT_H));
        net.setInput(input_blob);

        // GPU compute timing only
        auto t_gpu_start = std::chrono::high_resolution_clock::now();
        output = net.forward();
        auto t_gpu_end = std::chrono::high_resolution_clock::now();

        auto t_end = std::chrono::high_resolution_clock::now();

        double gpu_ms  = std::chrono::duration<double, std::milli>(t_gpu_end - t_gpu_start).count();
        double full_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

        gpu_times.push_back(gpu_ms);
        full_times.push_back(full_ms);
    }

    // Compute averages
    double avg_gpu  = std::accumulate(gpu_times.begin(), gpu_times.end(), 0.0) / gpu_times.size();
    double avg_full = std::accumulate(full_times.begin(), full_times.end(), 0.0) / full_times.size();

    std::cout << "\n===== PERFORMANCE SUMMARY =====\n";
    std::cout << "GPU Compute Time  : " << avg_gpu << " ms\n";
    std::cout << "Full Pipeline Time: " << avg_full << " ms\n";
    std::cout << "FPS (compute only): " << 1000.0 / avg_gpu << "\n";
    std::cout << "================================\n";

    // Postprocess
    cv::Mat mask_mat = output.reshape(1, INPUT_H);  // single channel
    cv::resize(mask_mat, mask_mat, img.size(), 0, 0, cv::INTER_NEAREST);

    cv::Mat mask_bin;
    cv::threshold(mask_mat, mask_bin, CONF_THRESHOLD, 255, cv::THRESH_BINARY);
    mask_bin.convertTo(mask_bin, CV_8U);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask_bin, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Mat overlay = original.clone();
    cv::drawContours(overlay, contours, -1, cv::Scalar(0, 0, 255), 2);

    fs::create_directories(SAVE_DIR);
    std::string base = fs::path(IMAGE_PATH).stem().string();

    cv::imwrite(SAVE_DIR + "/" + base + "_mask.jpg", mask_bin);
    cv::imwrite(SAVE_DIR + "/" + base + "_overlay.jpg", overlay);

    std::cout << "Saved mask to: " << SAVE_DIR + "/" + base + "_mask.jpg\n";
    std::cout << "Saved overlay to: " << SAVE_DIR + "/" + base + "_overlay.jpg\n";

    return 0;
}