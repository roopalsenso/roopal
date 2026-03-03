#include <iostream>
#include <fstream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"

using namespace nvinfer1;

// ================= CONFIG =================
const std::string ENGINE_PATH = "/root/old-data/home/roopal/engines/anomaly/anomalyint8_backbone.trt";
const std::string IMAGE_PATH  = "/root/old-data/home/roopal/datasets/MvTecAD/MvTecAD/OnlyOkbottles_test/images/contamination/006.png";

const int INPUT_C = 3;
const int INPUT_H = 224;
const int INPUT_W = 224;
const int INPUT_SIZE  = INPUT_C * INPUT_H * INPUT_W;
const int OUTPUT_SIZE = 512; // Hardcode based on your engine output size

const int WARMUP_RUNS  = 10;
const int MEASURE_RUNS = 50;

const std::vector<float> MEAN = {0.485f, 0.456f, 0.406f};
const std::vector<float> STD  = {0.229f, 0.224f, 0.225f};
// ==========================================

class Logger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} logger;

#define CHECK(status) do { auto ret = (status); if (ret != 0) { std::cerr << "CUDA Error: " << ret << "\n"; abort(); } } while(0)

int main() {
    // Load TensorRT engine
    std::ifstream file(ENGINE_PATH, std::ios::binary);
    if (!file) { std::cerr << "Engine file not found\n"; return -1; }
    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);
    std::vector<char> engine_data(size);
    file.read(engine_data.data(), size);
    file.close();

    IRuntime* runtime = createInferRuntime(logger);
    ICudaEngine* engine = runtime->deserializeCudaEngine(engine_data.data(), size);
    IExecutionContext* context = engine->createExecutionContext();

    // Allocate memory
    void* d_input; void* d_output;
    CHECK(cudaMalloc(&d_input, INPUT_SIZE * sizeof(float)));
    CHECK(cudaMalloc(&d_output, OUTPUT_SIZE * sizeof(float)));

    // Load image
    cv::Mat img = cv::imread(IMAGE_PATH);
    if (img.empty()) { std::cerr << "Could not read image\n"; return -1; }
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    cv::resize(img, img, cv::Size(INPUT_W, INPUT_H));
    cv::Mat img_float; img.convertTo(img_float, CV_32F, 1.0/255.0);

    std::vector<float> inputTensor(INPUT_SIZE);
    int img_area = INPUT_H * INPUT_W;
    for (int c = 0; c < INPUT_C; c++)
        for (int i = 0; i < INPUT_H; i++)
            for (int j = 0; j < INPUT_W; j++)
                inputTensor[c*img_area + i*INPUT_W + j] = (img_float.at<cv::Vec3f>(i,j)[c] - MEAN[c]) / STD[c];

    cudaStream_t stream; CHECK(cudaStreamCreate(&stream));
    std::vector<float> output(OUTPUT_SIZE);

    void* bindings[2] = {d_input, d_output};

    // Warmup
    for (int i = 0; i < WARMUP_RUNS; i++) {
        CHECK(cudaMemcpyAsync(d_input, inputTensor.data(), INPUT_SIZE*sizeof(float), cudaMemcpyHostToDevice, stream));
        context->executeV2(bindings);
        CHECK(cudaMemcpyAsync(output.data(), d_output, OUTPUT_SIZE*sizeof(float), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);
    }

    // Timing
    std::vector<float> gpu_times;
    std::vector<float> full_times;
    for (int i = 0; i < MEASURE_RUNS; i++) {
        cudaEvent_t start, end;
        cudaEventCreate(&start); cudaEventCreate(&end);

        // GPU only
        cudaEventRecord(start, stream);
        context->executeV2(bindings);
        cudaEventRecord(end, stream); cudaEventSynchronize(end);
        float ms = 0; cudaEventElapsedTime(&ms, start, end); gpu_times.push_back(ms);
        cudaEventDestroy(start); cudaEventDestroy(end);

        // Full pipeline
        cudaEventCreate(&start); cudaEventCreate(&end);
        cudaEventRecord(start, stream);
        CHECK(cudaMemcpyAsync(d_input, inputTensor.data(), INPUT_SIZE*sizeof(float), cudaMemcpyHostToDevice, stream));
        context->executeV2(bindings);
        CHECK(cudaMemcpyAsync(output.data(), d_output, OUTPUT_SIZE*sizeof(float), cudaMemcpyDeviceToHost, stream));
        cudaEventRecord(end, stream); cudaEventSynchronize(end);
        cudaEventElapsedTime(&ms, start, end); full_times.push_back(ms);
        cudaEventDestroy(start); cudaEventDestroy(end);
    }

    float gpu_avg = std::accumulate(gpu_times.begin(), gpu_times.end(), 0.0f)/gpu_times.size();
    float full_avg = std::accumulate(full_times.begin(), full_times.end(), 0.0f)/full_times.size();

    std::cout << "GPU Compute Time: " << gpu_avg << " ms\n";
    std::cout << "Full Pipeline Time: " << full_avg << " ms\n";

    // PatchCore-style max score
    float anomaly_score = *std::max_element(output.begin(), output.end());
    std::cout << "Image Anomaly Score: " << anomaly_score << "\n";

    // Cleanup
    cudaFree(d_input); cudaFree(d_output); cudaStreamDestroy(stream);

    return 0;
}