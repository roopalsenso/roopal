#include <iostream>
#include <fstream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <filesystem>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"

using namespace nvinfer1;
namespace fs = std::filesystem;

// ================= CONFIG =================
const std::string ENGINE_PATH =
"/root/old-data/home/roopal/engines/512_2_cls_talbros/clsint8.trt";

const std::string IMAGE_PATH =
"/root/old-data/home/roopal/datasets/512_2_clstalbros_tets_color/CAM_02_NG25_left.jpg";

const std::string SAVE_DIR =
"/root/old-data/home/roopal/inference/results/int8trt_nocalib/cpp/512cls_talbros";

const int INPUT_H = 512;
const int INPUT_W = 512;

const int WARMUP_RUNS  = 10;
const int MEASURE_RUNS = 50;
// ==========================================

class Logger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} logger;

#define CHECK(status)                                   \
    do {                                                \
        auto ret = (status);                            \
        if (ret != 0) {                                 \
            std::cerr << "CUDA Error: " << ret << "\n"; \
            abort();                                    \
        }                                               \
    } while (0)

int main() {

    fs::create_directories(SAVE_DIR);

    // ========= LOAD ENGINE =========
    std::ifstream file(ENGINE_PATH, std::ios::binary);
    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);

    std::vector<char> engine_data(size);
    file.read(engine_data.data(), size);
    file.close();

    IRuntime* runtime = createInferRuntime(logger);
    ICudaEngine* engine =
        runtime->deserializeCudaEngine(engine_data.data(), size);
    IExecutionContext* context =
        engine->createExecutionContext();

    const char* inputName  = engine->getIOTensorName(0);
    const char* outputName = engine->getIOTensorName(1);

    Dims inputDims  = engine->getTensorShape(inputName);
    Dims outputDims = engine->getTensorShape(outputName);

    std::cout << "Input  : " << inputName << "\n";
    std::cout << "Output : " << outputName << "\n";

    int inputSize = 1;
    for (int i = 0; i < inputDims.nbDims; i++)
        inputSize *= inputDims.d[i];

    int outputSize = 1;
    for (int i = 0; i < outputDims.nbDims; i++)
        outputSize *= outputDims.d[i];

    int num_classes = outputDims.d[1];  // [1, C]

    // ========= IMAGE PREPROCESS =========
    cv::Mat img = cv::imread(IMAGE_PATH);
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    cv::resize(img, img, cv::Size(INPUT_W, INPUT_H));

    cv::Mat img_float;
    img.convertTo(img_float, CV_32F, 1.0 / 255.0);

    std::vector<float> inputTensor(inputSize);

    int channels = 3;
    int img_area = INPUT_H * INPUT_W;

    for (int c = 0; c < channels; c++) {
        for (int i = 0; i < INPUT_H; i++) {
            for (int j = 0; j < INPUT_W; j++) {
                inputTensor[c * img_area + i * INPUT_W + j] =
                    img_float.at<cv::Vec3f>(i, j)[c];
            }
        }
    }

    // ========= CUDA MEMORY =========
    void* d_input;
    void* d_output;

    CHECK(cudaMalloc(&d_input, inputSize * sizeof(float)));
    CHECK(cudaMalloc(&d_output, outputSize * sizeof(float)));

    context->setTensorAddress(inputName, d_input);
    context->setTensorAddress(outputName, d_output);

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    std::vector<float> output(outputSize);

    // ========= WARMUP =========
    std::cout << "Warming up...\n";
    for (int i = 0; i < WARMUP_RUNS; i++) {

        CHECK(cudaMemcpyAsync(d_input, inputTensor.data(),
              inputSize * sizeof(float),
              cudaMemcpyHostToDevice, stream));

        context->enqueueV3(stream);

        CHECK(cudaMemcpyAsync(output.data(), d_output,
              outputSize * sizeof(float),
              cudaMemcpyDeviceToHost, stream));

        cudaStreamSynchronize(stream);
    }

    // ========= TIMING =========
    std::vector<float> gpu_times;
    std::vector<float> full_times;

    for (int i = 0; i < MEASURE_RUNS; i++) {

        cudaEvent_t start, end;
        cudaEventCreate(&start);
        cudaEventCreate(&end);

        // GPU only
        cudaEventRecord(start, stream);
        context->enqueueV3(stream);
        cudaEventRecord(end, stream);
        cudaEventSynchronize(end);

        float ms = 0;
        cudaEventElapsedTime(&ms, start, end);
        gpu_times.push_back(ms);

        cudaEventDestroy(start);
        cudaEventDestroy(end);

        // Full pipeline
        cudaEventCreate(&start);
        cudaEventCreate(&end);

        cudaEventRecord(start, stream);

        CHECK(cudaMemcpyAsync(d_input, inputTensor.data(),
              inputSize * sizeof(float),
              cudaMemcpyHostToDevice, stream));

        context->enqueueV3(stream);

        CHECK(cudaMemcpyAsync(output.data(), d_output,
              outputSize * sizeof(float),
              cudaMemcpyDeviceToHost, stream));

        cudaEventRecord(end, stream);
        cudaEventSynchronize(end);

        cudaEventElapsedTime(&ms, start, end);
        full_times.push_back(ms);

        cudaEventDestroy(start);
        cudaEventDestroy(end);
    }

    float gpu_avg =
        std::accumulate(gpu_times.begin(), gpu_times.end(), 0.0f)
        / gpu_times.size();

    float full_avg =
        std::accumulate(full_times.begin(), full_times.end(), 0.0f)
        / full_times.size();

    std::cout << "\n===== PERFORMANCE SUMMARY =====\n";
    std::cout << "GPU Compute Time  : " << gpu_avg << " ms\n";
    std::cout << "Full Pipeline Time: " << full_avg << " ms\n";
    std::cout << "FPS (compute only): " << 1000.0f / gpu_avg << "\n";
    std::cout << "================================\n";

    // ========= POSTPROCESS (CLASSIFICATION) =========
    int predicted_class = 0;
    float max_prob = output[0];

    for (int i = 1; i < num_classes; i++) {
        if (output[i] > max_prob) {
            max_prob = output[i];
            predicted_class = i;
        }
    }

    std::cout << "\n===== PREDICTION =====\n";
    std::cout << "Predicted Class : " << predicted_class << "\n";
    std::cout << "Confidence      : " << max_prob << "\n";
    std::cout << "======================\n";

    return 0;
}