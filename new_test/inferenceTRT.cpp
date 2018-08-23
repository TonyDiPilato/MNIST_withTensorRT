/* A simple C++ program to import an UFF model, read an image in .pgm format and make fast inference on GPU with NVIDIA TensorRT */

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <unordered_map>
#include <cassert>
#include <vector>
#include "NvInfer.h"
#include "NvUffParser.h"

#include "NvUtils.h"

using namespace nvuffparser;
using namespace nvinfer1;
#include "common.h"


#define MAX_WORKSPACE (1 << 20)
const int maxBatchSize = 1;
static Logger gLogger; // object for warning and error reports

// Attributes of the model
static const int INPUT_H = 28;
static const int INPUT_W = 28;
static const int OUTPUT_SIZE = 10;
const char* INPUT_TENSOR_NAME = "X";
const char* OUTPUT_TENSOR_NAME = "Softmax/Softmax";
const std::string dir{"../"};
const std::string fileName{dir + "new_test/test_network.uff"};


int main(int argc, char** argv){

  std::cout << "*** FILE TO PROCESS: " << fileName << std::endl;

  int batchSize = 1;
  float ms;

  // *** IMPORTING THE MODEL *** 
  std::cout << "*** IMPORTING THE UFF MODEL ***" << std::endl;

  // Create the builder and the network
  IBuilder* builder = createInferBuilder(gLogger);
  INetworkDefinition* network = builder->createNetwork();

  // Create the UFF parser
  IUffParser* parser = createUffParser();
  assert(parser);

  // Declare the network inputs and outputs of the model to the parser
  parser->registerInput(INPUT_TENSOR_NAME, DimsCHW(1, 28, 28), UffInputOrder::kNCHW);
  parser->registerOutput(OUTPUT_TENSOR_NAME);

  // Parse the imported model to populate the network
  parser->parse(fileName.c_str(), *network, nvinfer1::DataType::kFLOAT);

  std::cout << "*** IMPORTING DONE ***" << std::endl; 


  // *** BUILDING THE ENGINE ***
  std::cout << "*** BUILDING THE ENGINE ***" << std::endl;
  
  //Build the engine using the builder object
  builder->setMaxBatchSize(maxBatchSize);
  builder->setMaxWorkspaceSize(1 << 30);
  //builder->setFp16Mode(true); //16-bit kernels are permitted
  ICudaEngine* engine = builder->buildCudaEngine(*network);
  //assert(engine);
  std::cout << "*** BUILDING DONE ***" << std::endl; 

  // Destroy network, builder and parser
  network->destroy();
  builder->destroy();
  parser->destroy();


  // *** SERIALIZE THE ENGINE HERE IF NEEDED FOR LATER USE ***


  // *** PERFORMING INFERENCE ***
  std::cout << "*** PERFORMING INFERENCE ***" << std::endl;

  // Create a context to store intermediate activation values
  IExecutionContext *context = engine->createExecutionContext();
  assert(context);

  // Create the input and the output buffers on Host
  float input[INPUT_H * INPUT_W];
  float output[OUTPUT_SIZE];

  // Create the buffer for the reading of the PGM file
  uint8_t fileData[INPUT_H * INPUT_W];

  // Read a random .pgm file
  srand(unsigned(time(nullptr)));
  const int num = rand() % 10;
  
  readPGMFile(dir + std::to_string(num) + ".pgm", fileData, INPUT_H, INPUT_W);
  std::cout << "\n\n*** IMAGE READ: " << num << ".pgm ***" << std::endl;
  
  // Initialize the input buffer
  for (int i = 0; i < INPUT_H * INPUT_W; i++)
    input[i] = 1.0 - float(fileData[i]) / 255.0;

  
  // Engine requires exactly IEngine::getNbBindings() number of buffers  
  int nbBindings = engine->getNbBindings();
  assert(nbBindings == 2);
  void* buffers[nbBindings];
  //std::vector<void*> buffers(nbBindings);

  const int inputIndex = engine->getBindingIndex(INPUT_TENSOR_NAME);
  const int outputIndex = engine->getBindingIndex(OUTPUT_TENSOR_NAME);


  // Create GPU buffers on device                                            
  CHECK(cudaMalloc(&buffers[inputIndex], batchSize * INPUT_H * INPUT_W * sizeof(float)));
  CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

  // Create stream                                                           
  cudaStream_t stream;
  CHECK(cudaStreamCreate(&stream));

  // Copy the data from host to device
  CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
  //CHECK(cudaMemcpy(buffers[inputIndex], input, batchSize * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice));
  
  // Enqueue the kernels on a CUDA stream (TensorRT execution is typically asynchronous)
  auto t_start = std::chrono::high_resolution_clock::now();
  context->enqueue(batchSize, buffers, stream, nullptr);
  //context->execute(batchSize, buffers);
  auto t_end = std::chrono::high_resolution_clock::now();
  ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();

  // Copy the data from device to host
  CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
  //CHECK(cudaMemcpy(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

  // Synchronize
  cudaStreamSynchronize(stream);

  // Release buffers
  cudaStreamDestroy(stream);
  CHECK(cudaFree(buffers[inputIndex]));
  CHECK(cudaFree(buffers[outputIndex]));


  // Destroy the context and the engine
  context->destroy();
  engine->destroy();

  // Print the time of execution and histogram of the output distribution     
  std::cout << "\n*** OUTPUT ***\n\n";
  std::cout << "Time to perform inference: " << ms << "ms\n" << std::endl;
  
  for (int i = 0; i < OUTPUT_SIZE; i++)
    std::cout << i << ": " << output[i] << "\n";

 std::cout << std::endl;

  shutdownProtobufLibrary();
  return 0;

}





