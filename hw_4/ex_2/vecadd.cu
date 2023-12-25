#include <iostream>
#include <random>
#include <chrono>
#include <string>

#define DataType double
#define TPB 32
#define VEC_RANGE 1e10

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len) {
  //@@ Insert code to implement vector addition here
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= len) return;
  out[i] = in1[i] + in2[i];
}

//@@ Insert code to implement timer
class Timer{
  std::chrono::time_point<std::chrono::high_resolution_clock> start;

public:
  Timer(){
    start = std::chrono::high_resolution_clock::now();
  }

  double get(){
    auto stop = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(stop - start).count();
  }
};

int main(int argc, char **argv) {
  cudaFree(0);
  
  int inputLength;
  DataType *hostInput1;
  DataType *hostInput2;
  DataType *hostOutput;
  DataType *resultRef;
  DataType *deviceInput1;
  DataType *deviceInput2;
  DataType *deviceOutput;

  //@@ Insert code below to read in inputLength from args
  if(argc > 1) inputLength = std::stoi(argv[1]);
  else { 
    std::cout << "Input length was not provided";
    return;
  }

  printf("The input length is %d\n", inputLength);
  
  //@@ Insert code below to allocate Host memory for input and output
  cudaHostAlloc(&hostInput1, sizeof(DataType) * inputLength, cudaHostAllocDefault);
  cudaHostAlloc(&hostInput2, sizeof(DataType) * inputLength, cudaHostAllocDefault);
  cudaHostAlloc(&hostOutput, sizeof(DataType) * inputLength, cudaHostAllocDefault);
  
  //@@ Insert code below to initialize hostInput1 and hostInput2 to random numbers, and create reference result in CPU
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<DataType> dis(-VEC_RANGE, VEC_RANGE);
  for(int i = 0; i < inputLength; i++){
    hostInput1[i] = dis(gen);
    hostInput2[i] = dis(gen);
  }

  // calc reference result
  Timer cpuTimer;
  resultRef = (DataType*) malloc(sizeof(DataType) * inputLength);
  for(int i = 0; i < inputLength; i++) 
    resultRef[i] = hostInput1[i] + hostInput2[i];
  printf("CPU time: %fs\n", cpuTimer.get());

  Timer gpuTimer;
  //@@ Insert code below to allocate GPU memory here
  Timer gpuMallocTimer;
  cudaMalloc(&deviceInput1, sizeof(DataType) * inputLength);
  cudaMalloc(&deviceInput2, sizeof(DataType) * inputLength);
  cudaMalloc(&deviceOutput, sizeof(DataType) * inputLength);
  printf("GPU malloc time: %fs\n", gpuMallocTimer.get());

  //@@ Insert code to below to Copy memory to the GPU here
  Timer h2dTimer;
  cudaMemcpy(deviceInput1, hostInput1, sizeof(DataType) * inputLength, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceInput2, hostInput2, sizeof(DataType) * inputLength, cudaMemcpyHostToDevice);
  printf("Host to device copy time: %fs\n", h2dTimer.get());

  //@@ Initialize the 1D grid and block dimensions here
  dim3 blockSize(TPB);
  dim3 gridSize((inputLength + TPB - 1)/TPB);

  //@@ Launch the GPU Kernel here
  Timer kernelTimer;
  vecAdd<<<gridSize,blockSize>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);
  cudaDeviceSynchronize();
  printf("Kernel time: %fs\n", kernelTimer.get());

  //@@ Copy the GPU memory back to the CPU here
  Timer d2hTimer;
  cudaMemcpy(hostOutput, deviceOutput, sizeof(DataType) * inputLength, cudaMemcpyDeviceToHost);
  printf("Device to host copy time: %fs\n", d2hTimer.get());
  
  printf("GPU time: %fs\n", gpuTimer.get());

  //@@ Insert code below to compare the output with the reference
  printf("VERIFYING\n");
  bool isCorrect = true;
  for(int i = 0; i < inputLength; i++) {
    if(resultRef[i] != hostOutput[i]) { 
      isCorrect = false;
      break;
    }
  }

  if (isCorrect)
    printf("Correct\n");
  else
    printf("CPU and GPU result differ\n");

  //@@ Free the GPU memory here
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);

  //@@ Free the CPU memory here
  cudaFreeHost(hostInput1);
  cudaFreeHost(hostInput2);
  cudaFreeHost(hostOutput);

  return 0;
}
