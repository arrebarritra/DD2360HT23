#include <iostream>
#include <random>
#include <chrono>
#include <string>
#include <limits>

#define DataType double
#define TPBD 16
#define VEC_RANGE 1000

// Compute C = A * B
__global__ void gemm(DataType *A, DataType *B, DataType *C, int numARows,
                      int numAColumns, int numBRows, int numBColumns){
  //@@ Insert code to implement matrix multiplication here
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;
  if(i >= numARows) return;
  if(j >= numBColumns) return;

  C[numBColumns * i + j] = 0.0;
  for(int k = 0; k < numAColumns; k++) {
    C[numBColumns * i + j] += A[numAColumns * i + k] * B[numBColumns * k + j];
  }
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
  cudaFree(0); // initialize CUDA context

  DataType *hostA; // The A matrix
  DataType *hostB; // The B matrix
  DataType *hostC; // The output C matrix
  DataType *resultRef; // The reference result
  DataType *deviceA;
  DataType *deviceB;
  DataType *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;
  int numCColumns;

  //@@ Insert code below to read in numARows, numAColumns, numBColumns from args
  if(argc > 3){
    numARows = std::stoi(argv[1]);
    numAColumns = std::stoi(argv[2]);
    numBColumns = std::stoi(argv[3]);
  } else {
    std::cout << "Matrix dimensions were not provided";
    return;
  }
  numBRows = numAColumns;
  numCRows = numARows;
  numCColumns = numBColumns;

  printf("Input matrix dim (%d x %d) (%d x %d) (%d x %d)\n", numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
  
  //@@ Insert code below to allocate Host memory for input and output
  hostA = (DataType*) malloc(sizeof(DataType) * numARows * numAColumns);
  hostB = (DataType*) malloc(sizeof(DataType) * numBRows * numBColumns);
  hostC = (DataType*) malloc(sizeof(DataType) * numCRows * numCColumns);
  
  //@@ Insert code below to initialize hostA and hostB to random numbers, and create reference result in CPU
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<DataType> dis(-VEC_RANGE, VEC_RANGE);

  for (int i = 0; i < numARows; i++) {
    for (int j = 0; j < numAColumns; j++) {
      hostA[numAColumns * i + j] = dis(gen);
    }
  }

  for (int i = 0; i < numBRows; i++) {
    for (int j = 0; j < numBColumns; j++){
      hostB[numBColumns * i + j] = dis(gen);
    }
  }

  // calc reference result
  Timer cpuTimer;
  resultRef = (DataType*) malloc(sizeof(DataType) * numCRows * numCColumns);
  for (int i = 0; i < numCRows; i++) {
    for (int j = 0; j < numCColumns; j++){
      resultRef[numCColumns * i + j] = 0.0;
      // calculate dot product of row A[i] and column B[j]
      for(int k = 0; k < numAColumns; k++){
        resultRef[numCColumns * i + j] += hostA[numAColumns * i + k] * hostB[numBColumns * k + j];
      }
    }
  }
  printf("CPU time: %fs\n", cpuTimer.get());

  Timer gpuTimer;
  //@@ Insert code below to allocate GPU memory here
  Timer gpuMallocTimer;
  cudaMalloc(&deviceA, sizeof(DataType) * numARows * numAColumns);
  cudaMalloc(&deviceB, sizeof(DataType) * numBRows * numBColumns);
  cudaMalloc(&deviceC, sizeof(DataType) * numCRows * numCColumns);
  printf("GPU malloc time: %fs\n", gpuMallocTimer.get());

  //@@ Insert code to below to Copy memory to the GPU here
  Timer h2dTimer;
  cudaMemcpy(deviceA, hostA, sizeof(DataType) * numARows * numAColumns, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, sizeof(DataType) * numBRows * numBColumns, cudaMemcpyHostToDevice);
  printf("Host to device copy time: %fs\n", h2dTimer.get());

  //@@ Initialize the grid and block dimensions here
  dim3 blockSize(TPBD, TPBD);
  dim3 gridSize((numCRows + TPBD - 1) / TPBD, (numCColumns + TPBD - 1) / TPBD);

  //@@ Launch the GPU Kernel here
  Timer kernelTimer;
  gemm<<<gridSize, blockSize>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns);
  cudaDeviceSynchronize();
  printf("Kernel time: %fs\n", kernelTimer.get());

  //@@ Copy the GPU memory back to the CPU here
  Timer d2hTimer;
  cudaMemcpy(hostC, deviceC, sizeof(DataType) * numCRows * numCColumns, cudaMemcpyDeviceToHost);
  printf("Device to host copy time: %fs\n", d2hTimer.get());
  
  printf("GPU time: %fs\n", gpuTimer.get());

  //@@ Insert code below to compare the output with the reference
  printf("VERIFYING\n");
  bool isCorrect = true;
  DataType maxDiff = 0.0;
  for (int i = 0; i < numCRows; i++) {
    for (int j = 0; j < numCColumns; j++){
      DataType diff = std::abs(resultRef[numCColumns * i + j] - hostC[numCColumns * i + j]);
      if(diff > std::numeric_limits<DataType>::min()){
        if (diff > maxDiff) maxDiff = diff;
        isCorrect = false;
      }
    }
  }
  
  if (isCorrect){
    printf("Correct\n");
  } else {
    printf("CPU and GPU result differ. This may be due to numerical inacurracy.\n");
    printf("Max difference (element wise): %.5e\n", maxDiff);
  }

  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  //@@ Free the CPU memory here
  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
