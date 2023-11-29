
#include <iostream>
#include <chrono>
#include <random>
#include <string>

#define NUM_BINS 4096
#define TPB_HIST 1024
#define TPB_CONV 32

__global__ void histogram_kernel(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements,
                                 unsigned int num_bins) {

//@@ Insert code below to compute histogram of input using shared memory and atomics
  // input element
  const int el = blockIdx.x * blockDim.x + threadIdx.x;
  if (el >= num_elements) return;
  atomicAdd(&bins[input[el]], 1u);
}

__global__ void convert_kernel(unsigned int *bins, unsigned int num_bins) {

//@@ Insert code below to clean up bins that saturate at 127

  const unsigned int bin = blockIdx.x * blockDim.x + threadIdx.x;
  if(bin >= num_bins) return;
  unsigned int count = bins[bin];
  bins[bin] = count > 127 ? 127 : count;
}

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
  unsigned int *hostInput;
  unsigned int *hostBins;
  unsigned int *resultRef;
  unsigned int *deviceInput;
  unsigned int *deviceBins;

  //@@ Insert code below to read in inputLength from args
  if (argc > 1)
    inputLength = std::stoi(argv[1]);
  printf("The input length is %d\n", inputLength);
  
  //@@ Insert code below to allocate Host memory for input and output
  hostInput = (unsigned int*) malloc(sizeof(unsigned int) * inputLength);
  hostBins = (unsigned int*) malloc(sizeof(unsigned int) * NUM_BINS);
  
  //@@ Insert code below to initialize hostInput to random numbers whose values range from 0 to (NUM_BINS - 1)
  std::random_device rd;
  std::minstd_rand gen(rd());
  std::uniform_int_distribution<unsigned int> dis(0, NUM_BINS-1);
  for(int i = 0; i < inputLength; i++){
    hostInput[i] = dis(gen);
  }

  //@@ Insert code below to create reference result in CPU
  Timer cpuTimer;
  resultRef = (unsigned int*) calloc(NUM_BINS, sizeof(unsigned int));
  for(int i = 0; i < inputLength; i++){
    unsigned int input = hostInput[i];
    if (resultRef[input] < 127)
      resultRef[input]++;
  }
  printf("CPU time: %fs\n", cpuTimer.get());

  Timer gpuTimer;
  //@@ Insert code below to allocate GPU memory here
  cudaMalloc(&deviceInput, sizeof(unsigned int) * inputLength);
  
  //@@ Insert code to Copy memory to the GPU here
  cudaMemcpy(deviceInput, hostInput, sizeof(unsigned int) * inputLength, cudaMemcpyHostToDevice);
  
  //@@ Insert code to initialize GPU results
  cudaMalloc(&deviceBins, sizeof(unsigned int) * NUM_BINS);
  cudaMemset(deviceBins, 0u, sizeof(unsigned int) * NUM_BINS);

  //@@ Initialize the grid and block dimensions here
  const dim3 blockSizeHist(TPB_HIST);
  const dim3 gridSizeHist((inputLength + TPB_HIST - 1) / TPB_HIST);

  //@@ Launch the GPU Kernel here
  histogram_kernel<<<gridSizeHist,blockSizeHist>>>(deviceInput, deviceBins, inputLength, NUM_BINS);
  
  //@@ Initialize the second grid and block dimensions here
  const dim3 blockSizeConv(TPB_CONV);
  const dim3 gridSizeConv((NUM_BINS + TPB_CONV - 1) / TPB_CONV);
  
  //@@ Launch the second GPU Kernel here
  convert_kernel<<<gridSizeConv,blockSizeConv>>>(deviceBins, NUM_BINS);

  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostBins, deviceBins, sizeof(unsigned int) * NUM_BINS, cudaMemcpyDeviceToHost);
  printf("GPU time: %fs\n", gpuTimer.get());

  //@@ Insert code below to compare the output with the reference
  printf("VERIFYING\n");
  bool isCorrect = true;
  for (int i = 0; i < NUM_BINS; i++){
    if(resultRef[i] != hostBins[i]){
      printf("%d: %d \t %d\n", i, resultRef[i], hostBins[i]);
      isCorrect = false;
      break;
    }
  }

  if (isCorrect)
    printf("Correct\n");
  else
    printf("CPU and GPU results do not match\n");


  //@@ Free the GPU memory here
  cudaFree(deviceInput);
  cudaFree(deviceBins);

  //@@ Free the CPU memory here
  free(hostInput);
  free(hostBins);
  free(resultRef);

  return 0;
}

