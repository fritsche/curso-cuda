//#include <wb.h>
#include "/home/prof/wagner/ci853/labs/wb3.h" // use our lib instead (under construction)
//#include "/home/wagner/ci853/labs-achel/wb.h" // use our lib instead (under construction)

#include <string.h>

#define BLOCK_DIM 256

__global__ void vecAdd(float *in1, float *in2, float *out, int len) {
  //@@ Insert code to implement vector addition here
  int i = threadIdx.x+blockDim.x*blockIdx.x;
  if(i<len) out[i] = in1[i] + in2[i];
  //@done
}

__host__ void printResult (float *out, int len) {
  int i;
  printf("%d\n", len);
  for (i = 0; i < len; ++i) {
    printf("%.2f\n", out[i]);
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  int inputLength;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;
  float *deviceInput1;
  float *deviceInput2;
  float *deviceOutput;

  args = wbArg_read(argc, argv);
  // show_args( args ); // debug

  wbTime_start(Generic, "Importing data and creating memory on host");

  hostInput1 =
      (float *)wbImport( wbArg_getInputFile(args, 0), &inputLength );
  hostInput2 =
      (float *)wbImport( wbArg_getInputFile(args, 1), &inputLength );
  hostOutput = (float *)malloc(inputLength * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The input length is ", inputLength);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  // compute the size of the vector in the memory
  int size = inputLength * sizeof(float);
  // malloc ((cast to void because the methods expected a pointer to void input parameter))
  cudaError_t err = cudaMalloc((void **) &deviceInput1, size);
  if (err  != cudaSuccess)  {
    printf("%s in %s at line %d\n",cudaGetErrorString(err), __FILE__,__LINE__);
    exit(EXIT_FAILURE);
  }
  err = cudaMalloc((void **) &deviceInput2, size);
  if (err  != cudaSuccess)  {
    printf("%s in %s at line %d\n",cudaGetErrorString(err), __FILE__,__LINE__);
    exit(EXIT_FAILURE);
  }
  err = cudaMalloc((void **) &deviceOutput, size);
  if (err  != cudaSuccess)  {
    printf("%s in %s at line %d\n",cudaGetErrorString(err), __FILE__,__LINE__);
    exit(EXIT_FAILURE);
  }
  //@done
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  // memcpy (destination, origin, size in memory, flag for orientation)
  cudaMemcpy(deviceInput1, hostInput1, size, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceInput2, hostInput2, size, cudaMemcpyHostToDevice);
  // cudaMemcpy(deviceOutput, hostOutput, size, cudaMemcpyHostToDevice);
  // @done
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  int blockDim = BLOCK_DIM;
  int gridDim = (((inputLength-1)/blockDim)+1); // ceil(n/256.0);
  //@done
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  vecAdd<<<gridDim, blockDim>>> (deviceInput1, deviceInput2, deviceOutput, inputLength);
  //@done
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  // memcpy (destination, origin, size in memory, flag for orientation)
  // cudaMemcpy(hostInput1, deviceInput1, size, cudaMemcpyDeviceToHost);
  // cudaMemcpy(hostInput2, deviceInput2, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(hostOutput, deviceOutput, size, cudaMemcpyDeviceToHost);
  // @done
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);
  //@done
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, inputLength);

  // printResult(hostOutput, inputLength);

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}
