//#include <wb.h>
#include "/home/prof/wagner/ci853/labs/wb4.h" // use our lib instead (under construction)


#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define BLUR_SIZE 5
#define CHANNELS 3

#define GTX480   480
#define GTX680   680
#define GPU     GTX680
#if GPU == GTX480
  #define MP    15  // number of mutiprocessors (SMs) in GTX480
  #define GRID1 (MP*2) // GRID size
  #define NT1   768
#elif GPU == GTX680
  #define MP    8 // number of mutiprocessors (SMs) in GTX680
  #define GRID1 (MP*2)
  #define NT1    1024
#endif

int main(int argc, char *argv[]) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  char *inputImageFile;
  wbImage_t inputImage;
  wbImage_t outputImage;
  unsigned char *hostInputImageData;
  unsigned char *hostOutputImageData;
  unsigned char *deviceInputImageData;
  unsigned char *deviceOutputImageData;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 1);

  inputImage = wbImport(inputImageFile);

  // The input image is in grayscale, so the number of channels
  // is 1
  imageWidth  = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);

  outputImage = wbImage_new(imageWidth, imageHeight, CHANNELS);

  hostInputImageData  = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  cudaMalloc((void **)&deviceInputImageData,
             imageWidth * imageHeight * CHANNELS * sizeof(unsigned char));
  cudaMalloc((void **)&deviceOutputImageData,
             imageWidth * imageHeight * CHANNELS * sizeof(unsigned char));
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  cudaMemcpy(deviceInputImageData, hostInputImageData,
             imageWidth * imageHeight * CHANNELS * sizeof(unsigned char),
             cudaMemcpyHostToDevice);
  wbTime_stop(Copy, "Copying data to the GPU");
  ///////////////////////////////////////////////////////
  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ INSERT CODE HERE
  dim3 DimGrid((imageWidth-1)/16 + 1, (imageHeight-1)/16+1, 1);
  dim3 DimBlock(16, 16, 1);
  imageBlur<<<DimGrid,DimBlock>>>(deviceInputImageData, deviceOutputImageData, imageWidth, imageHeight );

  wbTime_stop(Compute, "Doing the computation on the GPU");

  ///////////////////////////////////////////////////////
  wbTime_start(Copy, "Copying data from the GPU");
  cudaMemcpy(hostOutputImageData, deviceOutputImageData,
             imageWidth * imageHeight * CHANNELS * sizeof(unsigned char),
             cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  wbSolution(args, outputImage);
  wbExport("blured.ppm", outputImage);

  cudaFree(deviceInputImageData);
  cudaFree(deviceOutputImageData);

  wbImage_delete(outputImage);
  wbImage_delete(inputImage);

  return 0;
}
