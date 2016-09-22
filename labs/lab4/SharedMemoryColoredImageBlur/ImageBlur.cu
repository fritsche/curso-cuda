#include "wb4.h"
//#include "/home/prof/wagner/ci853/labs/wb4.h" // use our lib instead (under construction)

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
#define MYGPU     GTX680
#if MYGPU == GTX480
  #define MP    15  // number of mutiprocessors (SMs) in GTX480
  #define GRID1 (MP*2) // GRID size
  #define NT1   768
#elif MYGPU == GTX680
  #define MP    8 // number of mutiprocessors (SMs) in GTX680
  #define GRID1 (MP*2)
  #define NT1    1024
#endif


__global__ void rgb2uintKernelSHM (unsigned int* inputImage, unsigned int* outputImage, int imageHeight, int imageWidth) {
  __shared__ unsigned int sharedInputImage [NT1];
  unsigned char * ucharInputImage = ( unsigned char *) sharedInputImage;

  // unique thread index
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int indexAtImage;
  char r, g, b;

  // iterate over the tiles
  for (int i = 0; i < imageHeight*imageWidth/(float)NT1; ++i)
  {

    // a thread can be outside the image?
    indexAtImage = index+i*NT1;
    // (((inputLength-1)/blockDim)+1)
    if ( indexAtImage < (imageWidth*imageHeight*CHANNELS/4.0) + 1) {
      // load the chunk
      sharedInputImage[threadIdx.x] = inputImage[indexAtImage];
    }
    __syncthreads();

    if ( indexAtImage < imageWidth*imageHeight) {
     
      r = ucharInputImage[threadIdx.x*CHANNELS+0];
      g = ucharInputImage[threadIdx.x*CHANNELS+1];
      b = ucharInputImage[threadIdx.x*CHANNELS+2];

      // printf("[%d %d %d]\n", r, g, b);

      unsigned int v = ((unsigned int)r << 16) + ((unsigned int)g << 8) + (unsigned int)b;

      // printf("%u\n",v);

      outputImage[index+i*NT1] = v;
    }
    __syncthreads();
  }

}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  char *inputImageFile; // name of the file
  wbImage_t inputImage; // the image loaded 
  wbImage_t outputImage;
  unsigned char *hostInputImageData;
  unsigned char *hostOutputImageData; // the final image
  unsigned char *deviceInputImageData;
  unsigned char *deviceOutputImageData;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 1);

  inputImage = wbImport(inputImageFile);

  imageWidth  = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);

  outputImage = wbImage_new(imageWidth, imageHeight, CHANNELS);

  hostInputImageData  = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);

  /// TESTE
  hostInputImageData = (unsigned char*) malloc (sizeof(unsigned char) * 15);
  hostInputImageData[0] = 0;
  hostInputImageData[1] = 1;
  hostInputImageData[2] = 2; // 258
  hostInputImageData[3] = 3;
  hostInputImageData[4] = 4;
  hostInputImageData[5] = 5; // 197637
  hostInputImageData[6] = 6;
  hostInputImageData[7] = 7;
  hostInputImageData[8] = 8; // 395016
  hostInputImageData[9] = 9;
  hostInputImageData[10] = 10;
  hostInputImageData[11] = 11; // 592395
  hostInputImageData[12] = 0;
  hostInputImageData[13] = 1;
  hostInputImageData[14] = 2; // 258

  imageWidth  = 5;
  imageHeight = 1;
  /// TESTE
  
  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  cudaMalloc((void **)&deviceInputImageData,
             imageWidth * imageHeight * CHANNELS * sizeof(unsigned char) + 7);
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
  
  unsigned int * deviceAuxInputImage = (unsigned int *) deviceInputImageData;
  unsigned int * deviceConvertedImage;
  unsigned int * hostConvertedImage;
  //vector = (float *)malloc( n * sizeof(float) ); 
  hostConvertedImage = (unsigned int *) malloc (imageWidth * imageHeight * CHANNELS * sizeof(unsigned int));

  cudaMalloc((void **)&deviceConvertedImage,
             imageWidth * imageHeight * CHANNELS * sizeof(unsigned int));

  //(unsigned int* inputImage, unsigned int* outputImage, int imageHeight, int imageWidth) {
  rgb2uintKernelSHM<<<GRID1, NT1>>> (deviceAuxInputImage, deviceConvertedImage, imageHeight, imageWidth);

  cudaMemcpy(hostConvertedImage, deviceConvertedImage,
             imageWidth * imageHeight * CHANNELS * sizeof(unsigned int),
             cudaMemcpyDeviceToHost);

  for (int i = 0; i < imageWidth * imageHeight; ++i)
  {
    printf("%u\n", hostConvertedImage[i]);
  }

  // //@@ INSERT CODE HERE
  // dim3 DimGrid((imageWidth-1)/16 + 1, (imageHeight-1)/16+1, 1);
  // dim3 DimBlock(16, 16, 1);
  // imageBlur<<<DimGrid,DimBlock>>>(deviceInputImageData, deviceOutputImageData, imageWidth, imageHeight );

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
