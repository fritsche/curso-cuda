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

  // the thread index
  int index = threadIdx.x + blockIdx.x * blockDim.x;

  char r, g, b;

  // iterate over the tiles
  for (int i = 0; i < imageHeight*imageWidth/NT1; ++i)
  {
    // a thread can be outside the image?
    if (! index+i*NT1 > imageWidth*imageHeight) {
      // load the chunk
      sharedInputImage[index] = inputImage[index+i*NT1];
    }
    __syncthreads();


    r = ucharInputImage[index+0];
    g = ucharInputImage[index+1];
    b = ucharInputImage[index+2];
    unsigned int v = ((unsigned int)r << 16) + ((unsigned int)g << 8) + (unsigned int)b;

    outputImage[index+i*NT1] = v;

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

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  cudaMalloc((void **)&deviceInputImageData,
             imageWidth * imageHeight * CHANNELS * sizeof(unsigned char) + 3);
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
             imageWidth * imageHeight * CHANNELS * sizeof(unsigned char),
             cudaMemcpyDeviceToHost);

  for (int i = 0; i < imageWidth * imageHeight * CHANNELS; ++i)
  {
    printf("%d\n", hostConvertedImage[i]);
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
