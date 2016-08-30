//#include <wb.h>
#include "/home/prof/wagner/ci853/labs/wb3.h" // use our lib instead (under construction)
//#include "/home/wagner/ci853/labs-achel/wb.h" // use our lib instead (under construction)

#include <string.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ INSERT CODE HERE

# define CHANNELS 3

__global__ void colorConvert (unsigned char* grayImage, unsigned char* rgbImage, 
  int width, int height) {

  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if ( x < width && y < height ) {
    // get 1D coordinate for the grayscale image
    int grayOffset = y*width + x;
    // one can think of the RGB image having
    // CHANNEL times columns than the gray scale image
    int rgbOffset = grayOffset *CHANNELS;
    unsigned char r =   rgbImage [rgbOffset]; 
    // red value for pixel 
    unsigned char g = rgbImage [rgbOffset + 1]; 
    // green value for pixel
    unsigned char b = rgbImage [rgbOffset + 2]; 
    // blue value for pixel

    grayImage[grayOffset] = 0.21f*r + 0.71f*g + 0.07f*b;
  }
}


// DONE

int main(int argc, char *argv[]) {
  wbArg_t args;
  int imageChannels;
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
  // show_args( args ); // debug

  inputImageFile = wbArg_getInputFile(args, 1);

  inputImage = wbImport(inputImageFile);

  imageWidth  = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  // For this lab the value is always 3
  imageChannels = wbImage_getChannels(inputImage);

  // Since the image is monochromatic, it only contains one channel
  outputImage = wbImage_new(imageWidth, imageHeight, 1);

  hostInputImageData  = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  cudaMalloc((void **)&deviceInputImageData,
             imageWidth * imageHeight * imageChannels * sizeof(unsigned char));
  cudaMalloc((void **)&deviceOutputImageData,
             imageWidth * imageHeight * sizeof(unsigned char));
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  cudaMemcpy(deviceInputImageData, hostInputImageData,
             imageWidth * imageHeight * imageChannels * sizeof(unsigned char),
             cudaMemcpyHostToDevice);
  wbTime_stop(Copy, "Copying data to the GPU");

  ///////////////////////////////////////////////////////
  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ INSERT CODE HERE
  dim3 DimGrid((imageWidth-1)/16 + 1, (imageHeight-1)/16+1, 1);
  dim3 DimBlock(16, 16, 1);
  colorConvert<<<DimGrid,DimBlock>>>(deviceOutputImageData, deviceInputImageData, imageWidth, imageHeight );

  wbTime_stop(Compute, "Doing the computation on the GPU");

  ///////////////////////////////////////////////////////
  wbTime_start(Copy, "Copying data from the GPU");
  cudaMemcpy(hostOutputImageData, deviceOutputImageData,
             imageWidth * imageHeight * sizeof(unsigned char),
             cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  wbSolution(args, outputImage);

  cudaFree(deviceInputImageData);
  cudaFree(deviceOutputImageData);

  wbImage_delete(outputImage);
  wbImage_delete(inputImage);

  return 0;
}
