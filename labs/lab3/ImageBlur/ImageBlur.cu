//#include <wb.h>
#include "/home/prof/wagner/ci853/labs/wb3.h" // use our lib instead (under construction)
//#include "/home/wagner/ci853/labs-achel/wb.h" // use our lib instead (under construction)


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

//@@ INSERT CODE HERE


__global__ void imageBlur (unsigned char* in, unsigned char* out, 
  int w, int h) {

  int Col = threadIdx.x + blockIdx.x * blockDim.x;
  int Row = threadIdx.y + blockIdx.y * blockDim.y;

  if ( Col < w && Row < h ) {
    
    int pixVal = 0;
    int pixels = 0;
    // Get the average of the surrounding 2xBLUR_SIZE x 2xBLUR_SIZE box
    for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE+1; ++blurRow) {
      for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE+1; ++blurCol) {
        int curRow = Row + blurRow;
        int curCol = Col + blurCol;
        // Verify we have a valid image pixel
        if (curRow > -1 && curRow < h && curCol > -1 && curCol < w) { 
          pixVal += in[curRow * w + curCol];
          pixels++; 
          // Keep track of number of pixels in the accumulated total
        }
      }
    }
    // Write our new pixel value out
    out[Row * w + Col] = (unsigned char)(pixVal / pixels);
  }
}

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

  // Since the image is monochromatic, it only contains only one channel
  outputImage = wbImage_new(imageWidth, imageHeight, 1);

  hostInputImageData  = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  cudaMalloc((void **)&deviceInputImageData,
             imageWidth * imageHeight * sizeof(unsigned char));
  cudaMalloc((void **)&deviceOutputImageData,
             imageWidth * imageHeight * sizeof(unsigned char));
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  cudaMemcpy(deviceInputImageData, hostInputImageData,
             imageWidth * imageHeight * sizeof(unsigned char),
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
