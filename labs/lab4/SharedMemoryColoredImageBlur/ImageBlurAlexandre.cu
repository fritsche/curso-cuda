//#include <wb.h>
#include "wb4.h" // use our lib instead (under construction)
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
#define CHANNELS 3

#define GTX480 480
#define GTX680 680
#define GPU GTX680
#if GPU == GTX680
	#define MP 15
	#define GRID1 (MP*2)
 	#define NT1 768
#elif GPU == GTX680
	#define MP 8
	#define GRID1(MP*2)
  #define NT1 1024
#endif



 __global__ void rgb2uintKernelSHM(unsigned int * inputImage, unsigned int * outputImage,
   int height, int width){

    __shared__ unsigned int sm[NT1];
    __shared__ unsigned char *sm2;

    unsigned int i;
    int tx = threadIdx.x;
    int size = height * width;
    unsigned char r, g, b, r2;
    for(i = threadIdx.x + blockIdx.x * blockDim.x; i < size; i+= gridDim.x * blockDim.x){
        /* code */

        sm[tx] = inputImage[i];
        
        __syncthreads();
        sm2 = (unsigned char *)sm;

        r = sm2[tx*CHANNELS+0];
        g = sm2[tx*CHANNELS+1];
        b = sm2[tx*CHANNELS+2];


        outputImage[i] = ((unsigned int)r << 16) + ((unsigned int)g << 8) + (unsigned int) b;
        __syncthreads();
      }
    }

 __global__ void uint2rgbKernelSHM(unsigned int * inputImage, unsigned int * outputImage,
   int height, int width){

    // __shared__ unsigned int sm[NT1];
    // __shared__ unsigned char *sm2;
  unsigned char *sm2;

    unsigned int i;
    int tx = threadIdx.x;
    int size = height * width;
    unsigned char r, g, b, r2;
    unsigned char * outputImageChar = (unsigned char *) outputImage;

    for(i = threadIdx.x + blockIdx.x * blockDim.x; i < size; i+= gridDim.x * blockDim.x){
        /* code */

        // sm[tx] = inputImage[i];
        
        //__syncthreads();
        
        sm2 = (unsigned char *) inputImage;

        r = sm2[tx*4+0];
        g = sm2[tx*4+1];
        b = sm2[tx*4+2];

        outputImageChar[i+0] = b;
        outputImageChar[i+1] = g;
        outputImageChar[i+2] = r;

        printf("alexandre: [%d %d %d]\n", outputImageChar[i+0], outputImageChar[i+1], outputImageChar[i+2] );

        //__syncthreads();
      }
    }




int main(int argc, char *argv[]) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  char *inputImageFile;
  wbImage_t inputImage;
  wbImage_t outputImage;
  //blur
  //unsigned char *hostInputImageData;
  //unsigned char *hostOutputImageData;
  //unsigned char *deviceInputImageData;
  //unsigned char *deviceOutputImageData;
  // rgb to urgb
  unsigned int *deviceUrgbInput;
  unsigned char *hostUrgbInput;
  unsigned int *deviceUrgbOutput;
  unsigned int *hostUrgbOutput;
  //TESTE
  unsigned char *hostInputImageData;
  unsigned int * hostConvertedImage;

  args = wbArg_read(argc, argv); /* parse the input arguments */
  inputImageFile = wbArg_getInputFile(args, 1);
  inputImage = wbImport(inputImageFile);
  imageWidth  = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);

  outputImage = wbImage_new(imageWidth, imageHeight, CHANNELS);

  hostUrgbInput  = wbImage_getData(inputImage);
  hostUrgbOutput = (unsigned int *) malloc(imageWidth * imageHeight * sizeof(unsigned int));
  //TESTE
  hostInputImageData = (unsigned char*) malloc (sizeof(unsigned char) * 15);
  hostInputImageData[0] = 0;
  hostInputImageData[1] = 1;
  hostInputImageData[2] = 2;
  hostInputImageData[3] = 3;
  hostInputImageData[4] = 4;
  hostInputImageData[5] = 5;
  hostInputImageData[6] = 6;
  hostInputImageData[7] = 7;
  hostInputImageData[8] = 8;
  hostInputImageData[9] = 9;
  hostInputImageData[10] = 10;
  hostInputImageData[11] = 11;
  hostInputImageData[12] = 0;
  hostInputImageData[13] = 1;
  hostInputImageData[14] = 2;

  imageWidth  = 5;
  imageHeight = 1;
  //TEste
  //allocating memory first kernel
  cudaMalloc((void **)&deviceUrgbInput,
             imageWidth * imageHeight * sizeof(unsigned int));
  cudaMalloc((void **)&deviceUrgbOutput,
             imageWidth * imageHeight * sizeof(unsigned int));
  //copying data to GPU
  cudaMemcpy(deviceUrgbInput, hostInputImageData,
             imageWidth * imageHeight * sizeof(unsigned int),
             cudaMemcpyHostToDevice);

  //lauching kernell
  hostConvertedImage = (unsigned int *) malloc (imageWidth * imageHeight * sizeof(unsigned char));
  // for (int i = 0; i < imageWidth * imageHeight * CHANNELS; ++i)
  // {
  //    printf("%u \n", hostConvertedImage[i]);
  //  }

  rgb2uintKernelSHM<<<GRID1,NT1>>>(deviceUrgbInput, deviceUrgbOutput, imageHeight, imageWidth);
  //copy data from gpu
  cudaMemcpy(hostConvertedImage, deviceUrgbOutput,
   imageWidth * imageHeight * sizeof(unsigned int),cudaMemcpyDeviceToHost);

   printf("chegou aqui\n");
   for (int i = 0; i < imageWidth * imageHeight; ++i)
   {
      printf("%u\n", hostConvertedImage[i]);
    }


  uint2rgbKernelSHM<<<GRID1,NT1>>>(deviceUrgbOutput, deviceUrgbInput, imageHeight, imageWidth);


  unsigned char * saida;

  cudaMemcpy(saida, deviceUrgbInput,
    imageWidth * imageHeight * CHANNELS * sizeof(unsigned char), cudaMemcpyDeviceToHost);

   printf("chegou aqui\n");
   for (int i = 0; i < imageWidth * imageHeight * CHANNELS; ++i)
   {
      printf("saida[%d]=%u\n", i, saida[i]);
    }

  //wbExport("blurred.ppm", hostUrgbOutput);

/*  wbSolution(args, outputImage);

  cudaFree(deviceInputImageData);
  cudaFree(deviceOutputImageData);

  wbImage_delete(outputImage);
  wbImage_delete(inputImage);*/

  return 0;
}
