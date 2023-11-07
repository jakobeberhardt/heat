#include <stdio.h>
#include <stdlib.h>

#ifndef NumElem
#define NumElem 512
#endif


#include <sys/times.h>
#include <sys/resource.h>

float GetTime(void)        
{
  struct timeval tim;
  struct rusage ru;
  getrusage(RUSAGE_SELF, &ru);
  tim=ru.ru_utime;
  return ((float)tim.tv_sec + (float)tim.tv_usec / 1000000.0)*1000.0;
}

__global__ void Kernel07(double *g_idata, double *g_odata, int N) {
  __shared__ double sdata[NumElem];
  unsigned int s;

  // Cada thread realiza la suma parcial de los datos que le
  // corresponden y la deja en la memoria compartida
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
  unsigned int gridSize = blockDim.x*2*gridDim.x;
  sdata[tid] = 0;
  while (i < N) {
    sdata[tid] += g_idata[i] + g_idata[i+blockDim.x];
    i += gridSize;
  }
  __syncthreads();

  // Hacemos la reduccion en la memoria compartida
  for (s=blockDim.x/2; s>32; s>>=1) {
    if (tid < s)
      sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  // desenrrollamos el ultimo warp activo
  if (tid < 32) {
    volatile double *smem = sdata;

    smem[tid] += smem[tid + 32];
    smem[tid] += smem[tid + 16];
    smem[tid] += smem[tid + 8];
    smem[tid] += smem[tid + 4];
    smem[tid] += smem[tid + 2];
    smem[tid] += smem[tid + 1];
  }


  // El thread 0 escribe el resultado de este bloque en la memoria global
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];

}


void InitV(int N, double *v);
int Test(int N, double *v, double sum, double *res);

int main(int argc, char** argv) {
  unsigned int N;
  unsigned int numBytesV, numBytesW;
  unsigned int nBlocks, nThreads;
  int test;
  float SeqTime, elapsedTime;
  float t1,t2; 

  cudaEvent_t start, stop;

  double *h_v, *h_w;
  double *d_v, *d_w;

  double SUM, SumSeq;
  int i;
  int count, gpu, tmp;

  N = 1024 * 1024 * 16;
  nThreads = NumElem;  // Este valor ha de coincidir con el numero de elementos que trata el kernel

  // Numero maximo de Block Threads = 65535
  nBlocks = 4096;
  
  numBytesV = N * sizeof(double);
  numBytesW = nBlocks * sizeof(double);

  // Buscar GPU de forma aleatoria
  cudaGetDeviceCount(&count);
  srand(time(NULL));
  tmp = rand();
  gpu = (tmp>>3) % count;
  cudaSetDevice(gpu);


  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Obtener Memoria en el host
  h_v = (double*) malloc(numBytesV);
  h_w = (double*) malloc(numBytesW);

  // Obtiene Memoria [pinned] en el host
  //cudaMallocHost((double**)&h_v, numBytesV);
  //cudaMallocHost((double**)&h_w, numBytesW);

  // Inicializa los vectores
  InitV(N, h_v);


  // Obtener Memoria en el device
  cudaMalloc((double**)&d_v, numBytesV);
  cudaMalloc((double**)&d_w, numBytesW);

  // Copiar datos desde el host en el device
  cudaMemcpy(d_v, h_v, numBytesV, cudaMemcpyHostToDevice);

  cudaEventRecord(start, 0);

  // Ejecutar el kernel
  Kernel07<<<nBlocks, nThreads>>>(d_v, d_w, N);

  // Obtener el resultado parcial desde el host
  cudaMemcpy(h_w, d_w, numBytesW, cudaMemcpyDeviceToHost);


  SUM = 0.0;
  for (i=0; i<nBlocks; i++)
    SUM = SUM + h_w[i];


  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  // Liberar Memoria del device 
  cudaFree(d_v);
  cudaFree(d_w);
 
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("\nKERNEL 07\n");
  printf("GPU used: %d\n", gpu);
  printf("Vector Size: %d\n", N);
  printf("nThreads: %d\n", nThreads);
  printf("nBlocks: %d\n", nBlocks);
  printf("Total Time %4.6f ms\n", elapsedTime);
  printf("Bandwidth %4.3f GB/s\n", (N * sizeof(double)) / (1000000 * elapsedTime));

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  t1=GetTime();
  test = Test(N, h_v, SUM, &SumSeq);
  t2=GetTime();
  SeqTime = t2 - t1;
  printf("Speedup: x%2.3f \n", SeqTime/elapsedTime);

  if (test)
    printf ("TEST PASS, Time seq: %f ms\n", SeqTime);
  else {
    printf ("ERROR: %f(GPU) : %f(CPU) : %f(diff) : %f(error) \n", SumSeq, SUM, abs(SumSeq-SUM), abs(SumSeq - SUM)/SumSeq);
    printf ("TEST FAIL\n");
  }
}


