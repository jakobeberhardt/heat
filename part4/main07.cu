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


