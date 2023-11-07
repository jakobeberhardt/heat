#include <math.h>
#include <float.h>
#include <cuda.h>

__global__ void gpu_Heat (double *h, double *g, int N) {
	int i = blockIdx.y* blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >0 && i < N -1 && j> 0 && j< N-1){
		h[i*N+j]= 0.25 * (g[i * N + (j-1) ]+  // left
					      g[ i * N + (j+1) ]+  // right
				          g[ (i-1) * N + j ]+  // top
				          g[ (i+1) * N + j ]); // bottom
	}
}

__global__ void gpu_residual(double *u, double *utmp, double *residual, int N) {
  // Calculate the thread's unique index
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int index = i * N + j;

    // Shared memory for in-block reduction
    extern __shared__ double sdata[];

    // Each thread computes one element (if within the domain boundaries)
    double diff = 0.0;
    if (i > 0 && i < N - 1 && j > 0 && j < N - 1) {
        diff = utmp[index] - u[index];
    }
    sdata[threadIdx.y * blockDim.x + threadIdx.x] = diff * diff;

    __syncthreads(); // Wait for all threads in the block to finish updating shared memory

    // Perform in-block reduction
    // For simplicity, we assume blockDim.x * blockDim.y is a power of 2
    int blockSize = blockDim.x * blockDim.y;
    if (blockSize >= 512 && threadIdx.x < 256) { sdata[threadIdx.y * blockDim.x + threadIdx.x] += sdata[threadIdx.y * blockDim.x + threadIdx.x + 256]; } __syncthreads();
    if (blockSize >= 256 && threadIdx.x < 128) { sdata[threadIdx.y * blockDim.x + threadIdx.x] += sdata[threadIdx.y * blockDim.x + threadIdx.x + 128]; } __syncthreads();
    if (blockSize >= 128 && threadIdx.x <  64) { sdata[threadIdx.y * blockDim.x + threadIdx.x] += sdata[threadIdx.y * blockDim.x + threadIdx.x +  64]; } __syncthreads();

    // Now that we are using 64 threads or less, we can assume that we are within a warp and no longer need to synchronize
    if (threadIdx.x < 32) {
        volatile double* smem = sdata;
        if (blockSize >=  64) { smem[threadIdx.x] += smem[threadIdx.x + 32]; }
        if (blockSize >=  32) { smem[threadIdx.x] += smem[threadIdx.x + 16]; }
        if (blockSize >=  16) { smem[threadIdx.x] += smem[threadIdx.x +  8]; }
        if (blockSize >=   8) { smem[threadIdx.x] += smem[threadIdx.x +  4]; }
        if (blockSize >=   4) { smem[threadIdx.x] += smem[threadIdx.x +  2]; }
        if (blockSize >=   2) { smem[threadIdx.x] += smem[threadIdx.x +  1]; }
    }

    // Write the result for this block to global memory
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        residual[blockIdx.x + gridDim.x * blockIdx.y] = sdata[0];
    }
}

__global__ void gpu_Residual(double *u, double *utmp,double *dev_diff, double *residuals, int N){
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int index = i * N + j;
  unsigned int diff_index = (i-1)*(N-2)+j-1;
    if (i > 0 && i < N - 1 && j > 0 && j < N - 1) {
        dev_diff[diff_index] = utmp[index] - u[index];
        residuals[diff_index]=dev_diff[diff_index]*dev_diff[diff_index];
}
}

__global__ void Kernel07(double *g_idata, double *g_odata, int N) {
  __shared__ double sdata[1024];
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

__global__ void finalReduceKernel(double *g_idata, double *g_odata, int N) {
    extern __shared__ double sdata[];

    unsigned int tid = threadIdx.x;

    // Load block sums from global memory to shared memory
    sdata[tid] = (tid < N) ? g_idata[tid] : 0;
    __syncthreads();

    // Perform final reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write the final result to global memory
    if (tid == 0) g_odata[0] = sdata[0];
}