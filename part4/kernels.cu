#include <math.h>
#include <float.h>
#include <cuda.h>

__global__ void gpu_Heat (float *h, float *g, int N) {
	int i = blockIdx.y* blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >0 && i < N -1 && j> 0 && j< N-1){
		h[i*N+j]= 0.25 * (g[ i*N + (j-1) ]+  // left
					     g[ i*N + (j+1) ]+  // right
				         g[ (i-1)*N+ j ]+  // top
				         g[ (i+1)*N + j]); // bottom
	}
}
