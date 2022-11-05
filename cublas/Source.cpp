#include "Header.h"

void verifySolution(float* A, float* B, float* C, int N) {
	float sum;
	float averageError = 0.0f;
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			sum = 0;
			for (int k = 0; k < N; k++) {
				sum += A[k * N + i] * B[j * N + k];
			}
			averageError += fabs(sum - C[j * N + i]);
		}
	}
	averageError /= (N * N);
	cout << "Average Error: " << averageError << endl;
}

int main() {
	int N = 1 << 10;
	int bytes = N * N * sizeof(float);
	float* A, * B, * C;
	float* d_A, * d_B, * d_C;
	A = (float*)malloc(bytes);
	B = (float*)malloc(bytes);
	C = (float*)malloc(bytes);
	cudaMalloc(&d_A, bytes);
	cudaMalloc(&d_B, bytes);
	cudaMalloc(&d_C, bytes);
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventRecord(start, 0);
	
	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, duration_cast<nanoseconds>(high_resolution_clock::now().time_since_epoch()).count());
	curandGenerateUniform(gen, d_A, N * N);
	curandGenerateUniform(gen, d_B, N * N);

	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout << "Time to generate random numbers: " << elapsedTime << " ms" << endl;
	
	cudaEventCreate(&start);
	cudaEventRecord(start, 0);
	
	cublasHandle_t handle;
	cublasCreate(&handle);
	float alpha = 1.0f;
	float beta = 0.0f;
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N);
	cudaMemcpy(A, d_A, bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(B, d_B, bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(C, d_C, bytes, cudaMemcpyDeviceToHost);

	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout << "Time to perform matrix multiplication: " << elapsedTime << " ms" << endl;
	
	cudaEventCreate(&start);
	cudaEventRecord(start, 0);
	
	verifySolution(A, B, C, N);
	
	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout << "Time to verify solution: " << elapsedTime << " ms" << endl;
	
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	free(A);
	free(B);
	free(C);
	
	return 0;
}