#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <iostream>
#include <chrono>

using std::cout;
using std::endl;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::nanoseconds;

int main() {
	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, duration_cast<nanoseconds>(high_resolution_clock::now().time_since_epoch()).count());
	
	cublasHandle_t handle;
	cublasCreate(&handle);
	
	float elapsedTime;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	float alpha = 1.0f;
	float beta = 0.0f;
	int maxIterations = 100;
	
	int a = 500;
	int b = 700;
	int c = 300;
	
	float* A, * B, * C;
	cudaMallocManaged(&A, a * b * sizeof(float));
	cudaMallocManaged(&B, b * c * sizeof(float));
	cudaMallocManaged(&C, a * c * sizeof(float));
	
	float* hC = new float[a * c];
	float* hA = new float[a * b];
	float* hB = new float[b * c];
	
	

	// matrix times matrix
	curandGenerateUniform(gen, A, a * b);
	curandGenerateUniform(gen, B, b * c);

	int iterations = maxIterations;
	cudaEventRecord(start, 0);
	while (iterations--) {
		cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, c, a, b, &alpha, B, c, A, b, &beta, C, c);
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout << "Time: " << elapsedTime / maxIterations << " ms" << endl;
	
	cudaMemcpy(hA, A, a * b * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(hB, B, b * c * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(hC, C, a * c * sizeof(float), cudaMemcpyDeviceToHost);
	
	float* output = new float[a * c];
	for (int i = 0; i < a; i++) {
		for (int j = 0; j < c; j++) {
			float sum = 0;
			for (int k = 0; k < b; k++) {
				sum += hA[i * b + k] * hB[k * c + j];
			}
			output[i * c + j] = sum;
		}
	}
	
	float error = 0;
	for (int i = 0; i < a * c; i++) {
		error += abs(hC[i] - output[i]);
	}
	cout << "Average error: " << error / (a * c) << endl;
	
	delete[] output;


	
	// transposed matrix times matrix
	curandGenerateUniform(gen, C, a * c);
	curandGenerateUniform(gen, A, a * b);

	iterations = maxIterations;
	cudaEventRecord(start, 0);
	while (iterations--) {
		cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, c, b, a, &alpha, C, c, A, b, &beta, B, c);
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout << "Time: " << elapsedTime / maxIterations << " ms" << endl;
	
	cudaMemcpy(hA, A, a * b * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(hB, B, b * c * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(hC, C, a * c * sizeof(float), cudaMemcpyDeviceToHost);
	
	output = new float[b * c];
	for (int i = 0; i < b; i++) {
		for (int j = 0; j < c; j++) {
			float sum = 0;
			for (int k = 0; k < a; k++) {
				sum += hA[k * b + i] * hC[k * c + j];
			}
			output[i * c + j] = sum;
		}
	}
	
	error = 0;
	for (int i = 0; i < b * c; i++) {
		error += abs(hB[i] - output[i]);
	}
	cout << "Average error: " << error / (b * c) << endl;

	delete[] output;

	
	
	// matrix times transposed matrix
	curandGenerateUniform(gen, C, a * c);
	curandGenerateUniform(gen, B, b * c);
	
	iterations = maxIterations;
	cudaEventRecord(start, 0);
	while (iterations--) {
		cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, b, a, c, &alpha, B, c, C, c, &beta, A, b);
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout << "Time: " << elapsedTime / maxIterations << " ms" << endl;
	
	cudaMemcpy(hA, A, a * b * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(hB, B, b * c * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(hC, C, a * c * sizeof(float), cudaMemcpyDeviceToHost);
	
	output = new float[a * b];
	for (int i = 0; i < a; i++) {
		for (int j = 0; j < b; j++) {
			float sum = 0;
			for (int k = 0; k < c; k++) {
				sum += hC[i * c + k] * hB[j * c + k];
			}
			output[i * b + j] = sum;
		}
	}
	
	error = 0;
	for (int i = 0; i < a * b; i++) {
		error += abs(hA[i] - output[i]);
	}
	cout << "Average error: " << error / (a * b) << endl;

	delete[] output;
	

	
	cudaFree(A);
	cudaFree(B);
	cudaFree(C);
	delete[] hA;
	delete[] hB;
	delete[] hC;
	
	curandDestroyGenerator(gen);
	cublasDestroy(handle);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return 0;
}