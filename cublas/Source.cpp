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

/*
write a program that uses cublas to multiply two matrices
A x B = C
a x b * b x c = a x c
*/

int main() {
	// initialize random number generator
	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, duration_cast<nanoseconds>(high_resolution_clock::now().time_since_epoch()).count());

	// initialize cublas
	cublasHandle_t handle;
	cublasCreate(&handle);

	// initialize matrices
	int a = 1 << 12;
	int b = 1 << 12;
	int c = 1 << 12;
	float* A, * B, * C;
	cudaMallocManaged(&A, a * b * sizeof(float));
	cudaMallocManaged(&B, b * c * sizeof(float));
	cudaMallocManaged(&C, a * c * sizeof(float));

	// fill matrices with random numbers
	curandGenerateUniform(gen, A, a * b);
	curandGenerateUniform(gen, B, b * c);

	// multiply matrices
	float elapsedTime;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventRecord(start, 0);
	
	float alpha = 1.0f;
	float beta = 0.0f;
	int iterations = 10;
	while (iterations--) {
		cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, c, a, b, &alpha, B, c, A, b, &beta, C, c);
	}
	
	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout << "Time: " << elapsedTime / 10 << " ms" << endl;

	// copy to host
	float* hC = new float[a * c];
	cudaMemcpy(hC, C, a * c * sizeof(float), cudaMemcpyDeviceToHost);
	float* hA = new float[a * b];
	cudaMemcpy(hA, A, a * b * sizeof(float), cudaMemcpyDeviceToHost);
	float* hB = new float[b * c];
	cudaMemcpy(hB, B, b * c * sizeof(float), cudaMemcpyDeviceToHost);

	// calculate result on host
	float* hC2 = new float[a * c];
	for (int i = 0; i < a; i++) {
		for (int j = 0; j < c; j++) {
			float sum = 0;
			for (int k = 0; k < b; k++) {
				sum += hA[i * b + k] * hB[k * c + j];
			}
			hC2[i * c + j] = sum;
		}
	}
	
	// print average error
	float error = 0;
	for (int i = 0; i < a * c; i++) {
		error += abs(hC[i] - hC2[i]);
	}
	cout << "Average error: " << error / (a * c) << endl;

	/*cout << "A:" << endl;
	for (int i = 0; i < a; i++) {
		for (int j = 0; j < b; j++) {
			cout << hA[i * b + j] << " ";
		}
		cout << endl;
	}
	cout << "B:" << endl;
	for (int i = 0; i < b; i++) {
		for (int j = 0; j < c; j++) {
			cout << hB[i * c + j] << " ";
		}
		cout << endl;
	}*/

	// free memory
	cudaFree(A);
	cudaFree(B);
	cudaFree(C);
	delete[] hC;
	delete[] hA;
	delete[] hB;
	delete[] hC2;
	cublasDestroy(handle);
	curandDestroyGenerator(gen);

	return 0;
}