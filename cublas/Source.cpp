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
	/*int N = 1000;
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
	curandGenerateNormal(gen, d_A, N * N, 0, 1);
	curandGenerateNormal(gen, d_B, N * N, 0, 1);

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

	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout << "Time to create handle: " << elapsedTime << " ms" << endl;

	cudaEventCreate(&start);
	cudaEventRecord(start, 0);

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
	free(C);*/

	uint32_t inputEntries = 1000;
	uint32_t inputFeatures = 4096;
	uint32_t outputFeatures = 1000;
	uint32_t inputBytes = inputEntries * inputFeatures * sizeof(float);
	uint32_t weightBytes = inputFeatures * outputFeatures * sizeof(float);
	uint32_t biasBytes = outputFeatures * sizeof(float);
	uint32_t outputBytes = inputEntries * outputFeatures * sizeof(float);
	
	cudaEvent_t start, stop;
	float elapsedTime;
	
	cudaEventCreate(&start);
	cudaEventRecord(start, 0);
	
	float* GPUinput, * GPUweights, * GPUbias, * GPUoutput;
	
	cudaMalloc(&GPUinput, inputBytes);
	cudaMalloc(&GPUweights, weightBytes);
	cudaMalloc(&GPUbias, biasBytes);
	cudaMalloc(&GPUoutput, outputBytes);

	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout << "Time to allocate memory: " << elapsedTime << " ms" << endl;
	
	cudaEventCreate(&start);
	cudaEventRecord(start, 0);

	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, duration_cast<nanoseconds>(high_resolution_clock::now().time_since_epoch()).count());
	curandGenerateNormal(gen, GPUinput, inputEntries * inputFeatures, 0, 1);
	curandGenerateNormal(gen, GPUweights, inputFeatures * outputFeatures, 0, 1);
	curandGenerateNormal(gen, GPUbias, outputFeatures, 0, 1);

	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout << "Time to generate random numbers: " << elapsedTime << " ms" << endl;
	
	cudaEventCreate(&start);
	cudaEventRecord(start, 0);
	
	cublasHandle_t handle;
	cublasCreate(&handle);
	
	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout << "Time to create handle: " << elapsedTime << " ms" << endl;

	cudaEventCreate(&start);
	cudaEventRecord(start, 0);
	
	float alpha = 1.0f;
	float beta = 0.0f;
	
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, inputEntries, outputFeatures, inputFeatures, &alpha, GPUinput, inputEntries, GPUweights, inputFeatures, &beta, GPUoutput, inputEntries);
	
	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout << "Time to perform matrix multiplication: " << elapsedTime << " ms" << endl;
	
	/*cout << "Output:" << endl;
	float* output = (float*)malloc(outputBytes);
	cudaMemcpy(output, GPUoutput, outputBytes, cudaMemcpyDeviceToHost);
	for (int i = 0; i < inputEntries; i++) {
		for (int j = 0; j < outputFeatures; j++) {
			cout << output[i * outputFeatures + j] << " ";
		}
		cout << endl;
	}*/
	
	cudaEventCreate(&start);
	cudaEventRecord(start, 0);
	
	for (int i = 0; i < inputEntries; i++) {
		cublasSaxpy(handle, outputFeatures, &alpha, GPUbias, 1, GPUoutput + i * outputFeatures, 1);
	}
		
	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout << "Time to perform matrix addition: " << elapsedTime << " ms" << endl;

	/*cout << "Output with bias:" << endl;
	cudaMemcpy(output, GPUoutput, outputBytes, cudaMemcpyDeviceToHost);
	for (int i = 0; i < inputEntries; i++) {
		for (int j = 0; j < outputFeatures; j++) {
			cout << output[i * outputFeatures + j] << " ";
		}
		cout << endl;
	}*/

	return 0;
}