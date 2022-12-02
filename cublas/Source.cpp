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
	curandGenerator_t randomGenerator;
	curandCreateGenerator(&randomGenerator, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(randomGenerator, duration_cast<nanoseconds>(high_resolution_clock::now().time_since_epoch()).count());
	
	cublasHandle_t handle;
	cublasCreate(&handle);
	
	float elapsedTime;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	const float ONE = 1.0f;
	const float ZERO = 0.0f;
	const int numIterations = 100;
	
	int batchSize = 1 << 9;
	int inputFeatures = 1 << 10;
	int outputFeatures = 1 << 8;
	
	float* gpuInputMatrix, * gpuWeightMatrix, * gpuOutputMatrix;
	cudaMallocManaged(&gpuInputMatrix, batchSize * inputFeatures * sizeof(float));
	cudaMallocManaged(&gpuWeightMatrix, inputFeatures * outputFeatures * sizeof(float));
	cudaMallocManaged(&gpuOutputMatrix, batchSize * outputFeatures * sizeof(float));
	
	float* cpuInputMatrix = new float[batchSize * inputFeatures];
	float* cpuWeightMatrix = new float[inputFeatures * outputFeatures];
	float* cpuOutputMatrix = new float[batchSize * outputFeatures];

	float* output;
	
	

	// matrix times matrix
	curandGenerateUniform(randomGenerator, gpuInputMatrix, batchSize * inputFeatures);
	curandGenerateUniform(randomGenerator, gpuWeightMatrix, inputFeatures * outputFeatures);

	int iterations = numIterations;
	cudaEventRecord(start, 0);
	while (iterations--) {
		cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, outputFeatures, batchSize, inputFeatures, &ONE, gpuWeightMatrix, outputFeatures, gpuInputMatrix, inputFeatures, &ZERO, gpuOutputMatrix, outputFeatures);
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout << "Time: " << elapsedTime / numIterations << " ms" << endl;
	
	cudaMemcpy(cpuInputMatrix, gpuInputMatrix, batchSize * inputFeatures * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(cpuWeightMatrix, gpuWeightMatrix, inputFeatures * outputFeatures * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(cpuOutputMatrix, gpuOutputMatrix, batchSize * outputFeatures * sizeof(float), cudaMemcpyDeviceToHost);
	
	output = new float[batchSize * outputFeatures];
	for (int i = 0; i < batchSize; i++) {
		for (int j = 0; j < outputFeatures; j++) {
			float sum = 0;
			for (int k = 0; k < inputFeatures; k++) {
				sum += cpuInputMatrix[i * inputFeatures + k] * cpuWeightMatrix[k * outputFeatures + j];
			}
			output[i * outputFeatures + j] = sum;
		}
	}
	
	float error = 0;
	for (int i = 0; i < batchSize * outputFeatures; i++) {
		error += abs(cpuOutputMatrix[i] - output[i]);
	}
	cout << "Average error: " << error / (batchSize * outputFeatures) << endl;
	delete[] output;


	
	// transposed matrix times matrix
	curandGenerateUniform(randomGenerator, gpuOutputMatrix, batchSize * outputFeatures);
	curandGenerateUniform(randomGenerator, gpuInputMatrix, batchSize * inputFeatures);

	iterations = numIterations;
	cudaEventRecord(start, 0);
	while (iterations--) {
		cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, outputFeatures, inputFeatures, batchSize, &ONE, gpuOutputMatrix, outputFeatures, gpuInputMatrix, inputFeatures, &ZERO, gpuWeightMatrix, outputFeatures);
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout << "Time: " << elapsedTime / numIterations << " ms" << endl;
	
	cudaMemcpy(cpuInputMatrix, gpuInputMatrix, batchSize * inputFeatures * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(cpuWeightMatrix, gpuWeightMatrix, inputFeatures * outputFeatures * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(cpuOutputMatrix, gpuOutputMatrix, batchSize * outputFeatures * sizeof(float), cudaMemcpyDeviceToHost);
	
	output = new float[inputFeatures * outputFeatures];
	for (int i = 0; i < inputFeatures; i++) {
		for (int j = 0; j < outputFeatures; j++) {
			float sum = 0;
			for (int k = 0; k < batchSize; k++) {
				sum += cpuInputMatrix[k * inputFeatures + i] * cpuOutputMatrix[k * outputFeatures + j];
			}
			output[i * outputFeatures + j] = sum;
		}
	}
	
	error = 0;
	for (int i = 0; i < inputFeatures * outputFeatures; i++) {
		error += abs(cpuWeightMatrix[i] - output[i]);
	}
	cout << "Average error: " << error / (inputFeatures * outputFeatures) << endl;
	delete[] output;

	
	
	// matrix times transposed matrix
	curandGenerateUniform(randomGenerator, gpuOutputMatrix, batchSize * outputFeatures);
	curandGenerateUniform(randomGenerator, gpuWeightMatrix, inputFeatures * outputFeatures);
	
	iterations = numIterations;
	cudaEventRecord(start, 0);
	while (iterations--) {
		cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, inputFeatures, batchSize, outputFeatures, &ONE, gpuWeightMatrix, outputFeatures, gpuOutputMatrix, outputFeatures, &ZERO, gpuInputMatrix, inputFeatures);
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout << "Time: " << elapsedTime / numIterations << " ms" << endl;
	
	cudaMemcpy(cpuInputMatrix, gpuInputMatrix, batchSize * inputFeatures * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(cpuWeightMatrix, gpuWeightMatrix, inputFeatures * outputFeatures * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(cpuOutputMatrix, gpuOutputMatrix, batchSize * outputFeatures * sizeof(float), cudaMemcpyDeviceToHost);
	
	output = new float[batchSize * inputFeatures];
	for (int i = 0; i < batchSize; i++) {
		for (int j = 0; j < inputFeatures; j++) {
			float sum = 0;
			for (int k = 0; k < outputFeatures; k++) {
				sum += cpuOutputMatrix[i * outputFeatures + k] * cpuWeightMatrix[j * outputFeatures + k];
			}
			output[i * inputFeatures + j] = sum;
		}
	}
	
	error = 0;
	for (int i = 0; i < batchSize * inputFeatures; i++) {
		error += abs(cpuInputMatrix[i] - output[i]);
	}
	cout << "Average error: " << error / (batchSize * inputFeatures) << endl;
	delete[] output;
	

	
	cudaFree(gpuInputMatrix);
	cudaFree(gpuWeightMatrix);
	cudaFree(gpuOutputMatrix);
	delete[] cpuInputMatrix;
	delete[] cpuWeightMatrix;
	delete[] cpuOutputMatrix;
	
	curandDestroyGenerator(randomGenerator);
	cublasDestroy(handle);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return 0;
}