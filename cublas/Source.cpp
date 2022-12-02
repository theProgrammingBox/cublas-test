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

const float ONE = 1.0f;
const float ZERO = 0.0f;

void MatMulMat(cublasHandle_t handle, int a, int b, int c, float* A, float* B, float* C) {
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, c, a, b, &ONE, B, c, A, b, &ZERO, C, c);
}

void MatTMulMat(cublasHandle_t handle, int a, int b, int c, float* A, float* B, float* C) {
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, c, a, b, &ONE, B, c, A, a, &ZERO, C, c);
}

void MatMulMatT(cublasHandle_t handle, int a, int b, int c, float* A, float* B, float* C) {
	cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, c, a, b, &ONE, B, b, A, b, &ZERO, C, c);
}

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
	
	const int numIterations = 100;

	/*int batchSize = 1 << 10;
	int inputFeatures = 1 << 9;
	int outputFeatures = 1 << 8;

	int batchSize = 1 << 10;
	int inputFeatures = 1 << 8;
	int outputFeatures = 1 << 9;

	int batchSize = 1 << 8;
	int inputFeatures = 1 << 10;
	int outputFeatures = 1 << 9;

	int batchSize = 1 << 8;
	int inputFeatures = 1 << 9;
	int outputFeatures = 1 << 10;

	int batchSize = 1 << 9;
	int inputFeatures = 1 << 8;
	int outputFeatures = 1 << 10;*/

	int batchSize = 1 << 9;
	int inputFeatures = 1 << 10;
	int outputFeatures = 1 << 8;/**/
	
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
		// ab x bc = ac
		// abc, ABC
		MatMulMat(handle, batchSize, inputFeatures, outputFeatures, gpuInputMatrix, gpuWeightMatrix, gpuOutputMatrix);
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
		// ba x ac = bc
		// bac, ACB
		MatTMulMat(handle, inputFeatures, batchSize, outputFeatures, gpuInputMatrix, gpuOutputMatrix, gpuWeightMatrix);
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
		// ac x cb = ab
		// acb, CBA
		MatMulMatT(handle, batchSize, outputFeatures, inputFeatures, gpuOutputMatrix, gpuWeightMatrix, gpuInputMatrix);
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