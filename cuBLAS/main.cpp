#include <iostream>
//#include "cuda_runtime.h"
#include <cublas_v2.h>
#include <curand.h>


// Fill the array A(nr_rows_A, nr_cols_A) with random numbers on GPU
void GPU_fill_rand(float* A, int nr_rows_A, int nr_cols_A) {
	// Create a pseudo-random number generator
	curandGenerator_t prng;
	curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
	// Set the seed for the random number generator using the system clock
	curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)rand());
	// Fill the array with random numbers on the device
	curandGenerateUniform(prng, A, nr_rows_A * nr_cols_A);
}

// Multiply the arrays A and B on GPU and save the result in C
// C(m,n) = A(m,k) * B(k,n)
void gpu_blas_mmul(cublasHandle_t& handle, const float* A, const float* B, float* C, const int m, const int k, const int n) {
	int lda = m, ldb = k, ldc = m;
	const float alf = 1.0f;
	const float bet = 0.0f;
	const float* alpha = &alf;
	const float* beta = &bet;
	// Do the actual multiplication
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

//Print matrix C
void print_matrix(const float* A, int nr_rows_A, int nr_cols_A,const float* B, int nr_rows_B, int nr_cols_B,const float* C, int nr_rows_C, int nr_cols_C) {
	float* h_A = new float[nr_rows_A*nr_cols_A];
	float* h_B = new float[nr_rows_B*nr_cols_B];
	float* h_C = new float[nr_rows_C*nr_cols_C];
	cudaMemcpy(h_A, A, nr_rows_A*nr_cols_A * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_B, B, nr_rows_B*nr_cols_B * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_C, C, nr_rows_C*nr_cols_C * sizeof(float), cudaMemcpyDeviceToHost);
	float cc = 0.0f;
	for (int i = 0; i < nr_cols_A; i++) {
		cc += h_A[i*nr_rows_A] * h_B[i];
	}
	std::cout << "out check" << std::endl;
	std::cout << "CPU_C[0] " << cc << std::endl;
	std::cout << "GPU_C[0] " << h_C[0] << std::endl;
}

int main() {
	// Allocate 3 arrays on CPU
	int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_C, nr_cols_C;
	// for simplicity we are going to use square arrays
	nr_rows_A = nr_cols_A = nr_rows_B = nr_cols_B = nr_rows_C = nr_cols_C = 4096*4;
	float* d_A, * d_B, * d_C;
	cudaMalloc(&d_A, nr_rows_A * nr_cols_A * sizeof(float));
	cudaMalloc(&d_B, nr_rows_B * nr_cols_B * sizeof(float));
	cudaMalloc(&d_C, nr_rows_C * nr_cols_C * sizeof(float));

	// 初期化
	cudaEvent_t start, stop;//時間計測用
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cublasHandle_t handle;//一応必要？
	cublasCreate(&handle);
	int loop = 2;//SGEMM施行回数

	// Fill the arrays A and B on GPU with random numbers
	GPU_fill_rand(d_A, nr_rows_A, nr_cols_A);
	GPU_fill_rand(d_B, nr_rows_B, nr_cols_B);
	
	//初回カーネル起動
	gpu_blas_mmul(handle, d_A, d_B, d_C, nr_rows_A, nr_cols_A, nr_cols_B);
	cudaDeviceSynchronize();//同期命令
	
	//計測開始
	std::cout << "start N=" << nr_rows_A << std::endl;
	cudaEventRecord(start);
	for (int i = 0; i < loop; i++) {
		gpu_blas_mmul(handle, d_A, d_B, d_C, nr_rows_A, nr_cols_A, nr_cols_B);
	}
	// 終了時間を記録
	cudaEventRecord(stop);
	//イベントの終了を待つ。
	cudaEventSynchronize(stop);

	// ms単位でstartとstopの差を計算する。24h
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	// 終了処理
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cublasDestroy(handle);

	//Print the result
	std::cout << "end" << std::endl;
	std::cout << "TIME(ms) =" << milliseconds << std::endl;
	std::cout << ((double)nr_rows_A * nr_rows_A * nr_rows_A * 2 / 1000 / 1000 / 1000 / milliseconds * loop) << "TFLOPS" << std::endl;

	//一応結果確認
	print_matrix(d_A, nr_rows_A, nr_cols_A, d_B, nr_rows_B, nr_cols_B, d_C, nr_rows_C, nr_cols_C);

	//終了
	std::cout << std::endl;
	std::cout << "Anykey and Enter to exit" << std::endl;
	std::cin >> loop;
	return 0;
}