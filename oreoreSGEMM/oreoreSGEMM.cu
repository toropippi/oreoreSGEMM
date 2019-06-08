#include <stdio.h>
#include <time.h>

#define TSK 16
#define WPTM 8
#define WPTN 8
#define TSM (TSK * WPTM)
#define TSN (TSK * WPTN)
#define RTSM (TSM/WPTM)
#define RTSN (TSN/WPTN)
#define LPTA (TSK*TSM)
#define LPTB (TSK*TSN)

// Use 2D register blocking (further increase in work per thread)




//C=A*B
__global__ void oreoreSGEMM(int M, int N, int K, float* A, float* B, float* C) {
	// Thread identifiers
	int tidm = threadIdx.x; // Local row ID (max: TSM/WPTM)
	int tidn = threadIdx.y; // Local col ID (max: TSN/WPTN)

	int offsetM = TSM * blockIdx.x + tidm; // Work-group offset
	int offsetN = TSN * blockIdx.y + tidn; // Work-group offset

	// Local memory to fit a tile of A and B
	//2 is to avoid bank conflict ?
	__shared__ float4 Asub[TSM*TSK / 4];
	__shared__ float4 Bsub[TSK*TSN / 4];


	// Allocate register space
	float4 Areg;
	float4 Breg[2];
	float acc[WPTM*WPTN];

	// Initialise the accumulation registers
	for (int wm = 0; wm < WPTM; wm++) {
		for (int wn = 0; wn < WPTN; wn++) {
			acc[wm * 8 + wn] = 0.0f;
		}
	}

	// Loop over all tiles
	int numTiles = K / TSK;
	int tid = tidn * 16 + tidm;
	int Boffset = tidn / 2 * N + (tidn % 2) * 64 + offsetM;   //+TSK*t*N+la*8*N
	int Aoffset = tidm + offsetN * K;//+TSK*t+0*K
	for (int t = 0; t < numTiles; t++) {
		// Load one tile of A and B into local memory
		//A 
		float4 dt;
		dt.x = A[Aoffset]; Aoffset += 16 * K;
		dt.y = A[Aoffset]; Aoffset += 16 * K;
		dt.z = A[Aoffset]; Aoffset += 16 * K;
		dt.w = A[Aoffset]; Aoffset += 16 * K;
		Asub[tid] = dt;
		dt.x = A[Aoffset]; Aoffset += 16 * K;
		dt.y = A[Aoffset]; Aoffset += 16 * K;
		dt.z = A[Aoffset]; Aoffset += 16 * K;
		dt.w = A[Aoffset]; Aoffset -= 112 * K - 16;
		Asub[tid + 256] = dt;

		//B
		dt.x = B[Boffset];
		dt.y = B[Boffset + 16];
		dt.z = B[Boffset + 32];
		dt.w = B[Boffset + 48];
		Bsub[tid] = dt;
		Boffset += 8 * N;
		dt.x = B[Boffset];
		dt.y = B[Boffset + 16];
		dt.z = B[Boffset + 32];
		dt.w = B[Boffset + 48];
		Bsub[tid + 256] = dt;
		Boffset += 8 * N;

		// Synchronise to make sure the tile is loaded
		__syncthreads();


		int tidmk = tidm;//+k*TSM
		int tidnk = tidn * 16;
		// Loop over the values of a single tile
		for (int k = 0; k < TSK; k++) {
			// Cache the values of Bsub in registers
			Breg[0] = Bsub[tidmk]; tidmk += 16;
			Breg[1] = Bsub[tidmk]; tidmk += 16;
			// Perform the computation
			Areg = Asub[tidnk]; tidnk += 256;
			acc[0] += Areg.x * Breg[0].x;
			acc[1] += Areg.x * Breg[0].y;
			acc[2] += Areg.x * Breg[0].z;
			acc[3] += Areg.x * Breg[0].w;
			acc[4] += Areg.x * Breg[1].x;
			acc[5] += Areg.x * Breg[1].y;
			acc[6] += Areg.x * Breg[1].z;
			acc[7] += Areg.x * Breg[1].w;

			acc[8 + 0] += Areg.y * Breg[0].x;
			acc[8 + 1] += Areg.y * Breg[0].y;
			acc[8 + 2] += Areg.y * Breg[0].z;
			acc[8 + 3] += Areg.y * Breg[0].w;
			acc[8 + 4] += Areg.y * Breg[1].x;
			acc[8 + 5] += Areg.y * Breg[1].y;
			acc[8 + 6] += Areg.y * Breg[1].z;
			acc[8 + 7] += Areg.y * Breg[1].w;

			acc[16 + 0] += Areg.z * Breg[0].x;
			acc[16 + 1] += Areg.z * Breg[0].y;
			acc[16 + 2] += Areg.z * Breg[0].z;
			acc[16 + 3] += Areg.z * Breg[0].w;
			acc[16 + 4] += Areg.z * Breg[1].x;
			acc[16 + 5] += Areg.z * Breg[1].y;
			acc[16 + 6] += Areg.z * Breg[1].z;
			acc[16 + 7] += Areg.z * Breg[1].w;

			acc[24 + 0] += Areg.w * Breg[0].x;
			acc[24 + 1] += Areg.w * Breg[0].y;
			acc[24 + 2] += Areg.w * Breg[0].z;
			acc[24 + 3] += Areg.w * Breg[0].w;
			acc[24 + 4] += Areg.w * Breg[1].x;
			acc[24 + 5] += Areg.w * Breg[1].y;
			acc[24 + 6] += Areg.w * Breg[1].z;
			acc[24 + 7] += Areg.w * Breg[1].w;


			Areg = Asub[tidnk]; tidnk -= 255;
			acc[32 + 0] += Areg.x * Breg[0].x;
			acc[32 + 1] += Areg.x * Breg[0].y;
			acc[32 + 2] += Areg.x * Breg[0].z;
			acc[32 + 3] += Areg.x * Breg[0].w;
			acc[32 + 4] += Areg.x * Breg[1].x;
			acc[32 + 5] += Areg.x * Breg[1].y;
			acc[32 + 6] += Areg.x * Breg[1].z;
			acc[32 + 7] += Areg.x * Breg[1].w;

			acc[40 + 0] += Areg.y * Breg[0].x;
			acc[40 + 1] += Areg.y * Breg[0].y;
			acc[40 + 2] += Areg.y * Breg[0].z;
			acc[40 + 3] += Areg.y * Breg[0].w;
			acc[40 + 4] += Areg.y * Breg[1].x;
			acc[40 + 5] += Areg.y * Breg[1].y;
			acc[40 + 6] += Areg.y * Breg[1].z;
			acc[40 + 7] += Areg.y * Breg[1].w;

			acc[48 + 0] += Areg.z * Breg[0].x;
			acc[48 + 1] += Areg.z * Breg[0].y;
			acc[48 + 2] += Areg.z * Breg[0].z;
			acc[48 + 3] += Areg.z * Breg[0].w;
			acc[48 + 4] += Areg.z * Breg[1].x;
			acc[48 + 5] += Areg.z * Breg[1].y;
			acc[48 + 6] += Areg.z * Breg[1].z;
			acc[48 + 7] += Areg.z * Breg[1].w;

			acc[56 + 0] += Areg.w * Breg[0].x;
			acc[56 + 1] += Areg.w * Breg[0].y;
			acc[56 + 2] += Areg.w * Breg[0].z;
			acc[56 + 3] += Areg.w * Breg[0].w;
			acc[56 + 4] += Areg.w * Breg[1].x;
			acc[56 + 5] += Areg.w * Breg[1].y;
			acc[56 + 6] += Areg.w * Breg[1].z;
			acc[56 + 7] += Areg.w * Breg[1].w;
		}

		// Synchronise before loading the next tile
		__syncthreads();
	}

	// Store the final results in C
	for (int wm = 0; wm < WPTM; wm++) {
		int globalRow = offsetM + wm * RTSM;
		for (int wn = 0; wn < WPTN; wn++) {
			int globalCol = offsetN + wn * RTSN;
			C[globalCol*M + globalRow] = acc[wn * 8 + wm];
		}
	}

}





//乱数生成
__host__ void Generaterand(float* h_,int nr_rows_ ,int nr_cols_) {
	//行列をA[y][x]としたときxで連続。yは行目、xは列目をあらわす
	for (int i = 0; i < nr_rows_; i++) {
		for (int j = 0; j < nr_cols_; j++) {
			h_[i*nr_cols_ + j] = (float)rand()*0.000030517578125f;
		}
	}
}


//Print matrix C
__host__ void print_matrix(const float* A, int nr_rows_A, int nr_cols_A, const float* B, int nr_rows_B, int nr_cols_B, const float* C, int nr_rows_C, int nr_cols_C) {
	float cc = 0.0f;
	for (int i = 0; i < nr_cols_A; i++) {
		cc += A[i] * B[i*nr_cols_B];
	}
	printf("out check\n");
	printf("CPU_C[0] %f\n", cc);
	printf("GPU_C[0] %f\n", C[0]);
}





int main() {
	int N = 4096*(time(NULL)%4+1);
	printf("start N=%d\n", N);
	// Allocate 3 arrays on CPU
	int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_C, nr_cols_C;
	nr_rows_A = nr_cols_A = nr_rows_B = nr_cols_B = nr_rows_C = nr_cols_C = N;
	float* const h_A = (float *)malloc(nr_rows_A * nr_cols_A * sizeof(float));
	float* const h_B = (float *)malloc(nr_rows_B * nr_cols_B * sizeof(float));
	float* const h_C = (float *)malloc(nr_rows_C * nr_cols_C * sizeof(float));
	// Allocate 3 arrays on GPU
	float *d_A, * d_B, * d_C;
	cudaMalloc(&d_A, nr_rows_A * nr_cols_A * sizeof(float));
	cudaMalloc(&d_B, nr_rows_B * nr_cols_B * sizeof(float));
	cudaMalloc(&d_C, nr_rows_C * nr_cols_C * sizeof(float));
	//rand
	Generaterand(h_A, nr_rows_A, nr_cols_A);
	Generaterand(h_B, nr_rows_B, nr_cols_B);
	//HostToDevice
	cudaMemcpy(d_A, h_A, nr_rows_A * nr_cols_A * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, nr_rows_B * nr_cols_B * sizeof(float), cudaMemcpyHostToDevice);

	// 初期化
	cudaEvent_t start, stop;//時間計測用
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//初回カーネル起動
	dim3 block(TSM / WPTM, TSN / WPTN);
	dim3 grid(N / TSM, N / TSN);
	oreoreSGEMM <<<grid, block >>> (nr_rows_A, nr_cols_B, nr_cols_A, d_A, d_B, d_C);
	cudaDeviceSynchronize();//まち

	int loops = 65536/ N; //SGEMM施行回数
	cudaEventRecord(start);
	for (int i = 0; i < loops; i++) {
		oreoreSGEMM <<<grid, block >>> (nr_rows_A, nr_cols_B, nr_cols_A, d_A, d_B, d_C);
	}
	//cudaDeviceSynchronize();
	// 終了時間を記録
	cudaEventRecord(stop);
	//イベントの終了を待つ。
	cudaEventSynchronize(stop);

	printf("end\n");

	float elapsed;
	cudaEventElapsedTime(&elapsed, start, stop);
	printf( "Time: %fms, %f TFLOPS\n", elapsed, (double)N*N*N*2 / elapsed / 1000000000* loops);

	//結果確認
	cudaMemcpy(h_C, d_C, nr_rows_C * nr_cols_C * sizeof(float), cudaMemcpyDeviceToHost);
	print_matrix(h_A,nr_rows_A,nr_cols_A,h_B,nr_rows_B,nr_cols_B,h_C,nr_rows_C,nr_cols_C);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	return 0;
}