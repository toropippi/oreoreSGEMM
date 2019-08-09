#https://cnugteren.github.io/tutorial/pages/page8.htmlを参照
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import numpy as np
import time
np.set_printoptions(suppress=True)#強制的に小数表記

TSK=16#                 // The tile-size in dimension K
WPTM=8#                 // The work-per-thread in dimension M
WPTN=8#                 // The work-per-thread in dimension N
TSM=TSK*WPTM #128           // The tile-size in dimension M
TSN=TSK*WPTN #128            // The tile-size in dimension N
RTSM=(TSM//WPTM)#16        // The reduced tile-size in dimension M
RTSN=(TSN//WPTN)#16        // The reduced tile-size in dimension N
LPTA=((TSK*TSM)//(RTSM*RTSN))#8 // Loads-per-thread for A
LPTB=((TSK*TSN)//(RTSM*RTSN))#8 // Loads-per-thread for B

source="""
#define TSM """+str(TSM)+"""
#define TSN """+str(TSN)+"""
#define TSK """+str(TSK)+"""
#define WPTM """+str(WPTM)+"""
#define WPTN """+str(WPTN)+"""
#define RTSM (TSM/WPTM)        // The reduced tile-size in dimension M
#define RTSN (TSN/WPTN)        // The reduced tile-size in dimension N
#define LPTA ((TSK*TSM)/(RTSM*RTSN)) // Loads-per-thread for A
#define LPTB ((TSK*TSN)/(RTSM*RTSN)) // Loads-per-thread for B

__global__ void copypasteSGEMM(int M,int N,int K,float* A,float* B,float* C) {
    // Thread identifiers
    int tidm = threadIdx.x; // Local row ID (max: TSM/WPTM)
    int tidn = threadIdx.y; // Local col ID (max: TSN/WPTN)
    int offsetM = TSM*blockIdx.x; // Work-group offset
    int offsetN = TSN*blockIdx.y; // Work-group offset
 
    // Local memory to fit a tile of A and B
    //2 is to avoid bank conflict ?
    __shared__ float Asub[TSK][TSM];
    __shared__ float Bsub[TSN][TSK+2];
    
 
    // Allocate register space
    float Areg;
    float Breg[WPTN];
    float acc[WPTM][WPTN];
 
    // Initialise the accumulation registers
    for (int wm=0; wm<WPTM; wm++) {
        for (int wn=0; wn<WPTN; wn++) {
            acc[wm][wn] = 0.0f;
        }
    }
    
    // Loop over all tiles
    int numTiles = K/TSK;
    for (int t=0; t<numTiles; t++) {
 
        // Load one tile of A and B into local memory
        for (int la=0; la<LPTA; la++) {
            int tid = tidn*RTSM + tidm;
            int id = la*RTSN*RTSM + tid;
            int row = id % TSM;
            int col = id / TSM;
            int tiledIndex = TSK*t + col;
            Asub[col][row] = A[tiledIndex*M + offsetM + row];
            Bsub[row][col] = B[tiledIndex*N + offsetN + row];
        }
        
        // Synchronise to make sure the tile is loaded
        __syncthreads();
 
        // Loop over the values of a single tile
        for (int k=0; k<TSK; k++) {
 
            // Cache the values of Bsub in registers
            for (int wn=0; wn<WPTN; wn++) {
                int col = tidn + wn*RTSN;
                Breg[wn] = Bsub[col][k];
            }
 
            // Perform the computation
            for (int wm=0; wm<WPTM; wm++) {
                int row = tidm + wm*RTSM;
                Areg = Asub[k][row];
                for (int wn=0; wn<WPTN; wn++) {
                    acc[wm][wn] += Areg * Breg[wn];
                }
            }
        }
 
        // Synchronise before loading the next tile
        __syncthreads();
    }
 
    // Store the final results in C
    for (int wm=0; wm<WPTM; wm++) {
        int globalRow = offsetM + tidm + wm*RTSM;
        for (int wn=0; wn<WPTN; wn++) {
            int globalCol = offsetN + tidn + wn*RTSN;
            C[globalCol*M + globalRow] = acc[wm][wn];
        }
    }
}


//C=A*B
__global__ void oreoreSGEMM(int M,int N,int K,float* A,float* B,float* C) {
    // Thread identifiers
    int tidm = threadIdx.x; // Local row ID (max: TSM/WPTM)
    int tidn = threadIdx.y; // Local col ID (max: TSN/WPTN)
    int offsetM = TSM*blockIdx.x+tidm; // Work-group offset
    int offsetN = TSN*blockIdx.y+tidn; // Work-group offset
 
    // Local memory to fit a tile of A and B
    //2 is to avoid bank conflict ?
    __shared__ float4 Asub[TSM*TSK/4];
    __shared__ float4 Bsub[TSK*TSN/4];
    
 
    // Allocate register space
    float4 Areg;
    float4 Breg[2];
    float acc[WPTM*WPTN];
 
    // Initialise the accumulation registers
    for (int wm=0; wm<WPTM; wm++) {
        for (int wn=0; wn<WPTN; wn++) {
            acc[wm*8+wn] = 0.0f;
        }
    }
    
    // Loop over all tiles
    int numTiles = K/TSK;
    int tid = tidn*16 + tidm;
    int Boffset=tidn/2*N+(tidn%2)*64+offsetM;   //+TSK*t*N+la*8*N
    int Aoffset=tidm+offsetN*K;//+TSK*t+0*K
    for (int t=0; t<numTiles; t++) {
        // Load one tile of A and B into local memory
        //A 
        float4 dt;
        dt.x=A[Aoffset];Aoffset+=16*K; 
        dt.y=A[Aoffset];Aoffset+=16*K;
        dt.z=A[Aoffset];Aoffset+=16*K;
        dt.w=A[Aoffset];Aoffset+=16*K;
        Asub[tid]=dt;
        dt.x=A[Aoffset];Aoffset+=16*K;
        dt.y=A[Aoffset];Aoffset+=16*K;
        dt.z=A[Aoffset];Aoffset+=16*K;
        dt.w=A[Aoffset];Aoffset-=112*K-16;
        Asub[tid+256]=dt;
        
        //B
        dt.x=B[Boffset];
        dt.y=B[Boffset+16];
        dt.z=B[Boffset+32];
        dt.w=B[Boffset+48];
        Bsub[tid]=dt;
        Boffset+=8*N;
        dt.x=B[Boffset];
        dt.y=B[Boffset+16];
        dt.z=B[Boffset+32];
        dt.w=B[Boffset+48];
        Bsub[tid+256]=dt;
        Boffset+=8*N;
        
        // Synchronise to make sure the tile is loaded
        __syncthreads();
 
        
        int tidmk=tidm;//+k*TSM
        int tidnk=tidn*16;
        // Loop over the values of a single tile
        for (int k=0; k<TSK; k++) {
            // Cache the values of Bsub in registers
            Breg[0] = Bsub[tidmk];tidmk+=16;
            Breg[1] = Bsub[tidmk];tidmk+=16;
            // Perform the computation
                Areg = Asub[tidnk];tidnk+=256;
                acc[0] += Areg.x * Breg[0].x;
                acc[1] += Areg.x * Breg[0].y;
                acc[2] += Areg.x * Breg[0].z;
                acc[3] += Areg.x * Breg[0].w;
                acc[4] += Areg.x * Breg[1].x;
                acc[5] += Areg.x * Breg[1].y;
                acc[6] += Areg.x * Breg[1].z;
                acc[7] += Areg.x * Breg[1].w;
                
                acc[8+0] += Areg.y * Breg[0].x;
                acc[8+1] += Areg.y * Breg[0].y;
                acc[8+2] += Areg.y * Breg[0].z;
                acc[8+3] += Areg.y * Breg[0].w;
                acc[8+4] += Areg.y * Breg[1].x;
                acc[8+5] += Areg.y * Breg[1].y;
                acc[8+6] += Areg.y * Breg[1].z;
                acc[8+7] += Areg.y * Breg[1].w;
                
                acc[16+0] += Areg.z * Breg[0].x;
                acc[16+1] += Areg.z * Breg[0].y;
                acc[16+2] += Areg.z * Breg[0].z;
                acc[16+3] += Areg.z * Breg[0].w;
                acc[16+4] += Areg.z * Breg[1].x;
                acc[16+5] += Areg.z * Breg[1].y;
                acc[16+6] += Areg.z * Breg[1].z;
                acc[16+7] += Areg.z * Breg[1].w;
                
                acc[24+0] += Areg.w * Breg[0].x;
                acc[24+1] += Areg.w * Breg[0].y;
                acc[24+2] += Areg.w * Breg[0].z;
                acc[24+3] += Areg.w * Breg[0].w;
                acc[24+4] += Areg.w * Breg[1].x;
                acc[24+5] += Areg.w * Breg[1].y;
                acc[24+6] += Areg.w * Breg[1].z;
                acc[24+7] += Areg.w * Breg[1].w;
                
                
                Areg = Asub[tidnk];tidnk-=255;
                acc[32+0] += Areg.x * Breg[0].x;
                acc[32+1] += Areg.x * Breg[0].y;
                acc[32+2] += Areg.x * Breg[0].z;
                acc[32+3] += Areg.x * Breg[0].w;
                acc[32+4] += Areg.x * Breg[1].x;
                acc[32+5] += Areg.x * Breg[1].y;
                acc[32+6] += Areg.x * Breg[1].z;
                acc[32+7] += Areg.x * Breg[1].w;
                
                acc[40+0] += Areg.y * Breg[0].x;
                acc[40+1] += Areg.y * Breg[0].y;
                acc[40+2] += Areg.y * Breg[0].z;
                acc[40+3] += Areg.y * Breg[0].w;
                acc[40+4] += Areg.y * Breg[1].x;
                acc[40+5] += Areg.y * Breg[1].y;
                acc[40+6] += Areg.y * Breg[1].z;
                acc[40+7] += Areg.y * Breg[1].w;
                
                acc[48+0] += Areg.z * Breg[0].x;
                acc[48+1] += Areg.z * Breg[0].y;
                acc[48+2] += Areg.z * Breg[0].z;
                acc[48+3] += Areg.z * Breg[0].w;
                acc[48+4] += Areg.z * Breg[1].x;
                acc[48+5] += Areg.z * Breg[1].y;
                acc[48+6] += Areg.z * Breg[1].z;
                acc[48+7] += Areg.z * Breg[1].w;
                
                acc[56+0] += Areg.w * Breg[0].x;
                acc[56+1] += Areg.w * Breg[0].y;
                acc[56+2] += Areg.w * Breg[0].z;
                acc[56+3] += Areg.w * Breg[0].w;
                acc[56+4] += Areg.w * Breg[1].x;
                acc[56+5] += Areg.w * Breg[1].y;
                acc[56+6] += Areg.w * Breg[1].z;
                acc[56+7] += Areg.w * Breg[1].w;
        }
 
        // Synchronise before loading the next tile
        __syncthreads();
    }
 
    // Store the final results in C
    for (int wm=0; wm<WPTM; wm++) {
        int globalRow = offsetM + wm*RTSM;
        for (int wn=0; wn<WPTN; wn++) {
            int globalCol = offsetN + wn*RTSN;
            C[globalCol*M + globalRow] = acc[wn*8+wm];
        }
    }
}

"""
#超大事なメモ
#x,列,col
#y,行,row
#array[y][x]でx,y位置の数字を取得
#初期化ももちろんarray[行][列]
#メモリはx方向に連番 array[2][2]とarray[2][3]は連番
#つまりrow major ???
#copypasteSGEMMはB.T*A、oreoreSGEMMはA*Bをやってます

n=4096*4
print("N={0}".format(n))
npn=np.int32(n)

programid = SourceModule(source)
kernel = programid.get_function("oreoreSGEMM")  # 上で定義したカーネルを呼び出す

vram_A = drv.mem_alloc(n*n*4)
vram_B = drv.mem_alloc(n*n*4)
vram_C = drv.mem_alloc(n*n*4)

A=np.random.rand(n*n).astype(np.float32)
A=A.reshape([n,n])
B=np.random.rand(n*n).astype(np.float32)
B=B.reshape([n,n])
C=np.zeros(n*n,dtype=np.float32)

drv.memcpy_htod(vram_A, A)
drv.memcpy_htod(vram_B, B)

#初回カーネル起動
kernel(npn,npn,npn,vram_A,vram_B,vram_C, grid=(n//TSM,n//TSN, 1),block=(TSM // WPTM, TSN // WPTN, 1))
drv.Context.synchronize()#カーネル終了までまつ

loop=2
calct=time.time()
for i in range(loop):
    kernel(npn,npn,npn,vram_A,vram_B,vram_C, grid=(n//TSM,n//TSN, 1),block=(TSM // WPTM, TSN // WPTN, 1))
drv.Context.synchronize()#カーネル終了までまつ

calct=time.time()-calct
print("calc time={0}".format(calct))
print("{0}TFLOPS".format(n*n*n*2/calct/1000/1000/1000/1000*loop))

#ここからは正しく計算できてるかチェック
drv.memcpy_dtoh(C,vram_C)
print(C.reshape([n,n]))
err=np.dot(A,B)-C.reshape([n,n])
#err=np.dot(B.T,A)-C.reshape([n,n])#コピペSGEMMのときはここを使う
print("誤差")
print(np.max(err))#誤差max
print(np.min(err))#誤差min

drv.DeviceAllocation.free(vram_C)#これで解放
drv.DeviceAllocation.free(vram_B)
drv.DeviceAllocation.free(vram_A)