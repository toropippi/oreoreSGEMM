#https://cnugteren.github.io/tutorial/pages/page8.htmlを参照
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import numpy as np
import time
np.set_printoptions(suppress=True)#強制的に小数表記

source="""
#define TSN 128
#define TSM 128
#define TSK 16
#define WPTN 8
#define WPTM 8
#define RTSN (TSN/WPTN)        // The reduced tile-size in dimension N
#define RTSM (TSM/WPTM)        // The reduced tile-size in dimension M
#define LPTA ((TSK*TSN)/(RTSN*RTSM)) // Loads-per-thread for A
#define LPTB ((TSK*TSM)/(RTSN*RTSM)) // Loads-per-thread for B

//C=A*B  only k%16!=0 n>=128 m>=128
__global__ void oreoreSGEMM_a(int M,int N,int K,float* A,float* B,float* C) {
    // Thread identifiers
    int tidn = threadIdx.x; // Local row ID (max: TSN/WPTN)
    int tidm = threadIdx.y; // Local col ID (max: TSM/WPTM)
    int offsetN = TSN*blockIdx.x+tidn; // Work-group offset
    int offsetM = TSM*blockIdx.y+tidm; // Work-group offset
    if (blockIdx.y==M/128) offsetM-=128-M%128;
    if (blockIdx.x==N/128) offsetN-=128-N%128;
    int Boffset=tidm/2*N+(tidm%2)*64+offsetN;
    int Aoffset=tidn+offsetM*K;
 
    // Local memory to fit a tile of A and B
    __shared__ float4 Asub[TSN*TSK/4];
    __shared__ float4 Bsub[TSK*TSM/4];
 
    // Allocate register space
    float4 Areg;
    float4 Breg[2];
    float acc[WPTN*WPTM];
 
    // Initialise the accumulation registers
    for (int wn=0; wn<WPTN; wn++) {
        for (int wm=0; wm<WPTM; wm++) {
            acc[wn*8+wm] = 0.0f;
        }
    }
    
    // Loop over all tiles
    int numTiles = K/TSK;
    int tid = tidm*16 + tidn;
    for (int t=0; t<numTiles; t++) {
        // Load one tile of A and B into local memory
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
        
        dt.x=B[Boffset]; 
        dt.y=B[Boffset+16];
        dt.z=B[Boffset+32];
        dt.w=B[Boffset+48];Boffset+=8*N;
        Bsub[tid]=dt;
        dt.x=B[Boffset];
        dt.y=B[Boffset+16];
        dt.z=B[Boffset+32];
        dt.w=B[Boffset+48];
        Bsub[tid+256]=dt;
        Boffset+=8*N;
        
        // Synchronise to make sure the tile is loaded
        __syncthreads();
        
        int tidnk=tidn;
        int tidmk=tidm*16;
        // Loop over the values of a single tile
        for (int k=0; k<TSK; k++) {
            // Cache the values of Bsub in registers
            Breg[0] = Bsub[tidnk];tidnk+=16;
            Areg = Asub[tidmk];tidmk+=256;
            Breg[1] = Bsub[tidnk];tidnk+=16;
            // Perform the computation
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
            
            
            Areg = Asub[tidmk];tidmk-=255;
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
    
    /////////////////////////////////////////////////////////
    int km=K%16;
    int maxAidx=M*K-1;
    int maxBidx=N*K-1;

    float4 dta;
    float4 dtb;
    Boffset=min(Boffset,maxBidx);
    dta.x=A[Aoffset];Aoffset+=16*K;
    dtb.x=B[Boffset];Boffset=min(Boffset+16,maxBidx); 
    dta.y=A[Aoffset];Aoffset+=16*K;
    dtb.y=B[Boffset];Boffset=min(Boffset+16,maxBidx);
    dta.z=A[Aoffset];Aoffset+=16*K;
    dtb.z=B[Boffset];Boffset=min(Boffset+16,maxBidx);
    dta.w=A[Aoffset];Aoffset+=16*K;
    dtb.w=B[Boffset];Boffset=min(Boffset+8*N-48,maxBidx);
    Asub[tid]=dta;
    Bsub[tid]=dtb;tid+=256;
    dta.x=A[Aoffset];Aoffset+=16*K;
    dtb.x=B[Boffset];Boffset=min(Boffset+16,maxBidx);
    dta.y=A[Aoffset];Aoffset+=16*K;
    dtb.y=B[Boffset];Boffset=min(Boffset+16,maxBidx);
    dta.z=A[Aoffset];Aoffset=min(Aoffset+16*K,maxAidx);
    dtb.z=B[Boffset];Boffset=min(Boffset+16,maxBidx);
    dta.w=A[Aoffset];
    dtb.w=B[Boffset];
    Asub[tid]=dta;
    Bsub[tid]=dtb;
    __syncthreads();
    
    int tidnk=tidn;
    int tidmk=tidm*16;
    for (int k=0; k<km; k++) {
        // Cache the values of Bsub in registers
        Breg[0] = Bsub[tidnk];tidnk+=16;
        Areg = Asub[tidmk];tidmk+=256;
        Breg[1] = Bsub[tidnk];tidnk+=16;
        // Perform the computation
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
        
        Areg = Asub[tidmk];tidmk-=255;
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
    ////////////////////////////////////////////////////////
 
    // Store the final results in C
    
    for (int wn=0; wn<8; wn++) {
        int globalRow = offsetN + wn*RTSN;
        //if (globalRow>=N) break;
        for (int wm=0; wm<8; wm++) {
            int globalCol = offsetM + wm*RTSM;
            //if (globalCol>=M) break;
            C[globalCol*N + globalRow] = acc[wm*8+wn];
        }
    }
}





//C=A*B  only k%16==0 n>=128 m>=128
__global__ void oreoreSGEMM_k(int M,int N,int K,float* A,float* B,float* C) {
    // Thread identifiers
    int tidn = threadIdx.x; // Local row ID (max: TSN/WPTN)
    int tidm = threadIdx.y; // Local col ID (max: TSM/WPTM)
    int offsetN = TSN*blockIdx.x+tidn; // Work-group offset
    int offsetM = TSM*blockIdx.y+tidm; // Work-group offset
    if (blockIdx.y==M/128) offsetM-=128-M%128;
    if (blockIdx.x==N/128) offsetN-=128-N%128;
    int Boffset=tidm/2*N+(tidm%2)*64+offsetN;
    int Aoffset=tidn+offsetM*K;
 
    // Local memory to fit a tile of A and B
    __shared__ float4 Asub[TSN*TSK/4];
    __shared__ float4 Bsub[TSK*TSM/4];
 
    // Allocate register space
    float4 Areg;
    float4 Breg[2];
    float acc[WPTN*WPTM];
 
    // Initialise the accumulation registers
    for (int wn=0; wn<WPTN; wn++) {
        for (int wm=0; wm<WPTM; wm++) {
            acc[wn*8+wm] = 0.0f;
        }
    }
    
    // Loop over all tiles
    int numTiles = K/TSK;
    int tid = tidm*16 + tidn;
    for (int t=0; t<numTiles; t++) {
        // Load one tile of A and B into local memory
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
        
        dt.x=B[Boffset]; 
        dt.y=B[Boffset+16];
        dt.z=B[Boffset+32];
        dt.w=B[Boffset+48];Boffset+=8*N;
        Bsub[tid]=dt;
        dt.x=B[Boffset];
        dt.y=B[Boffset+16];
        dt.z=B[Boffset+32];
        dt.w=B[Boffset+48];
        Bsub[tid+256]=dt;
        Boffset+=8*N;
        
        // Synchronise to make sure the tile is loaded
        __syncthreads();
 
        int tidnk=tidn;
        int tidmk=tidm*16;
        // Loop over the values of a single tile
        for (int k=0; k<TSK; k++) {
            // Cache the values of Bsub in registers
            Breg[0] = Bsub[tidnk];tidnk+=16;
            Areg = Asub[tidmk];tidmk+=256;
            Breg[1] = Bsub[tidnk];tidnk+=16;
            // Perform the computation
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
            
            
            Areg = Asub[tidmk];tidmk-=255;
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
    for (int wn=0; wn<8; wn++) {
        int globalRow = offsetN + wn*RTSN;
        //if (globalRow>=N) break;
        for (int wm=0; wm<8; wm++) {
            int globalCol = offsetM + wm*RTSM;
            //if (globalCol>=M) break;
            C[globalCol*N + globalRow] = acc[wm*8+wn];
        }
    }
}



//C=A*B  only n<128 or m<128
__global__ void oreoreSGEMM_small(int M,int N,int K,float* A,float* B,float* C) {
    // Thread identifiers
    int tidn = threadIdx.x; // Local row ID (max: TSN/WPTN)
    int tidm = threadIdx.y; // Local col ID (max: TSM/WPTM)
    int offsetN = TSN*blockIdx.x+tidn; // Work-group offset
    int offsetM = TSM*blockIdx.y+tidm; // Work-group offset
    int Boffset=tidm/2*N+(tidm%2)*64+offsetN;
    int Aoffset=tidn+offsetM*K;
 
    // Local memory to fit a tile of A and B
    __shared__ float4 Asub[TSN*TSK/4];
    __shared__ float4 Bsub[TSK*TSM/4];
 
    // Allocate register space
    float4 Areg;
    float4 Breg[2];
    float acc[WPTN*WPTM];
 
    // Initialise the accumulation registers
    for (int wn=0; wn<WPTN; wn++) {
        for (int wm=0; wm<WPTM; wm++) {
            acc[wn*8+wm] = 0.0f;
        }
    }
    
    // Loop over all tiles
    int tid = tidm*16 + tidn;
    int maxAidx=M*K-1;
    int maxBidx=N*K-1;
    int nowAoffset=min(Aoffset,maxAidx);
    int nowBoffset=min(Boffset,maxBidx);
    for (int t=0; t<K; t+=16) {
        // Load one tile of A and B into local memory
        //AB load software pipelining
        float4 dta;
        float4 dtb;
        dta.x=A[nowAoffset];nowAoffset=min(Aoffset+16*K,maxAidx);
        dtb.x=B[nowBoffset];nowBoffset=min(Boffset+16,maxBidx); 
        dta.y=A[nowAoffset];nowAoffset=min(Aoffset+32*K,maxAidx);
        dtb.y=B[nowBoffset];nowBoffset=min(Boffset+32,maxBidx);
        dta.z=A[nowAoffset];nowAoffset=min(Aoffset+48*K,maxAidx);
        dtb.z=B[nowBoffset];nowBoffset=min(Boffset+48,maxBidx);
        dta.w=A[nowAoffset];nowAoffset=min(Aoffset+64*K,maxAidx);
        dtb.w=B[nowBoffset];Boffset+=8*N;nowBoffset=min(Boffset,maxBidx);
        Asub[tid]=dta;
        Bsub[tid]=dtb;
        dta.x=A[nowAoffset];nowAoffset=min(Aoffset+80*K,maxAidx);
        dtb.x=B[nowBoffset];nowBoffset=min(Boffset+16,maxBidx);
        dta.y=A[nowAoffset];nowAoffset=min(Aoffset+96*K,maxAidx);
        dtb.y=B[nowBoffset];nowBoffset=min(Boffset+32,maxBidx);
        dta.z=A[nowAoffset];nowAoffset=min(Aoffset+112*K,maxAidx);
        dtb.z=B[nowBoffset];nowBoffset=min(Boffset+48,maxBidx);
        dta.w=A[nowAoffset];Aoffset+=16;nowAoffset=min(Aoffset,maxAidx);
        dtb.w=B[nowBoffset];Boffset+=8*N;nowBoffset=min(Boffset,maxBidx);
        Asub[tid+256]=dta;
        Bsub[tid+256]=dtb;
        
        // Synchronise to make sure the tile is loaded
        __syncthreads();
 
        int tidnk=tidn;
        int tidmk=tidm*16;
        // Loop over the values of a single tile
        int kmin=min(K-t,16);
        for (int k=0; k<kmin; k++) {
            // Cache the values of Bsub in registers
            Breg[0] = Bsub[tidnk];tidnk+=16;
            Areg = Asub[tidmk];tidmk+=256;
            Breg[1] = Bsub[tidnk];tidnk+=16;
            // Perform the computation
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
            
            
            Areg = Asub[tidmk];tidmk-=255;
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
    for (int wn=0; wn<8; wn++) {
        int globalRow = offsetN + wn*RTSN;
        if (globalRow>=N) break;
        for (int wm=0; wm<8; wm++) {
            int globalCol = offsetM + wm*RTSM;
            if (globalCol>=M) break;
            C[globalCol*N + globalRow] = acc[wm*8+wn];
        }
    }
}









//A=A.T (not bank conflict)
__global__ void oreoreTrans(int M,int N,float* A,float* AT) {
    // Thread identifiers
    int tidn = threadIdx.x;
    int tidm = threadIdx.y;
    int tidoffset=(tidn+tidm)%16;
    int offsetN = 16*blockIdx.x+tidn;
    int offsetM = 16*blockIdx.y+tidm;
    offsetN=min(offsetN,N-1);
    offsetM=min(offsetM,M-1);
    int woffsetN = 16*blockIdx.x+tidm;
    int woffsetM = 16*blockIdx.y+tidn;
    woffsetN=min(woffsetN,N-1);
    woffsetM=min(woffsetM,M-1);
    
    // Local memory
    __shared__ float Asub[256];
    
    // load Global to Local
    //Asub[tidn+tidm*16]=A[offsetN+offsetM*N];
    Asub[tidoffset+tidm*16]=A[offsetN+offsetM*N];
    
    __syncthreads();
    
    // Store to AT
    //AT[woffsetN*M+woffsetM]=Asub[tidm+tidn*16];
    AT[woffsetN*M+woffsetM]=Asub[tidoffset+tidn*16];
}

"""
#超大事なメモ
#x,列,col
#y,行,row
#array[y][x]でx,y位置の数字を取得
#初期化ももちろんarray[行][列]
#メモリはx方向に連番 array[2][2]とarray[2][3]は連番
#つまりrow major
#row majorはnumpyと同じ仕様

#配列 A は m x k 行列、配列 B は k x n 行列、配列 C は m x n 行列
#ただしtransA,transBに転置を指定している場合、転置後のサイズがm,n,kとする

programid = SourceModule(source)
kernel_oreoreSGEMM_a=programid.get_function("oreoreSGEMM_a")
kernel_oreoreSGEMM_k=programid.get_function("oreoreSGEMM_k")
kernel_oreoreSGEMM_s=programid.get_function("oreoreSGEMM_small")
kernelt=programid.get_function("oreoreTrans")


def mySGEMM(transA,transB,m,n,k,vram_A,vram_B,vram_C):
    if (n<128)|(m<128):
        kernel = kernel_oreoreSGEMM_s
    else:
        if k%16==0:
            kernel = kernel_oreoreSGEMM_k
        else:
            kernel = kernel_oreoreSGEMM_a
    if transA=='T':
        if transB == 'T':
            #C=A.T*B.T
            vram_AT = drv.mem_alloc(m * k * 4)
            vram_BT = drv.mem_alloc(k * n * 4)
            kernelt(np.int32(k), np.int32(m), vram_A, vram_AT, grid=((m + 15) // 16, (k + 15) // 16, 1),
                    block=(16, 16, 1))
            kernelt(np.int32(n), np.int32(k), vram_B, vram_BT, grid=((k + 15) // 16, (n + 15) // 16, 1),
                    block=(16, 16, 1))
            kernel(np.int32(m), np.int32(n), np.int32(k), vram_AT,vram_BT, vram_C,
                   grid=((n + 127) // 128, (m + 127) // 128, 1),
                   block=(16, 16, 1))
            drv.DeviceAllocation.free(vram_AT)
            drv.DeviceAllocation.free(vram_BT)
        else:
            #C=A.T*B
            vram_AT = drv.mem_alloc(m * k * 4)
            kernelt(np.int32(k), np.int32(m), vram_A, vram_AT, grid=((m + 15) // 16, (k + 15) // 16, 1),
                    block=(16, 16, 1))
            kernel(np.int32(m), np.int32(n), np.int32(k), vram_AT, vram_B, vram_C,
                   grid=((n + 127) // 128, (m + 127) // 128, 1),
                   block=(16, 16, 1))
            drv.DeviceAllocation.free(vram_AT)
    else:
        if transB == 'T':
            #C=A*B.T
            vram_BT = drv.mem_alloc(k * n * 4)
            kernelt(np.int32(n), np.int32(k), vram_B, vram_BT, grid=((k + 15) // 16, (n + 15) // 16, 1),
                    block=(16, 16, 1))
            kernel(np.int32(m), np.int32(n), np.int32(k), vram_A, vram_BT, vram_C,
                   grid=((n + 127) // 128, (m + 127) // 128, 1),
                   block=(16, 16, 1))
            drv.DeviceAllocation.free(vram_BT)
        else:
            #C=A*B
            kernel(np.int32(m), np.int32(n), np.int32(k), vram_A, vram_B, vram_C,
                   grid=((n + 127) // 128, (m + 127) // 128, 1),
                   block=(16, 16, 1))
    return



def CreateABC(transA,transB,m,n,k):
    vram_A = drv.mem_alloc(m*k*4)
    vram_B = drv.mem_alloc(k*n*4)
    vram_C = drv.mem_alloc(m*n*4)

    A=np.random.rand(m*k).astype(np.float32)
    B=np.random.rand(k*n).astype(np.float32)
    C=np.zeros(m*n,dtype=np.float32)

    drv.memcpy_htod(vram_A, A)
    drv.memcpy_htod(vram_B, B)

    if transA=="T":
        A=A.reshape([k,m])
        A=A.T
    else:
        A = A.reshape([m, k])
    if transB=="T":
        B=B.reshape([n,k])
        B=B.T
    else:
        B = B.reshape([k, n])
    return A,B,C,vram_A,vram_B,vram_C




if __name__ == '__main__':
    #適当に初期値を生成
    m = 4096*4  # np.random.randint(4096*4)+1
    n = 4096*4  # np.random.randint(4096*4)+1
    k = 4096*4  # np.random.randint(4096*4)+1

    if np.random.randint(2) == 0:
        atflag = "T"  # 転置フラグ
    else:
        atflag = "N"

    if np.random.randint(2) == 0:
        btflag = "T"  # 転置フラグ
    else:
        btflag = "N"

    print("m={0} n={1} k={2} transA={3} transB={4}".format(m,n,k,atflag,btflag))
    #global mem確保、行列初期データ生成
    A,B,C,vram_A,vram_B,vram_C=CreateABC(transA=atflag,transB=btflag,m=m,n=n,k=k)

    #自作SGEMM
    #初回カーネル起動 初回はオーバーヘッドで正確な時間測定ができないため
    mySGEMM(atflag, btflag, m, n, k, vram_A, vram_B, vram_C)
    drv.Context.synchronize()#カーネル終了までまつ

    #時間計測
    loop=2
    calc=time.time()
    for i in range(loop):
        mySGEMM(atflag, btflag, m, n, k, vram_A, vram_B, vram_C)
    drv.Context.synchronize()#カーネル終了までまつ
    calc=time.time()-calc

    print("{0}TFLOPS".format(n*m*k*2/calc/1000/1000/1000/1000*loop))

    #結果確認
    drv.memcpy_dtoh(C,vram_C)
    err=np.dot(A,B)-C.reshape([m,n])
    print("最終誤差")
    print(np.max(err), np.min(err))