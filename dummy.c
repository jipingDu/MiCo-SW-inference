#ifdef RISCV_VEXII
#include "sim_stdlib.h"
#else
#include <stdio.h>
#endif

#include "profile.h"
#include "mico_nn.h"
#include "mico_qnn.h"

#define N 32
#define K 32
#define M 32

int main(){
    
    Tensor2D_Q8 x, w;
    printf("Hello MiCo!\n");

    #ifdef SPRAM
    uint32_t scratch_size = (uint32_t)__onchip_data_end-(uint32_t)__onchip_data_start;
    printf("Scratch Pad RAM detected! Start: %x, End: %x, Size: %d\n",
        (uint32_t)__onchip_data_start, 
        (uint32_t)__onchip_data_end,
        scratch_size
    );
    #endif

    #ifdef SPRAM    
    if(N*M*sizeof(int8_t) > scratch_size)
        x.data = (int8_t*)scratch_malloc(N*K*sizeof(int8_t), 8);
    else
        x.data = (int8_t*)malloc(N*K*sizeof(int8_t));
    #else
        x.data = (int8_t*)malloc(N*K*sizeof(int8_t));
    #endif
    w.data = (int8_t*)malloc(K*M*sizeof(int8_t));

    x.shape[0] = N;
    x.shape[1] = K;
    
    w.shape[0] = M;
    w.shape[1] = K;

    for (size_t i = 0; i < N*K; i++) {
        x.data[i] = (i % 8);
    }
    for (size_t i = 0; i < K*M; i++) {
        w.data[i] = (i + 7 % 8);
    }
    #ifdef REPEAT
    while(1){
    #endif 
        int32_t *o = (int32_t*)malloc(N*M*sizeof(int32_t));
        long start_time, end_time;
        printf("MiCo 8x8 MatMul Test\n");
        start_time = MiCo_time();
        MiCo_Q8_MatMul(o, &x, &w);
        end_time = MiCo_time();
        printf("MiCo 8x8 Time: %ld\n", end_time - start_time);
        free(o);
    #ifdef REPEAT
        delay(1000); // 1 second
    }
    #endif

    return 0;
}