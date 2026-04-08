#ifdef RISCV_VEXII
#include "sim_stdlib.h"
#else
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#endif

#include "profile.h"

#define DMA_TEST_BYTES   (32 * 1024)
#define DMA_WORDS        (DMA_TEST_BYTES / sizeof(uint32_t))
#define DMA_TIMEOUT      (20000000u)

#ifndef DMA_BENCH_WITH_SPARSE
#define DMA_BENCH_WITH_SPARSE 0
#endif

#define SPARSE_SRC_ADDR      0x82010000u
#define SPARSE_DST_CPU_ADDR  0x82020000u
#define SPARSE_DST_DMA_ADDR  0x82030000u

static uint32_t src[DMA_WORDS] __attribute__((aligned(64)));
static uint32_t dst_cpu[DMA_WORDS] __attribute__((aligned(64)));
static uint32_t dst_dma[DMA_WORDS] __attribute__((aligned(64)));

static inline void fence_rw_rw(void) {
    __asm__ volatile(
        "fence iorw, iorw\n\t"
        "fence.i"
        ::: "memory"
    );
}

static uint32_t xorshift32(uint32_t x) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}

static void init_buffers(void) {
    uint32_t s = 0x12345678u;
    for (uint32_t i = 0; i < DMA_WORDS; ++i) {
        s = xorshift32(s);
        src[i] = s ^ (i * 0x9e3779b9u);
        dst_cpu[i] = 0;
        dst_dma[i] = 0;
    }
}

static uint32_t check_equal(const uint32_t *a, const uint32_t *b) {
    for (uint32_t i = 0; i < DMA_WORDS; ++i) {
        if (a[i] != b[i]) return i + 1;
    }
    return 0;
}

static void fill_pattern(uint32_t *buf, uint32_t seed, uint32_t salt) {
    uint32_t s = seed;
    for (uint32_t i = 0; i < DMA_WORDS; ++i) {
        s = xorshift32(s);
        buf[i] = s ^ (i * salt);
    }
}

static void clear_buffer(uint32_t *buf) {
    for (uint32_t i = 0; i < DMA_WORDS; ++i) {
        buf[i] = 0;
    }
}

static long bench_cpu_copy(const uint32_t *srcPtr, uint32_t *dstPtr) {
    long start = MiCo_time();
    for (uint32_t i = 0; i < DMA_WORDS; ++i) {
        dstPtr[i] = srcPtr[i];
    }
    long end = MiCo_time();
    return end - start;
}

static long bench_dma_copy(const uint32_t *srcPtr, uint32_t *dstPtr, uint32_t *timeoutFlag, const char *tag) {
    dma_reset(DMA);
    dma_config_reader(DMA, (uint32_t)(uintptr_t)srcPtr, DMA_TEST_BYTES, 0);
    dma_config_writer(DMA, (uint32_t)(uintptr_t)dstPtr, DMA_TEST_BYTES, 0);
    fence_rw_rw();

    uint32_t rdBase = dma_reader_getBase(DMA);
    uint32_t wrBase = dma_writer_getBase(DMA);
    uint32_t rdLen = dma_reader_getLength(DMA);
    uint32_t wrLen = dma_writer_getLength(DMA);
    if (rdBase != (uint32_t)(uintptr_t)srcPtr || wrBase != (uint32_t)(uintptr_t)dstPtr ||
        rdLen != DMA_TEST_BYTES || wrLen != DMA_TEST_BYTES) {
        printf("%s DMA config mismatch rdBase=%08x wrBase=%08x rdLen=%u wrLen=%u\n",
               tag, (unsigned)rdBase, (unsigned)wrBase, (unsigned)rdLen, (unsigned)wrLen);
        *timeoutFlag = 1;
        return 0;
    }

    long start = MiCo_time();
    dma_start(DMA);

    uint32_t rdDone = dma_wait_reader_done(DMA, DMA_TIMEOUT);
    uint32_t wrDone = dma_wait_writer_done(DMA, DMA_TIMEOUT);
    long end = MiCo_time();

    uint32_t rdOff = dma_reader_getOffset(DMA);
    uint32_t wrOff = dma_writer_getOffset(DMA);
    uint32_t rdErr = dma_reader_getError(DMA);
    uint32_t wrErr = dma_writer_getError(DMA);
    uint32_t rdDoneNow = dma_reader_getDone(DMA);
    uint32_t wrDoneNow = dma_writer_getDone(DMA);

    dma_stop(DMA);
    fence_rw_rw();

    *timeoutFlag = (!rdDone || !wrDone);
    if (*timeoutFlag || rdErr || wrErr || !rdDoneNow || !wrDoneNow) {
        printf("%s DMA completion status rdDoneWait=%u wrDoneWait=%u rdDone=%u wrDone=%u rdOff=%u wrOff=%u rdErr=%u wrErr=%u\n",
               tag, (unsigned)rdDone, (unsigned)wrDone,
               (unsigned)rdDoneNow, (unsigned)wrDoneNow,
               (unsigned)rdOff, (unsigned)wrOff,
               (unsigned)rdErr, (unsigned)wrErr);
    }
    return end - start;
}

static void print_bandwidth(const char *name, long cycles) {
    if (cycles <= 0) {
        printf("%s: invalid cycle count %ld\n", name, cycles);
        return;
    }
    unsigned long long bytes = DMA_TEST_BYTES;
    unsigned long long bpc_x1000 = (bytes * 1000ULL) / (unsigned long long)cycles;
    printf("%s: cycles=%ld, %llu.%03llu B/cycle\n",
           name, cycles, bpc_x1000 / 1000ULL, bpc_x1000 % 1000ULL);
}

int main(void) {
    printf("DMA vs CPU memcpy benchmark\n");
    printf("Transfer size: %u bytes\n", (unsigned)DMA_TEST_BYTES);
    printf("src=%p dst_cpu=%p dst_dma=%p dma_reg=0x%08x\n",
           (void*)src, (void*)dst_cpu, (void*)dst_dma, (unsigned)DMA);

    init_buffers();

    long cpuCycles = bench_cpu_copy(src, dst_cpu);

    uint32_t dmaTimeout = 0;
    long dmaCycles = bench_dma_copy(src, dst_dma, &dmaTimeout, "onchip->onchip");

    if (dmaTimeout) {
        printf("DMA benchmark timeout\n");
        return 1;
    }
    if (dma_has_error(DMA)) {
        printf("DMA error flag set (reader=%u writer=%u)\n",
               (unsigned)dma_reader_getError(DMA),
               (unsigned)dma_writer_getError(DMA));
        return 1;
    }

    uint32_t cpuErr = check_equal(src, dst_cpu);
    uint32_t dmaErr = check_equal(src, dst_dma);
    if (cpuErr || dmaErr) {
        printf("Copy mismatch: cpuErrIndex=%u dmaErrIndex=%u\n",
               (unsigned)cpuErr, (unsigned)dmaErr);
        printf("DMA status: rdDone=%u wrDone=%u rdOff=%u wrOff=%u rdErr=%u wrErr=%u\n",
               (unsigned)dma_reader_getDone(DMA),
               (unsigned)dma_writer_getDone(DMA),
               (unsigned)dma_reader_getOffset(DMA),
               (unsigned)dma_writer_getOffset(DMA),
               (unsigned)dma_reader_getError(DMA),
               (unsigned)dma_writer_getError(DMA));
        printf("src[0..3]=%08x %08x %08x %08x\n",
               (unsigned)src[0], (unsigned)src[1], (unsigned)src[2], (unsigned)src[3]);
        printf("dma[0..3]=%08x %08x %08x %08x\n",
               (unsigned)dst_dma[0], (unsigned)dst_dma[1], (unsigned)dst_dma[2], (unsigned)dst_dma[3]);
        return 1;
    }

    print_bandwidth("CPU memcpy", cpuCycles);
    print_bandwidth("DMA memcpy", dmaCycles);

    if (dmaCycles > 0) {
        unsigned long long speedup_x1000 = ((unsigned long long)cpuCycles * 1000ULL) / (unsigned long long)dmaCycles;
        printf("Speedup DMA/CPU: %llu.%03llux\n",
               speedup_x1000 / 1000ULL, speedup_x1000 % 1000ULL);
    }

    printf("DMA benchmark passed\n");

#if DMA_BENCH_WITH_SPARSE
    volatile uint32_t *sparseSrcV = (volatile uint32_t *)(uintptr_t)SPARSE_SRC_ADDR;
    volatile uint32_t *sparseDstCpuV = (volatile uint32_t *)(uintptr_t)SPARSE_DST_CPU_ADDR;
    volatile uint32_t *sparseDstDmaV = (volatile uint32_t *)(uintptr_t)SPARSE_DST_DMA_ADDR;
    uint32_t *sparseSrc = (uint32_t *)sparseSrcV;
    uint32_t *sparseDstCpu = (uint32_t *)sparseDstCpuV;
    uint32_t *sparseDstDma = (uint32_t *)sparseDstDmaV;

    printf("Sparse memory benchmark enabled\n");
    printf("sparseSrc=%p sparseDstCpu=%p sparseDstDma=%p\n",
           (void *)sparseSrc, (void *)sparseDstCpu, (void *)sparseDstDma);

    fill_pattern(sparseSrc, 0x89abcdefu, 0x7f4a7c15u);
    clear_buffer(dst_cpu);
    clear_buffer(dst_dma);
    fence_rw_rw();

    long sparseToOnchipCpu = bench_cpu_copy(sparseSrc, dst_cpu);
    dmaTimeout = 0;
    long sparseToOnchipDma = bench_dma_copy(sparseSrc, dst_dma, &dmaTimeout, "sparse->onchip");
    if (dmaTimeout || dma_has_error(DMA)) {
        printf("sparse->onchip DMA failed timeout=%u rdErr=%u wrErr=%u\n",
               (unsigned)dmaTimeout,
               (unsigned)dma_reader_getError(DMA),
               (unsigned)dma_writer_getError(DMA));
        return 1;
    }
    uint32_t sparseCpuErr = check_equal(sparseSrc, dst_cpu);
    uint32_t sparseDmaErr = check_equal(sparseSrc, dst_dma);
    if (sparseCpuErr || sparseDmaErr) {
        printf("sparse->onchip mismatch cpuErrIndex=%u dmaErrIndex=%u\n",
               (unsigned)sparseCpuErr, (unsigned)sparseDmaErr);
        return 1;
    }

    print_bandwidth("CPU sparse->onchip", sparseToOnchipCpu);
    print_bandwidth("DMA sparse->onchip", sparseToOnchipDma);
    if (sparseToOnchipDma > 0) {
        unsigned long long speedup_x1000 = ((unsigned long long)sparseToOnchipCpu * 1000ULL) /
                                           (unsigned long long)sparseToOnchipDma;
        printf("Speedup sparse->onchip DMA/CPU: %llu.%03llux\n",
               speedup_x1000 / 1000ULL, speedup_x1000 % 1000ULL);
    }

    clear_buffer(sparseDstCpu);
    clear_buffer(sparseDstDma);
    fence_rw_rw();

    long onchipToSparseCpu = bench_cpu_copy(src, sparseDstCpu);
    dmaTimeout = 0;
    long onchipToSparseDma = bench_dma_copy(src, sparseDstDma, &dmaTimeout, "onchip->sparse");
    if (dmaTimeout || dma_has_error(DMA)) {
        printf("onchip->sparse DMA failed timeout=%u rdErr=%u wrErr=%u\n",
               (unsigned)dmaTimeout,
               (unsigned)dma_reader_getError(DMA),
               (unsigned)dma_writer_getError(DMA));
        return 1;
    }
    uint32_t onchipSparseCpuErr = check_equal(src, sparseDstCpu);
    uint32_t onchipSparseDmaErr = check_equal(src, sparseDstDma);
    if (onchipSparseCpuErr || onchipSparseDmaErr) {
        printf("onchip->sparse mismatch cpuErrIndex=%u dmaErrIndex=%u\n",
               (unsigned)onchipSparseCpuErr, (unsigned)onchipSparseDmaErr);
        return 1;
    }

    print_bandwidth("CPU onchip->sparse", onchipToSparseCpu);
    print_bandwidth("DMA onchip->sparse", onchipToSparseDma);
    if (onchipToSparseDma > 0) {
        unsigned long long speedup_x1000 = ((unsigned long long)onchipToSparseCpu * 1000ULL) /
                                           (unsigned long long)onchipToSparseDma;
        printf("Speedup onchip->sparse DMA/CPU: %llu.%03llux\n",
               speedup_x1000 / 1000ULL, speedup_x1000 % 1000ULL);
    }

    printf("Sparse memory benchmark passed\n");
#endif

    return 0;
}
