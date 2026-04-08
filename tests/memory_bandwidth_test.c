#ifdef RISCV_VEXII
#include "sim_stdlib.h"
#else
#include <stdio.h>
#include <stdlib.h>
#endif

#include <stdint.h>

#include "profile.h"

#ifndef TEST_ARRAY_BYTES
#define TEST_ARRAY_BYTES (256 * 1024)
#endif

#define WORD_BYTES (sizeof(uint32_t))
#define ELEM_COUNT (TEST_ARRAY_BYTES / WORD_BYTES)

static uint32_t xorshift32(uint32_t x) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}

static void print_bandwidth(const char* name, long cycles, size_t elem_count) {
    unsigned long long bytes = 2ULL * elem_count * WORD_BYTES; // one read + one write
    if (cycles <= 0) {
        printf("%s: invalid cycle count %ld\n", name, cycles);
        return;
    }
    unsigned long long bpc_x1000 = (bytes * 1000ULL) / (unsigned long long)cycles;
    printf(
        "%s: cycles=%ld, bytes=%llu, bandwidth=%llu.%03llu B/cycle\n",
        name,
        cycles,
        bytes,
        bpc_x1000 / 1000ULL,
        bpc_x1000 % 1000ULL
    );
}

static long bench_inorder_rw(volatile uint32_t* buf, size_t elem_count, uint32_t* checksum) {
    long start_time = MiCo_time();
    uint32_t acc = 0;
    for (size_t i = 0; i < elem_count; ++i) {
        uint32_t v = buf[i];
        v = v + (uint32_t)i + 1U;
        buf[i] = v;
        acc ^= v;
    }
    long end_time = MiCo_time();
    *checksum = acc;
    return end_time - start_time;
}

static long bench_random_rw(volatile uint32_t* buf, size_t elem_count, uint32_t* checksum) {
    long start_time = MiCo_time();
    uint32_t state = 0x12345678U;
    uint32_t acc = 0;
    for (size_t i = 0; i < elem_count; ++i) {
        state = xorshift32(state);
        size_t idx = (size_t)(state % elem_count);
        uint32_t v = buf[idx];
        v = (v ^ state) + 3U;
        buf[idx] = v;
        acc ^= v;
    }
    long end_time = MiCo_time();
    *checksum = acc;
    return end_time - start_time;
}

int main() {
    const size_t elem_count = ELEM_COUNT;
    volatile uint32_t* buffer = (volatile uint32_t*)malloc(elem_count * WORD_BYTES);
    if (!buffer) {
        printf("Failed to allocate %u bytes buffer\n", (unsigned)TEST_ARRAY_BYTES);
        return 1;
    }

    for (size_t i = 0; i < elem_count; ++i) {
        buffer[i] = (uint32_t)i;
    }

    printf("Memory bandwidth benchmark\n");
    printf("Array bytes: %u, elements: %u\n", (unsigned)TEST_ARRAY_BYTES, (unsigned)elem_count);

    uint32_t checksum = 0;

    // Warm up cache/interconnect path.
    (void)bench_inorder_rw(buffer, elem_count, &checksum);

    long inorder_cycles = bench_inorder_rw(buffer, elem_count, &checksum);
    printf("In-order RW checksum: 0x%08x\n", checksum);
    print_bandwidth("In-order RW", inorder_cycles, elem_count);

    long random_cycles = bench_random_rw(buffer, elem_count, &checksum);
    printf("Random RW checksum: 0x%08x\n", checksum);
    print_bandwidth("Random RW", random_cycles, elem_count);

    free((void*)buffer);
    return 0;
}
