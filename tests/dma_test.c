#ifdef RISCV_VEXII
#include "sim_stdlib.h"
#else
#include <stdio.h>
#include <stdint.h>
#endif

/*
 * DMA register-level smoke test for TilelinkDmaFiber.
 *
 * This test validates software-visible behavior:
 * - register programming
 * - zero-length transfer completion
 * - start/stop behavior and status fields
 *
 * It intentionally does not validate data movement because current HW wiring
 * exposes reader/writer streams as toplevel IOs, not looped back in SoC.
 */
int main() {
    const uint32_t timeout = 1000000;

    printf("DMA test start\n");

    dma_reset(DMA);

    // Configure a small aligned transfer window in RAM.
    dma_config_reader(DMA, 0x80000000u, 16u, 0u);
    dma_config_writer(DMA, 0x80000010u, 16u, 0u);

    if(dma_reader_getBase(DMA) != 0x80000000u) {
        printf("DMA reader base mismatch\n");
        return 1;
    }
    if(dma_writer_getBase(DMA) != 0x80000010u) {
        printf("DMA writer base mismatch\n");
        return 1;
    }
    if(dma_reader_getLength(DMA) != 16u || dma_writer_getLength(DMA) != 16u) {
        printf("DMA length mismatch\n");
        return 1;
    }

    // Zero-length must complete immediately when enabled.
    dma_config_reader(DMA, 0x80000000u, 0u, 0u);
    dma_start_reader(DMA);
    if(!dma_wait_reader_done(DMA, timeout)) {
        printf("DMA reader zero-length did not complete\n");
        dma_stop_reader(DMA);
        return 1;
    }
    dma_stop_reader(DMA);

    dma_config_writer(DMA, 0x80000000u, 0u, 0u);
    dma_start_writer(DMA);
    if(!dma_wait_writer_done(DMA, timeout)) {
        printf("DMA writer zero-length did not complete\n");
        dma_stop_writer(DMA);
        return 1;
    }
    dma_stop_writer(DMA);

    // Error should remain clear in this smoke sequence.
    if(dma_has_error(DMA)) {
        printf("DMA error flag unexpectedly set\n");
        return 1;
    }

    printf("DMA test passed\n");
    return 0;
}
