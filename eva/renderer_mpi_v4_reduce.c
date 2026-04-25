/*
 * renderer_mpi.c
 * MPI-parallel renderer: root reads records and scatters them to ranks;
 * each rank renders its chunk into an RGB buffer (alpha removed, colors opaque);
 * root receives per-rank RGB buffers in rank order and composites by overwriting
 * to reproduce serial ordering. Per-record format is: float32 x,y,radius + uint8 r,g,b.
 * Usage: mpirun -n <procs> ./renderer_mpi <input.bin> <output.png>
 */
#define _POSIX_C_SOURCE 200112L

#include <mpi.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define RECSZ (sizeof(float)*3 + 3)

static uint64_t *aligned_calloc64(size_t bytes){
    void *ptr = NULL;
    if(posix_memalign(&ptr, 64, bytes) != 0) return NULL;
    memset(ptr, 0, bytes);
    return (uint64_t *)ptr;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (argc < 3) {
        if (rank == 0) fprintf(stderr, "Usage: %s <input.bin> <output.png>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }
    const char *inpath = argv[1];
    const char *outpath = argv[2];

    uint32_t version = 0;
    uint64_t count = 0;
    float bbox[6] = {0};
    int W = 640, H = 480;
    /* Root I/O/scatter and composite timing (seconds, via MPI_Wtime) */
    double io_start = 0.0, io_end = 0.0, io_elapsed = 0.0;
    double comp_start = 0.0, comp_end = 0.0, comp_elapsed = 0.0;

    unsigned char *all_records = NULL;

    if (rank == 0) {
        FILE *f = fopen(inpath, "rb");
        io_start = MPI_Wtime();
        if (!f) { perror("fopen"); MPI_Abort(MPI_COMM_WORLD, 1); }
        char magic[5] = {0};
        if (fread(magic, 1, 4, f) != 4) { fprintf(stderr, "failed read magic\n"); fclose(f); MPI_Abort(MPI_COMM_WORLD, 1); }
        if (strncmp(magic, "CRDR", 4) != 0) { fprintf(stderr, "bad magic: %.4s\n", magic); fclose(f); MPI_Abort(MPI_COMM_WORLD, 1); }
        if (fread(&version, sizeof(version), 1, f) != 1) { fprintf(stderr, "failed read version\n"); fclose(f); MPI_Abort(MPI_COMM_WORLD, 1); }
        if (fread(&count, sizeof(count), 1, f) != 1) { fprintf(stderr, "failed read count\n"); fclose(f); MPI_Abort(MPI_COMM_WORLD, 1); }
        if (fread(bbox, sizeof(float), 6, f) != 6) { fprintf(stderr, "failed read bbox\n"); fclose(f); MPI_Abort(MPI_COMM_WORLD, 1); }

        W = (int)roundf(bbox[3] - bbox[0]);
        H = (int)roundf(bbox[4] - bbox[1]);
        if (W <= 0) W = 640;
        if (H <= 0) H = 480;

        fprintf(stderr, "rank0: magic=CRDR version=%u count=%llu image %dx%d\n", version, (unsigned long long)count, W, H);

        size_t totsz = (size_t)count * RECSZ;
        all_records = malloc(totsz);
        if (!all_records) { perror("malloc all_records"); fclose(f); MPI_Abort(MPI_COMM_WORLD, 1); }
        if (fread(all_records, 1, totsz, f) != totsz) { fprintf(stderr, "failed read records\n"); free(all_records); fclose(f); MPI_Abort(MPI_COMM_WORLD, 1); }
        fclose(f);
    }

    // Broadcast header info
    MPI_Bcast(&version, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&count, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(bbox, 6, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&W, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&H, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // compute per-rank counts and displacements (in bytes)
    int *sendcounts = malloc(sizeof(int) * nprocs);
    int *displs = malloc(sizeof(int) * nprocs);
    for (int i = 0; i < nprocs; ++i) { sendcounts[i] = 0; displs[i] = 0; }
    uint64_t *rec_counts = malloc(sizeof(uint64_t) * nprocs);

    // Heuristic: balance by per-record "cost" ~ area ~ r^2.
    // Root computes per-record weights and assigns contiguous ranges to reach
    // approximately equal total weight per rank. Then broadcast record counts
    // so all ranks build consistent byte sendcounts/displs.
    if (count == 0) {
        // nothing
        for (int i = 0; i < nprocs; ++i) rec_counts[i] = 0;
    } else if ((uint64_t)nprocs <= 1 || count <= (uint64_t)nprocs) {
        // fallback to simple division when few records
        uint64_t base = count / (uint64_t)nprocs;
        uint64_t rem = count % (uint64_t)nprocs;
        for (int i = 0; i < nprocs; ++i) {
            uint64_t cnt = base + (i < (int)rem ? 1 : 0);
            rec_counts[i] = cnt;
        }
    } else {
        if (rank == 0) {
            double *weights = malloc(sizeof(double) * (size_t)count);
            double totalw = 0.0;
            
            for (uint64_t i = 0; i < count; ++i) {
                float r = 0.0f;
                /* radius is the 3rd float (offset 2 floats) */
                memcpy(&r, all_records + i * RECSZ + sizeof(float)*2, sizeof(float));
                double w = (double)r * (double)r;
                weights[i] = w;
                totalw += w;
            }

            // for (uint64_t i = 0; i < count; ++i) {
            //     float vals[3];
            //     unsigned char *rec = all_records + i * RECSZ;
            //     memcpy(vals, rec, sizeof(vals));
            //     float cx = vals[0], cy = vals[1], r = vals[2];
            //     int xmin, xmax, ymin, ymax;
            //     xmin = (int)floorf(cx - r);
            //     xmax = (int)floorf(cx + r);
            //     ymin = (int)floorf(cy - r);
            //     ymax = (int)floorf(cy + r);
            //     if (xmax < 0 || ymax < 0 || xmin >= W || ymin >= H) {
            //         weights[i] = 0;
            //         totalw += 0;
            //         continue;
            //     }
            //     if (xmin < 0) xmin = 0;
            //     if (ymin < 0) ymin = 0;
            //     if (xmax >= W) xmax = W - 1;
            //     if (ymax >= H) ymax = H - 1;

            //     double w = (double)(xmax - xmin + 1) * (double)(ymax - ymin + 1);
            //     weights[i] = w;
            //     totalw += w;
            // }

            fprintf(stderr, "total weight: %.2f\n", totalw); 
            double target = totalw / (double)nprocs;
            fprintf(stderr, "target per process: %.2f\n", target); 
            // 每個圓形的 weights == r*r, 加總到 totalw 再分配給 nporc 個 processes 工作量
            uint64_t idx = 0;
            for (int p = 0; p < nprocs; ++p) {
                double acc = 0.0;
                uint64_t start = idx;
                while (idx < count && (acc < target || (count - idx) < (uint64_t)(nprocs - p))) {
                    acc += weights[idx];
                    idx++;
                }
                uint64_t cnt = idx - start;
                if (cnt == 0 && idx < count) { cnt = 1; idx++; }
                rec_counts[p] = cnt;
            }
            // ensure all records assigned
            uint64_t assigned = 0;
            for (int p = 0; p < nprocs; ++p) assigned += rec_counts[p];
            if (assigned < count) rec_counts[nprocs-1] += (count - assigned);
            free(weights);
        }
    }

    // Broadcast record counts per rank so all processes compute same sendcounts/displs
    MPI_Bcast(rec_counts, nprocs, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

    // build byte sendcounts/displs on all ranks from rec_counts
    size_t offset = 0;
    for (int i = 0; i < nprocs; ++i) {
        uint64_t cnt = rec_counts[i];
        sendcounts[i] = (int)(cnt * RECSZ);
        displs[i] = (int)offset;
        offset += (size_t)sendcounts[i];
    }
    free(rec_counts);

    int mybytes = sendcounts[rank];
    unsigned char *mybuf = NULL;
    if (mybytes > 0) {
        mybuf = malloc((size_t)mybytes);
        if (!mybuf) { perror("malloc mybuf"); MPI_Abort(MPI_COMM_WORLD, 1); }
    }

    // Scatterv the raw record bytes
    MPI_Scatterv(all_records, sendcounts, displs, MPI_BYTE, mybuf, mybytes, MPI_BYTE, 0, MPI_COMM_WORLD);

    if (rank == 0) free(all_records);
    io_end = MPI_Wtime();
    io_elapsed = io_end - io_start;
    if(rank == 0) fprintf(stderr, "rank0: read+scatter time: %.6f s\n", io_elapsed);
    
    // local image: premultiplied RGB per pixel (alpha removed; treat as opaque)
    size_t npix = (size_t)W * H;
    
    /* measure composite+write time (including receives and compositing) */
    comp_start = MPI_Wtime();

    /* Timing: measure local render time per rank using MPI_Wtime() */
    double local_start = 0.0, local_end = 0.0, local_elapsed = 0.0;


    // process local records (mybuf contains contiguous records)
    local_start = MPI_Wtime();
    
    uint32_t my_start_index = displs[rank];
    uint64_t *local_state = aligned_calloc64(npix * sizeof(uint64_t));
    uint64_t *pixel_state = NULL;

    uint64_t mycount = (uint64_t)mybytes / RECSZ;
    for (uint32_t i = 0; i < mycount; ++i){
        unsigned char *rec = mybuf + i * RECSZ;
        float floats[3];
        memcpy(floats, rec, sizeof(float)*3);
        unsigned char rgb[3];
        memcpy(rgb, rec + sizeof(float)*3, 3);
        float cx = floats[0];
        float cy = floats[1];
        float radius = floats[2];
        int xmin = (int)floorf(cx - radius);
        int xmax = (int)floorf(cx + radius);
        int ymin = (int)floorf(cy - radius);
        int ymax = (int)floorf(cy + radius);
        // if(xmax < 0 || ymax < 0 || xmin >= W || ymin >= H) continue;
        if (xmin < 0) xmin = 0; if (ymin < 0) ymin = 0;
        if (xmax >= W) xmax = W - 1; if (ymax >= H) ymax = H - 1;
        float r2 = radius * radius;
        uint32_t packed_rgb;
        uint32_t z = my_start_index + i + 1u;
        uint64_t state;

        packed_rgb = ((uint32_t)rgb[0] << 16) | ((uint32_t)rgb[1] << 8) | (uint32_t)rgb[2];
        state = ((uint64_t)z << 32) | packed_rgb;

        for (int y = ymin; y <= ymax; ++y) {
            float py = y + 0.5f;
            float dy = py - cy;
            float dy2 = dy*dy;
            if (dy2 > r2) continue;

            pixel_state = local_state + (size_t)y * W;
            float dx = (float)xmin + 0.5f - cx;

            for (int x = xmin; x <= xmax; ++x, dx += 1.0f) {               
                if ((dx*dx + dy2) <= r2) {
                    pixel_state[x] = state;
                }
            }
        }
    }

    local_end = MPI_Wtime();
    local_elapsed = local_end - local_start;
    fprintf(stderr, "rank %d: local render time: %.6f s\n", rank, local_elapsed);

    free(mybuf);

    /* Aggregate per-rank render times at root: min, max, avg */
    double min_time = 0.0, max_time = 0.0, sum_time = 0.0;
    MPI_Reduce(&local_elapsed, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_elapsed, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_elapsed, &sum_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double avg_time = sum_time / (double)nprocs;
        fprintf(stderr, "per-rank render time (s): min=%.6f max=%.6f avg=%.6f\n", min_time, max_time, avg_time);
    }

    uint64_t *final_state = NULL;
    if (rank == 0) {
        final_state = aligned_calloc64(npix * sizeof(uint64_t));
    }

    /* 直接取每個 pixel 的最大 z-index */
    MPI_Reduce(local_state, final_state, (int)npix, MPI_UINT64_T, MPI_MAX, 0, MPI_COMM_WORLD);

    // Root gathers/receives buffers in rank order and composites onto acc_img/acc_alpha
    if (rank == 0) {        
        unsigned char *pixels = (unsigned char *)malloc(npix * 3);
        for(size_t i = 0; i < npix; ++i){
            uint32_t rgb = (uint32_t)(final_state[i] & 0x00FFFFFFu);
            pixels[i*3 + 0] = (unsigned char)((rgb >> 16) & 0xFFu);
            pixels[i*3 + 1] = (unsigned char)((rgb >> 8) & 0xFFu);
            pixels[i*3 + 2] = (unsigned char)(rgb & 0xFFu);
        }
        
        // convert to 8-bit and write PNG
        size_t stride = (size_t)W * 3;
        if (!stbi_write_png(outpath, W, H, 3, pixels, (int)stride)) {
            fprintf(stderr, "stbi_write_png failed\n"); MPI_Abort(MPI_COMM_WORLD, 1);
        }
        comp_end = MPI_Wtime();
        comp_elapsed = comp_end - comp_start;
        fprintf(stderr, "rank0: composite+write time: %.6f s\n", comp_elapsed);
        fprintf(stderr, "Total runtime (comp_end - io_start): %.6f s\n", comp_end-io_start);
        free(pixels);
        free(final_state);
        fprintf(stderr, "rank0: Wrote PNG %s\n", outpath);
    } 

    free(sendcounts); free(displs);
    MPI_Finalize();
    return 0;
}