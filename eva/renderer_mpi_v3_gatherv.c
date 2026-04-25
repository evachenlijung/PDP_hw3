/*
 * renderer_mpi.c
 * MPI-parallel renderer: root reads records and scatters them to ranks;
 * each rank renders its chunk into an RGB buffer (alpha removed, colors opaque);
 * root receives per-rank RGB buffers in rank order and composites by overwriting
 * to reproduce serial ordering. Per-record format is: float32 x,y,radius + uint8 r,g,b.
 * Usage: mpirun -n <procs> ./renderer_mpi <input.bin> <output.png>
 */

#include <mpi.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define RECSZ (sizeof(float)*3 + 3)
#define ALLOC_MEM_THRESHOLD (400 * 1024)

// =====================================================================
// 1. 定義 Z-Buffer 像素結構體 (16 Bytes per pixel)
// =====================================================================
typedef struct {
    int rank_id;    // 記錄哪個 Rank 畫了這個像素 (-1 代表完全透明)
    float r, g, b;  // 像素顏色
} PixelData;

// =====================================================================
// 2. 自定義 MPI 疊加操作 (具備完美的交換律 Commutative)
// =====================================================================
void zbuffer_composite_op(void *invec, void *inoutvec, int *len, MPI_Datatype *datatype) {
    PixelData *in = (PixelData *)invec;
    PixelData *inout = (PixelData *)inoutvec;
    
    // 🌟 因為我們等一下會註冊自定義型態，這裡的 *len 會精準地是「像素個數」，不用再除以 sizeof 了！
    int num_pixels = *len; 
    
    for (int i = 0; i < num_pixels; ++i) {
        if (in[i].rank_id > inout[i].rank_id) {
            inout[i].rank_id = in[i].rank_id;
            inout[i].r = in[i].r;
            inout[i].g = in[i].g;
            inout[i].b = in[i].b;
        }
    }
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
        if(totsz >= ALLOC_MEM_THRESHOLD) MPI_Alloc_mem((MPI_Aint)totsz, MPI_INFO_NULL, &all_records);
        else all_records = malloc(totsz);
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
            fprintf(stderr, "total weight: %.2f\n", totalw); 
            double target = totalw / (double)nprocs;
            fprintf(stderr, "target per process: %.2f\n", target); 
            // 每個圓形的 weights == r*r, 加總到 totalw 再分配給 nporc 個 processes 工作量
            uint64_t idx = 0;
            for (int p = 0; p < nprocs; ++p) {
                // accumulator
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
        sendcounts[i] = (int)(rec_counts[i] * RECSZ);
        displs[i] = (int)offset;
        offset += (size_t)sendcounts[i];
    }
    uint64_t mycount = rec_counts[rank];
    size_t bytes_per_rank = mycount * RECSZ;
    free(rec_counts);

    /* OPT1: adaptive allocation for receive buffer */
    unsigned char *mybuf = NULL;
    int use_mpi_alloc_mybuf = (bytes_per_rank >= ALLOC_MEM_THRESHOLD);
    if (mycount > 0) {
        if (use_mpi_alloc_mybuf)
            MPI_Alloc_mem((MPI_Aint)bytes_per_rank, MPI_INFO_NULL, &mybuf);
        else
            mybuf = malloc(bytes_per_rank);
        if (!mybuf) { perror("alloc mybuf"); MPI_Abort(MPI_COMM_WORLD, 1); }
    }

    MPI_Scatterv(all_records, sendcounts, displs, MPI_BYTE,
                 mybuf, (int)bytes_per_rank, MPI_BYTE,
                 0, MPI_COMM_WORLD);

    /* free all_records with matching allocator */
    if (rank == 0) {
        size_t totsz = (size_t)count * RECSZ;
        if (totsz >= ALLOC_MEM_THRESHOLD) MPI_Free_mem(all_records);
        else free(all_records);
    }
    free(sendcounts);
    free(displs);

    io_end = MPI_Wtime();
    io_elapsed = io_end - io_start;
    if (rank == 0) fprintf(stderr, "rank0: read+scatter time: %.6f s\n", io_elapsed);

    // local image: premultiplied RGB per pixel (alpha removed; treat as opaque)
    size_t npix = (size_t)W * H;
    size_t img_sz = npix * 3 * sizeof(float);

    // float *img = NULL;
    // int use_mpi_alloc_img = (img_sz >= ALLOC_MEM_THRESHOLD);
    // if (use_mpi_alloc_img)
    //     MPI_Alloc_mem((MPI_Aint)img_sz, MPI_INFO_NULL, &img);
    // else
    //     img = malloc(img_sz);
    // if (!img) { perror("alloc img"); MPI_Abort(MPI_COMM_WORLD, 1); }
    // memset(img, 0, img_sz);

    PixelData *img = NULL;
    int use_mpi_alloc_img = (img_sz >= ALLOC_MEM_THRESHOLD);
    if (use_mpi_alloc_img)
        MPI_Alloc_mem((MPI_Aint)img_sz, MPI_INFO_NULL, &img);
    else
        img = malloc(img_sz);
    if (!img) { perror("alloc img"); MPI_Abort(MPI_COMM_WORLD, 1); }
    
    // 🌟 極度關鍵：初始化所有的像素為「未觸碰 (-1)」狀態
    for (size_t p = 0; p < npix; ++p) {
        img[p].rank_id = -1;
        img[p].r = 0.0f;
        img[p].g = 0.0f;
        img[p].b = 0.0f;
    }

    /* measure composite+write time (including receives and compositing) */
    comp_start = MPI_Wtime();

    /* Timing: measure local render time per rank using MPI_Wtime() */
    double local_start = 0.0, local_end = 0.0, local_elapsed = 0.0;

    // process local records (mybuf contains contiguous records)
    local_start = MPI_Wtime();

    for (uint64_t i = 0; i < mycount; ++i) {
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
        if (xmin < 0) xmin = 0; if (ymin < 0) ymin = 0;
        if (xmax >= W) xmax = W - 1; if (ymax >= H) ymax = H - 1;
        float Cr = rgb[0] / 255.0f;
        float Cg = rgb[1] / 255.0f;
        float Cb = rgb[2] / 255.0f;
        float Ca = 1.0f; /* opaque */
        float src_r = Ca * Cr;
        float src_g = Ca * Cg;
        float src_b = Ca * Cb;
        float r2 = radius * radius;
        for (int y = ymin; y <= ymax; ++y) {
            float dy  = (y + 0.5f) - cy;
            float dy2 = dy * dy;
            #pragma omp simd
            for (int x = xmin; x <= xmax; ++x) {
                float dx = (x + 0.5f) - cx;
                if ((dx*dx + dy2) <= r2) {
                    size_t p = (size_t)y * W + x;
                    // size_t idx = p * 3;
                    // // premultiplied compositing: out = src + (1-src_alpha) * dst
                    // img[idx + 0] = src_r + (1.0f - Ca) * img[idx + 0];
                    // img[idx + 1] = src_g + (1.0f - Ca) * img[idx + 1];
                    // img[idx + 2] = src_b + (1.0f - Ca) * img[idx + 2];
                    
                    // 🌟 畫上去的同時，宣示這塊像素的主權 (寫入自己的 rank_id)
                    img[p].rank_id = rank;
                    img[p].r = src_r; // 因為圓形是不透明的，直接覆蓋即可
                    img[p].g = src_g;
                    img[p].b = src_b;
                }
            }
        }
    }

    local_end = MPI_Wtime();
    local_elapsed = local_end - local_start;
    fprintf(stderr, "rank %d: local render time: %.6f s\n", rank, local_elapsed);

    if (mybuf) {
        if (use_mpi_alloc_mybuf) MPI_Free_mem(mybuf);
        else free(mybuf);
    }

    /* Aggregate per-rank render times at root: min, max, avg */
    double min_time = 0.0, max_time = 0.0, sum_time = 0.0;
    MPI_Reduce(&local_elapsed, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_elapsed, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_elapsed, &sum_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double avg_time = sum_time / (double)nprocs;
        fprintf(stderr, "per-rank render time (s): min=%.6f max=%.6f avg=%.6f\n", min_time, max_time, avg_time);
    }

    // int send_count = (int)(npix * 3);
    // float *recvbuf = NULL;
    // // int *recvcounts = NULL;
    // // int *recvdispls = NULL;

    // int use_mpi_alloc_recv = 0;

    // if(rank == 0){
    //     // recvbuf = malloc((size_t)nprocs * img_sz * sizeof(float));

    //     size_t recv_sz = img_sz * nprocs;
    //     use_mpi_alloc_recv = (recv_sz >= ALLOC_MEM_THRESHOLD);

    //     if(use_mpi_alloc_recv) MPI_Alloc_mem((MPI_Aint)recv_sz, MPI_INFO_NULL, &recvbuf);
    //     else recvbuf = malloc(recv_sz);

    //     if(!recvbuf){ perror("malloc recvbuf"); MPI_Abort(MPI_COMM_WORLD, 1); }

    //     // recvcounts = malloc(sizeof(int) * nprocs);
    //     // recvdispls = malloc(sizeof(int) * nprocs);
    //     // for(int i=0; i < nprocs; i++){
    //     //     recvcounts[i] = send_count;
    //     //     recvdispls[i] = i * send_count;
    //     // }
    // }

    // // MPI_Gatherv(img, send_count, MPI_FLOAT, recvbuf, recvcounts, recvdispls, MPI_FLOAT, 0, MPI_COMM_WORLD);
    // MPI_Gather(img, send_count, MPI_FLOAT, recvbuf, send_count, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // // Root gathers/receives buffers in rank order and composites onto acc_img/acc_alpha
    // if (rank == 0) {
    //     float *acc_img = calloc(npix * 3, sizeof(float));
    //     if (!acc_img) { perror("alloc acc"); MPI_Abort(MPI_COMM_WORLD, 1); }

    //     // // start with rank 0 local results
    //     // memcpy(acc_img, img, npix * 3 * sizeof(float));

    //     // float *tmp_img = malloc(npix * 3 * sizeof(float));
    //     for (int src = 0; src < nprocs; ++src) {
    //         // receive RGB image
    //         // MPI_Recv(tmp_img, (int)(npix * 3), MPI_FLOAT, src, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    //         // composite tmp over acc (order preserved). Since records are opaque, src overwrites dst where non-zero
    //         float *src_img = recvbuf + (size_t)src * (npix * 3);
    //         for (size_t p = 0; p < npix; ++p) {
    //             size_t idx = p * 3;
    //             // float sr = tmp_img[idx + 0];
    //             // float sg = tmp_img[idx + 1];
    //             // float sb = tmp_img[idx + 2];
    //             float sr = src_img[idx + 0];
    //             float sg = src_img[idx + 1];
    //             float sb = src_img[idx + 2];
    //             // if tmp has any coverage at pixel (non-zero), overwrite
    //             if (sr != 0.0f || sg != 0.0f || sb != 0.0f) {
    //                 acc_img[idx + 0] = sr;
    //                 acc_img[idx + 1] = sg;
    //                 acc_img[idx + 2] = sb;
    //             }
    //         }
    //     }
    //     // free(tmp_img);

    // int send_count = (int)(npix * 3);

    // ⚠️ 注意：我們徹底移除了 recvbuf 的宣告與 malloc！
    // 因為 MPI_Reduce 會直接把結果合併到 acc_img 裡面。

    // 只有 Rank 0 需要實際配置最終結果的 acc_img
    // float *acc_img = NULL;
    // =====================================================================
    // 4. 註冊自定義資料型態與 Operator，並執行 MPI_Reduce
    // =====================================================================
    
    // 🌟 神級修正 1：建立 MPI_PIXEL_TYPE，防止封包切割撕裂結構體
    MPI_Datatype MPI_PIXEL_TYPE;
    MPI_Type_contiguous(sizeof(PixelData), MPI_BYTE, &MPI_PIXEL_TYPE);
    MPI_Type_commit(&MPI_PIXEL_TYPE);

    // 建立交換律 Operator
    MPI_Op my_zbuffer_op;
    MPI_Op_create(&zbuffer_composite_op, 1, &my_zbuffer_op);

    // 🌟 神級修正 2：確保 acc_img 也使用跟 img 一模一樣的分配策略 (門當戶對)
    PixelData *acc_img = NULL;
    if (rank == 0) {
        if (use_mpi_alloc_img) {
            MPI_Alloc_mem((MPI_Aint)img_sz, MPI_INFO_NULL, &acc_img);
        } else {
            acc_img = malloc(img_sz);
        }
        if (!acc_img) { perror("alloc acc"); MPI_Abort(MPI_COMM_WORLD, 1); }
    }

    // 🌟 執行 Reduce：注意 count 變成了 npix，datatype 變成了 MPI_PIXEL_TYPE！
    int send_count = (int)npix; 
    MPI_Reduce(img, acc_img, send_count, MPI_PIXEL_TYPE, my_zbuffer_op, 0, MPI_COMM_WORLD);

    // 用完記得釋放型態與 Operator
    MPI_Op_free(&my_zbuffer_op);
    MPI_Type_free(&MPI_PIXEL_TYPE);

        // convert to 8-bit and write PNG
    if(rank == 0){
        size_t stride = (size_t)W * 3;
        unsigned char *pixels = malloc((size_t)H * stride);
        if (!pixels) { perror("malloc pixels"); MPI_Abort(MPI_COMM_WORLD, 1); }
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                size_t p = (size_t)y * W + x;
                // size_t idx = p * 3;
                // float fr = acc_img[idx + 0];
                // float fg = acc_img[idx + 1];
                // float fb = acc_img[idx + 2];
                
                float fr = acc_img[p].r;
                float fg = acc_img[p].g;
                float fb = acc_img[p].b;

                int ir = (int)roundf(fmaxf(0.0f, fminf(1.0f, fr)) * 255.0f);
                int ig = (int)roundf(fmaxf(0.0f, fminf(1.0f, fg)) * 255.0f);
                int ib = (int)roundf(fmaxf(0.0f, fminf(1.0f, fb)) * 255.0f);
                pixels[(size_t)y * stride + x*3 + 0] = (unsigned char)ir;
                pixels[(size_t)y * stride + x*3 + 1] = (unsigned char)ig;
                pixels[(size_t)y * stride + x*3 + 2] = (unsigned char)ib;
            }
        }
        if (!stbi_write_png(outpath, W, H, 3, pixels, (int)stride)) {
            fprintf(stderr, "stbi_write_png failed\n"); MPI_Abort(MPI_COMM_WORLD, 1);
        }
        comp_end = MPI_Wtime();
        comp_elapsed = comp_end - comp_start;
        fprintf(stderr, "rank0: composite+write time: %.6f s\n", comp_elapsed);
        fprintf(stderr, "Total runtime (comp_end - io_start): %.6f s\n", comp_end-io_start);
        
        free(pixels);
        // free(acc_img);

        // 🌟 神級修正 3：依照當時的分配方式，正確釋放 acc_img
        if (use_mpi_alloc_img) MPI_Free_mem(acc_img);
        else free(acc_img);

        fprintf(stderr, "rank0: Wrote PNG %s\n", outpath);

        // if (use_mpi_alloc_recv) MPI_Free_mem(recvbuf);
        // else free(recvbuf);

        // free(recvcounts);
        // free(recvdispls);
    }
    // else {
    //     // send local img to root in order
    //     MPI_Send(img, (int)(npix * 3), MPI_FLOAT, 0, 100, MPI_COMM_WORLD);
    // }

    // free(img);
    if (use_mpi_alloc_img) MPI_Free_mem(img);
    else free(img);

    MPI_Barrier(MPI_COMM_WORLD);
    
    MPI_Finalize();
    return 0;
}


