#include <mpi.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define RECSZ (sizeof(float) * 3 + 3)

#define _POSIX_C_SOURCE 200112L

static void *aligned_calloc64(size_t n, size_t elem_size) {
    void *ptr = NULL;
    size_t bytes = n * elem_size;

    if (posix_memalign(&ptr, 64, bytes) != 0) {
        return NULL;
    }
    memset(ptr, 0, bytes);
    return ptr;
}

static double estimate_record_weight(const unsigned char *rec, int W, int H) {
    float floats[3];
    memcpy(floats, rec, sizeof(floats));
    float cx = floats[0];
    float cy = floats[1];
    float radius = floats[2];
    int xmin = (int)floorf(cx - radius);
    int xmax = (int)floorf(cx + radius);
    int ymin = (int)floorf(cy - radius);
    int ymax = (int)floorf(cy + radius)

    if (xmax < 0 || ymax < 0 || xmin >= W || ymin >= H) {
        return 1.0;
    }

    if (xmin < 0) xmin = 0; if (ymin < 0) ymin = 0;
    if (xmax >= W) xmax = W - 1; if (ymax >= H) ymax = H - 1;

    return (double)(xmax - xmin + 1) * (double)(ymax - ymin + 1);
}

static void build_partitions_by_weight(const unsigned char *records,
                                       uint64_t count,
                                       int nprocs,
                                       int W,
                                       int H,
                                       uint64_t *rec_counts) {
    int p;

    for (p = 0; p < nprocs; ++p) rec_counts[p] = 0;

    if (count == 0) return;

    if (count <= (uint64_t)nprocs) {
        for (p = 0; p < nprocs; ++p) {
            rec_counts[p] = (p < (int)count) ? 1 : 0;
        }
        return;
    }

    {
        double *weights = (double *)malloc((size_t)count * sizeof(double));
        double total_w = 0.0;
        double target;
        uint64_t idx = 0;

        if (!weights) {
            fprintf(stderr, "malloc failed for weights\n");
            exit(2);
        }

        for (uint64_t i = 0; i < count; ++i) {
            weights[i] = estimate_record_weight(records + (size_t)i * RECSZ, W, H);
            total_w += weights[i];
        }

        target = total_w / (double)nprocs;

        for (p = 0; p < nprocs; ++p) {
            if (p == nprocs - 1) {
                rec_counts[p] = count - idx;
                idx = count;
                break;
            }

            uint64_t start = idx;
            uint64_t ranks_after = (uint64_t)(nprocs - p - 1);
            double acc = 0.0;

            while (idx < count - ranks_after) {
                acc += weights[idx];
                ++idx;
                if (acc >= target) {
                    break;
                }
            }

            if (idx == start) {
                ++idx;
            }
            rec_counts[p] = idx - start;
        }

        free(weights);
    }

    {
        uint64_t assigned = 0;
        for (p = 0; p < nprocs; ++p) {
            assigned += rec_counts[p];
        }
        if (assigned < count) {
            rec_counts[nprocs - 1] += (count - assigned);
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

    unsigned char *my_records_buf = NULL;
    uint64_t *rec_counts = NULL;
    int *sendcounts = NULL;
    int *displs = NULL;

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

        build_partitions_by_weight(all_records, count, nprocs, W, H, rec_counts);
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

    size_t disp = 0;
    for (int p = 0; p < nprocs; ++p) {
        uint64_t bytes64 = rec_counts[p] * (uint64_t)RECSZ;
        sendcounts[p] = (int)bytes64;
        displs[p] = (int)disp;
        disp += (size_t)sendcounts[p];
    }

    {
        int mybytes = sendcounts[rank];
        if (mybytes > 0) {
            my_records_buf = (unsigned char *)malloc((size_t)mybytes);
            if (!my_records_buf) {
                fprintf(stderr, "rank %d: malloc my_records_buf failed\n", rank);
                MPI_Abort(MPI_COMM_WORLD, 2);
            }
        }

        MPI_Scatterv(all_records,
                     sendcounts,
                     displs,
                     MPI_BYTE,
                     my_records_buf,
                     mybytes,
                     MPI_BYTE,
                     0,
                     MPI_COMM_WORLD);
    }

    if (rank == 0) {
        t_io_scatter_end = MPI_Wtime();
    }

    if (all_records) {
        free(all_records);
        all_records = NULL;
    }

    {
        size_t npix = (size_t)W * (size_t)H;
        uint64_t *local_state = (uint64_t *)aligned_calloc64(npix, sizeof(uint64_t));
        uint64_t my_records = rec_counts[rank];
        uint64_t my_start_index = 0;
        double t_render_start;
        double t_render_local;
        double render_min = 0.0;
        double render_max = 0.0;
        double render_sum = 0.0;

        for (int p = 0; p < rank; ++p) {
            my_start_index += rec_counts[p];
        }

        if (!local_state) {
            fprintf(stderr, "rank %d: allocate local_state failed\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 2);
        }

        t_render_start = MPI_Wtime();

        for (uint64_t i = 0; i < my_records; ++i) {
            const unsigned char *rec = my_records_buf + (size_t)i * RECSZ;
            float vals[3];
            uint8_t rgb[3];
            float cx;
            float cy;
            float radius;
            float r2;
            int xmin;
            int xmax;
            int ymin;
            int ymax;
            uint32_t packed_rgb;
            uint32_t z = (uint32_t)(my_start_index + i + 1ULL);
            uint64_t packed_state;

            memcpy(vals, rec, sizeof(vals));
            memcpy(rgb, rec + sizeof(float) * 3, sizeof(rgb));

            cx = vals[0];
            cy = vals[1];
            radius = vals[2];
            r2 = radius * radius;

            xmin = (int)floorf(cx - radius);
            xmax = (int)floorf(cx + radius);
            ymin = (int)floorf(cy - radius);
            ymax = (int)floorf(cy + radius);

            if (xmin < 0) xmin = 0;
            if (ymin < 0) ymin = 0;
            if (xmax >= W) xmax = W - 1;
            if (ymax >= H) ymax = H - 1;
            if (xmax < xmin || ymax < ymin) {
                continue;
            }

            packed_rgb = ((uint32_t)rgb[0] << 16) | ((uint32_t)rgb[1] << 8) | (uint32_t)rgb[2];
            packed_state = ((uint64_t)z << 32) | (uint64_t)packed_rgb;

            for (int y = ymin; y <= ymax; ++y) {
                float py = (float)y + 0.5f;
                float dy = py - cy;
                float dy2 = dy * dy;
                float dx;
                uint64_t *row;

                if (dy2 > r2) {
                    continue;
                }

                row = local_state + (size_t)y * (size_t)W;
                dx = ((float)xmin + 0.5f) - cx;

                for (int x = xmin; x <= xmax; ++x, dx += 1.0f) {
                    if (dx * dx + dy2 <= r2) {
                        row[x] = packed_state;
                    }
                }
            }
        }

        t_render_local = MPI_Wtime() - t_render_start;

        fprintf(stderr,
                "rank %d: render_sec=%.6f local_records=%llu\n",
                rank,
                t_render_local,
                (unsigned long long)my_records);

        MPI_Reduce(&t_render_local, &render_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&t_render_local, &render_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&t_render_local, &render_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        {
            int send_elems;
            int *gather_counts = NULL;
            int *gather_displs = NULL;
            uint64_t *gathered = NULL;
            double t_gather_start = MPI_Wtime();

            if (npix > (size_t)INT32_MAX) {
                if (rank == 0) {
                    fprintf(stderr, "image too large for MPI count int: npix=%zu\n", npix);
                }
                MPI_Abort(MPI_COMM_WORLD, 2);
            }
            send_elems = (int)npix;

            if (rank == 0) {
                gather_counts = (int *)malloc((size_t)nprocs * sizeof(int));
                gather_displs = (int *)malloc((size_t)nprocs * sizeof(int));
                if (!gather_counts || !gather_displs) {
                    fprintf(stderr, "rank0: allocate gather vectors failed\n");
                    MPI_Abort(MPI_COMM_WORLD, 2);
                }

                for (int p = 0; p < nprocs; ++p) {
                    size_t disp64 = (size_t)p * npix;
                    if (disp64 > (size_t)INT32_MAX) {
                        fprintf(stderr, "rank0: gather displacement overflow\n");
                        MPI_Abort(MPI_COMM_WORLD, 2);
                    }
                    gather_counts[p] = send_elems;
                    gather_displs[p] = (int)disp64;
                }

                gathered = (uint64_t *)malloc((size_t)nprocs * npix * sizeof(uint64_t));
                if (!gathered) {
                    fprintf(stderr, "rank0: allocate gathered buffer failed\n");
                    MPI_Abort(MPI_COMM_WORLD, 2);
                }
            }

            MPI_Gatherv(local_state,
                        send_elems,
                        MPI_UINT64_T,
                        gathered,
                        gather_counts,
                        gather_displs,
                        MPI_UINT64_T,
                        0,
                        MPI_COMM_WORLD);

            if (rank == 0) {
                uint64_t *final_state = (uint64_t *)aligned_calloc64(npix, sizeof(uint64_t));
                unsigned char *pixels = (unsigned char *)malloc(npix * 3);
                double t_gather_compose_write;
                double total_local;
                double total_max = 0.0;
                int ok;

                if (!final_state || !pixels) {
                    fprintf(stderr, "rank0: allocate final image buffers failed\n");
                    MPI_Abort(MPI_COMM_WORLD, 2);
                }

                for (int p = 0; p < nprocs; ++p) {
                    const uint64_t *src = gathered + (size_t)gather_displs[p];
                    for (size_t i = 0; i < npix; ++i) {
                        if ((uint32_t)(src[i] >> 32) > (uint32_t)(final_state[i] >> 32)) {
                            final_state[i] = src[i];
                        }
                    }
                }

                for (size_t i = 0; i < npix; ++i) {
                    uint32_t rgb = (uint32_t)(final_state[i] & 0x00FFFFFFu);
                    pixels[i * 3 + 0] = (unsigned char)((rgb >> 16) & 0xFFu);
                    pixels[i * 3 + 1] = (unsigned char)((rgb >> 8) & 0xFFu);
                    pixels[i * 3 + 2] = (unsigned char)(rgb & 0xFFu);
                }

                ok = stbi_write_png(outpath, W, H, 3, pixels, W * 3);
                if (!ok) {
                    fprintf(stderr, "rank0: stbi_write_png failed\n");
                    MPI_Abort(MPI_COMM_WORLD, 2);
                }

                t_gather_compose_write = MPI_Wtime() - t_gather_start;

                total_local = MPI_Wtime() - t_total_start;
                MPI_Reduce(&total_local, &total_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

                fprintf(stderr,
                        "rank0: header version=%u count=%llu image=%dx%d\n",
                        version,
                        (unsigned long long)count,
                        W,
                        H);
                fprintf(stderr, "rank0: io_scatter_sec=%.6f\n", t_io_scatter_end - t_io_scatter_start);
                fprintf(stderr,
                        "rank0: render_sec_min=%.6f render_sec_max=%.6f render_sec_avg=%.6f\n",
                        render_min,
                        render_max,
                        render_sum / (double)nprocs);
                fprintf(stderr, "rank0: gather_compose_write_sec=%.6f\n", t_gather_compose_write);
                fprintf(stderr, "rank0: total_sec=%.6f\n", total_max);
                fprintf(stderr, "rank0: wrote %s\n", outpath);

                free(pixels);
                free(final_state);
                free(gathered);
                free(gather_counts);
                free(gather_displs);
            } else {
                double total_local = MPI_Wtime() - t_total_start;
                MPI_Reduce(&total_local, NULL, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            }
        }

        free(local_state);
    }

    if (my_records_buf) {
        free(my_records_buf);
    }
    free(rec_counts);
    free(sendcounts);
    free(displs);

    MPI_Finalize();
    return 0;
}