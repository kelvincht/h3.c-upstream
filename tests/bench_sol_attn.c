/* Per-stage timing of one Sol-Attn sparse attention call (the production
 * kernels) on Q/K/V captured from a real generation (H3_ATTN_CAPTURE_DIR).
 * The capture's heads are repeated to 56; for a target longer than the
 * capture, video rows are tiled after the capture's sink rows (the key
 * centroid distribution, and so the routed fraction, is preserved).
 *
 *   ./h3_sol_attn_bench [--reps N] [--sink S] PREFIX ROWS...
 *
 * The output hash is printed so kernel changes can be checked for byte
 * identity. */
#include "h3_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { HEADS = 56, D = 128, BLOCK = 64, BK = 32, MAX_REPS = 32 };

static void die(const char *message, const h3_gpu *gpu) {
    fprintf(stderr, "h3_sol_attn_bench: %s%s%s\n", message, gpu ? ": " : "",
            gpu ? h3_gpu_error(gpu) : "");
    exit(1);
}

static double seconds(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return (double)value.tv_sec + (double)value.tv_nsec * 1e-9;
}

static double json_number(const char *text, const char *key) {
    char pattern[64];
    snprintf(pattern, sizeof(pattern), "\"%s\":", key);
    const char *at = strstr(text, pattern);
    if (!at) die("capture header lacks a field", NULL);
    return strtod(at + strlen(pattern), NULL);
}

static int cmp_double(const void *left, const void *right) {
    double a = *(const double *)left, b = *(const double *)right;
    return (a > b) - (a < b);
}

static double median(double *values, int count) {
    qsort(values, (size_t)count, sizeof(*values), cmp_double);
    return values[count / 2];
}

static double run(h3_gpu *gpu, unsigned stages, h3_gpu_tensor *out,
                  h3_gpu_tensor *q, h3_gpu_tensor *k, h3_gpu_tensor *v,
                  uint32_t rows, uint32_t sink) {
    h3_gpu_sparse_attention_set_stages(stages);
    double start = seconds();
    if (!h3_gpu_begin(gpu)) die("begin", gpu);
    h3_gpu_set_sdpa_inputs_head_major(gpu, 1);
    if (!h3_gpu_attention_sparse_bf16(gpu, out, q, k, v, rows, HEADS, D,
                                      1.0f / sqrtf((float)D), 1, sink, 1.3f))
        die("sparse attention", gpu);
    if (!h3_gpu_submit(gpu)) die("submit", gpu);
    return seconds() - start;
}

int main(int argc, char **argv) {
    int reps = 5;
    uint32_t sink = 3179;
    int first = 1;
    while (first < argc && argv[first][0] == '-') {
        if (!strcmp(argv[first], "--reps") && first + 1 < argc)
            reps = atoi(argv[++first]);
        else if (!strcmp(argv[first], "--sink") && first + 1 < argc)
            sink = (uint32_t)strtoul(argv[++first], NULL, 10);
        else die("usage: [--reps N] [--sink S] PREFIX ROWS...", NULL);
        first++;
    }
    if (argc - first < 2 || reps < 1 || reps > MAX_REPS)
        die("usage: [--reps N] [--sink S] PREFIX ROWS...", NULL);
    const char *prefix = argv[first];
    char path[4096], header[2048] = {0};
    snprintf(path, sizeof(path), "%s.json", prefix);
    FILE *file = fopen(path, "r");
    if (!file || !fread(header, 1, sizeof(header) - 1, file))
        die("cannot read capture header", NULL);
    fclose(file);
    uint32_t cap_rows = (uint32_t)json_number(header, "rows");
    uint32_t cap_heads = (uint32_t)json_number(header, "heads");
    uint32_t cap_sink = (uint32_t)json_number(header, "sink_rows");
    if (!(int)json_number(header, "head_major")) die("capture not head-major", NULL);
    size_t cap_count = (size_t)cap_rows * cap_heads * D;
    uint16_t *cap[3];
    const char *suffix[3] = {"q", "k", "v"};
    for (int t = 0; t < 3; t++) {
        snprintf(path, sizeof(path), "%s.%s", prefix, suffix[t]);
        file = fopen(path, "rb");
        cap[t] = malloc(cap_count * sizeof(uint16_t));
        if (!file || !cap[t] ||
            fread(cap[t], sizeof(uint16_t), cap_count, file) != cap_count)
            die("cannot read capture", NULL);
        fclose(file);
    }
    char error[1024];
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) { fprintf(stderr, "h3_sol_attn_bench: %s\n", error); return 1; }
    printf("capture=%s cap_rows=%u cap_heads=%u sink=%u\n", prefix, cap_rows,
           cap_heads, sink);

    for (int arg = first + 1; arg < argc; arg++) {
        uint32_t rows = (uint32_t)strtoul(argv[arg], NULL, 10);
        size_t count = (size_t)rows * HEADS * D;
        uint16_t *host = malloc(count * sizeof(uint16_t));
        h3_gpu_tensor *tensors[3];
        for (int t = 0; t < 3; t++) {
            for (uint32_t h = 0; h < HEADS; h++) {
                const uint16_t *src = cap[t] + (size_t)(h % cap_heads) * cap_rows * D;
                uint16_t *dst = host + (size_t)h * rows * D;
                for (uint32_t r = 0; r < rows; r++) {
                    uint32_t from = r < cap_sink ? r :
                        cap_sink + (r - cap_sink) % (cap_rows - cap_sink);
                    memcpy(dst + (size_t)r * D, src + (size_t)from * D,
                           D * sizeof(uint16_t));
                }
            }
            tensors[t] = h3_gpu_tensor_from_bf16(gpu, host, count);
            if (!tensors[t]) die("upload", gpu);
        }
        h3_gpu_tensor *out = h3_gpu_tensor_new_bf16(gpu, count);
        if (!out) die("allocation", gpu);

        run(gpu, 0xF, out, tensors[0], tensors[1], tensors[2], rows, sink);
        if (!h3_gpu_tensor_read_bf16(out, host, count)) die("read", gpu);
        uint64_t hash = UINT64_C(1469598103934665603);
        const uint8_t *bytes = (const uint8_t *)host;
        for (size_t i = 0; i < count * sizeof(uint16_t); i++)
            hash = (hash ^ bytes[i]) * UINT64_C(1099511628211);

        uint32_t blocks = (rows + BLOCK - 1) / BLOCK;
        uint32_t tile_rows = h3_gpu_sparse_attention_tile_rows();
        uint32_t tiles = (rows + tile_rows - 1) / tile_rows;
        uint32_t *counts = malloc((size_t)HEADS * tiles * sizeof(uint32_t));
        if (!h3_gpu_sparse_attention_read(gpu, rows, HEADS, NULL, NULL, counts))
            die("route readback", gpu);
        double routed = 0, virtual_keys = 0;
        for (size_t i = 0; i < (size_t)HEADS * tiles; i++) {
            routed += counts[i];
            virtual_keys += (double)((blocks - counts[i] + BK - 1) / BK * BK);
        }
        double density = routed / ((double)HEADS * tiles * blocks);
        /* Exact keys: 64 per routed block; virtual keys padded to BK tiles. */
        double exact_flops = 4.0 * tile_rows * routed * (double)BLOCK * (double)D;
        double virtual_flops = 4.0 * tile_rows * virtual_keys * (double)D;
        double dense_flops = 4.0 * (double)rows * rows * (double)HEADS * (double)D;
        /* Bytes per attention threadgroup: Q + O tiles, routed K/V, fp32
         * virtual key means and value sums. */
        double flash_bytes =
            (double)HEADS * tiles * tile_rows * (double)D * 2 * 2 +
            routed * (double)BLOCK * (double)D * 2 * 2 +
            virtual_keys * (double)D * 4 * 2;
        double qkv_bytes = 3.0 * (double)count * 2;

        static const struct { unsigned mask; const char *name; } stages[] = {
            {0xF, "total"}, {0x1, "summaries"}, {0x2, "key_stats"},
            {0x4, "route"}, {0x8, "sparse_flash"},
        };
        double times[5];
        for (int s = 0; s < 5; s++) {
            double samples[MAX_REPS];
            run(gpu, stages[s].mask, out, tensors[0], tensors[1], tensors[2],
                rows, sink);
            for (int rep = 0; rep < reps; rep++)
                samples[rep] = run(gpu, stages[s].mask, out, tensors[0],
                                   tensors[1], tensors[2], rows, sink);
            times[s] = median(samples, reps);
        }
        h3_gpu_sparse_attention_set_stages(0xF);
        printf("rows=%u blocks=%u tiles=%u density=%.4f hash=%016llx\n", rows,
               blocks, tiles, density, (unsigned long long)hash);
        printf("rows=%u dense_flops=%.3e exact_flops=%.3e virtual_flops=%.3e "
               "flash_bytes=%.3e qkv_bytes=%.3e\n", rows, dense_flops,
               exact_flops, virtual_flops, flash_bytes, qkv_bytes);
        for (int s = 0; s < 5; s++) {
            printf("rows=%u stage=%-12s time=%8.3f ms share=%5.1f%%", rows,
                   stages[s].name, times[s] * 1e3, 100.0 * times[s] / times[0]);
            if (stages[s].mask == 0x8)
                printf(" exact=%.1f TFLOPS exact+virtual=%.1f TFLOPS %.0f GB/s",
                       exact_flops / times[s] * 1e-12,
                       (exact_flops + virtual_flops) / times[s] * 1e-12,
                       flash_bytes / times[s] * 1e-9);
            if (stages[s].mask == 0x1)
                printf(" %.0f GB/s", qkv_bytes / times[s] * 1e-9);
            printf("\n");
        }
        fflush(stdout);
        free(counts);
        free(host);
        h3_gpu_tensor_free(out);
        for (int t = 0; t < 3; t++) h3_gpu_tensor_free(tensors[t]);
    }
    h3_gpu_free(gpu);
    for (int t = 0; t < 3; t++) free(cap[t]);
    return 0;
}
