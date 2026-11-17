/* Per-op timing of one H3 DiT block's int8 linear layers on the production
 * kernels and buffer layouts, with random int8 weights and activations.
 *
 *   ./h3_int8_linear_bench [--reps N] [--variants 0,1,2,3] [rows ...]
 *
 * Default rows: 25272 (26 x 972). Each H3_INT8_MM_VARIANT is timed per op
 * (QKV, attention output with its head-major quantizer, FC1+SwiGLU, FC2 input
 * quantizer, FC2) and every output buffer is hashed: a variant must be
 * byte-identical to variant 0. Dispatches GPU work: do not run beside a render. */
#include "h3_gpu.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum {
    HIDDEN = 5376, HEADS = 56, HEAD_DIM = 128, INNER = HEADS * HEAD_DIM,
    FFN = 14336, ROPE_HALF = 48, FC2_GROUPS = FFN / 1024, MAX_REPS = 32,
    OPS = 5
};

static const char *op_names[OPS] = {"qkv", "attn_out", "fc1", "fc2_quant",
                                    "fc2"};

static void die(const char *message, const h3_gpu *gpu) {
    fprintf(stderr, "h3_int8_linear_bench: %s%s%s\n", message,
            gpu ? ": " : "", gpu ? h3_gpu_error(gpu) : "");
    exit(1);
}

static double seconds(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return (double)value.tv_sec + (double)value.tv_nsec * 1e-9;
}

static uint64_t next_random(uint64_t *state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return *state * UINT64_C(0x2545f4914f6cdd1d);
}

static float unit_random(uint64_t *state) {
    return (float)(next_random(state) >> 40) * (1.0f / 16777216.0f);
}

static uint16_t to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

static h3_gpu_tensor *random_i8(h3_gpu *gpu, size_t count, uint64_t seed) {
    int8_t *host = malloc(count);
    if (!host) die("out of host memory", NULL);
    uint64_t state = seed * UINT64_C(0x9e3779b97f4a7c15) + 1;
    for (size_t index = 0; index < count; index++)
        host[index] = (int8_t)((int)(next_random(&state) % 255u) - 127);
    h3_gpu_tensor *tensor = h3_gpu_tensor_new_i8(gpu, count);
    if (!tensor || !h3_gpu_tensor_write_i8(tensor, host, count))
        die("int8 tensor", gpu);
    free(host);
    return tensor;
}

static float *random_f32_host(size_t count, uint64_t seed, float low,
                              float high) {
    float *host = malloc(count * sizeof(float));
    if (!host) die("out of host memory", NULL);
    uint64_t state = seed * UINT64_C(0x9e3779b97f4a7c15) + 1;
    for (size_t index = 0; index < count; index++)
        host[index] = low + (high - low) * unit_random(&state);
    return host;
}

static h3_gpu_tensor *random_f32(h3_gpu *gpu, size_t count, uint64_t seed,
                                 float low, float high) {
    float *host = random_f32_host(count, seed, low, high);
    h3_gpu_tensor *tensor = h3_gpu_tensor_from_f32(gpu, host, count);
    if (!tensor) die("f32 tensor", gpu);
    free(host);
    return tensor;
}

static h3_gpu_tensor *random_bf16(h3_gpu *gpu, size_t count, uint64_t seed,
                                  float low, float high) {
    uint16_t *host = malloc(count * sizeof(uint16_t));
    if (!host) die("out of host memory", NULL);
    uint64_t state = seed * UINT64_C(0x9e3779b97f4a7c15) + 1;
    for (size_t index = 0; index < count; index++)
        host[index] = to_bf16(low + (high - low) * unit_random(&state));
    h3_gpu_tensor *tensor = h3_gpu_tensor_from_bf16(gpu, host, count);
    if (!tensor) die("bf16 tensor", gpu);
    free(host);
    return tensor;
}

typedef struct {
    uint32_t rows, padded;
    h3_gpu_tensor *qkv_in, *qkv_in_scales, *qkv_w, *qkv_ws, *q_norm, *k_norm,
        *rope_cos, *rope_sin, *query, *key, *value;
    h3_gpu_tensor *heads, *out_q, *out_qs, *out_w, *out_ws, *attn_out;
    h3_gpu_tensor *mlp_q, *mlp_qs, *fc1_w, *fc1_ws, *fc2_w, *fc2_ws,
        *activated, *mlp_out;
    int8_t *mlp_in_host;
    float *mlp_in_scales_host;
} block_buffers;

static void allocate(h3_gpu *gpu, block_buffers *b, uint32_t rows) {
    uint32_t padded = (rows + 127u) & ~127u;
    size_t projected = (size_t)rows * INNER;
    b->rows = rows;
    b->padded = padded;
    /* Production QKV input arrives prequantized from the fused gate/AdaLN
     * quantizer; per-row scales put activations near unit range. */
    b->qkv_in = random_i8(gpu, (size_t)padded * HIDDEN, 11);
    b->qkv_in_scales = random_f32(gpu, padded, 12, 0.5f / 127, 2.0f / 127);
    b->qkv_w = random_i8(gpu, (size_t)3 * INNER * HIDDEN, 13);
    b->qkv_ws = random_f32(gpu, (size_t)3 * INNER, 14, 0.5e-3f, 2e-3f);
    b->q_norm = random_bf16(gpu, HEAD_DIM, 15, 0.5f, 1.5f);
    b->k_norm = random_bf16(gpu, HEAD_DIM, 16, 0.5f, 1.5f);
    b->rope_cos = random_bf16(gpu, (size_t)rows * ROPE_HALF, 17, -1, 1);
    b->rope_sin = random_bf16(gpu, (size_t)rows * ROPE_HALF, 18, -1, 1);
    b->query = h3_gpu_tensor_new_bf16(gpu, projected);
    b->key = h3_gpu_tensor_new_bf16(gpu, projected);
    b->value = h3_gpu_tensor_new_bf16(gpu, projected);
    b->heads = random_bf16(gpu, projected, 19, -2, 2);
    b->out_q = h3_gpu_tensor_new_i8(gpu, (size_t)padded * INNER);
    b->out_qs = h3_gpu_tensor_new_f32(gpu, padded);
    b->out_w = random_i8(gpu, (size_t)HIDDEN * INNER, 20);
    b->out_ws = random_f32(gpu, HIDDEN, 21, 0.5e-3f, 2e-3f);
    b->attn_out = h3_gpu_tensor_new_bf16(gpu, (size_t)rows * HIDDEN);
    b->mlp_q = h3_gpu_tensor_new_i8(gpu, (size_t)padded * FFN);
    b->mlp_qs = h3_gpu_tensor_new_f32(gpu, (size_t)padded * FC2_GROUPS);
    b->fc1_w = random_i8(gpu, (size_t)2 * FFN * HIDDEN, 22);
    b->fc1_ws = random_f32(gpu, (size_t)2 * FFN, 23, 0.5e-3f, 2e-3f);
    b->fc2_w = random_i8(gpu, (size_t)HIDDEN * FFN, 24);
    b->fc2_ws = random_f32(gpu, HIDDEN, 25, 0.5e-3f, 2e-3f);
    b->activated = h3_gpu_tensor_new_bf16(gpu, (size_t)rows * FFN);
    b->mlp_out = h3_gpu_tensor_new_bf16(gpu, (size_t)rows * HIDDEN);
    if (!b->query || !b->key || !b->value || !b->out_q || !b->out_qs ||
        !b->attn_out || !b->mlp_q || !b->mlp_qs || !b->activated ||
        !b->mlp_out) die("tensor allocation", gpu);
    /* FC2's quantizer overwrites the shared activation buffer, as in the
     * engine, so FC1's int8 input is restored before every checked run. */
    size_t input_bytes = (size_t)padded * HIDDEN;
    b->mlp_in_host = malloc(input_bytes);
    if (!b->mlp_in_host) die("out of host memory", NULL);
    uint64_t state = 26 * UINT64_C(0x9e3779b97f4a7c15) + 1;
    for (size_t index = 0; index < input_bytes; index++)
        b->mlp_in_host[index] =
            (int8_t)((int)(next_random(&state) % 255u) - 127);
    b->mlp_in_scales_host = random_f32_host(padded, 27, 0.5f / 127,
                                            2.0f / 127);
}

static void release(block_buffers *b) {
    h3_gpu_tensor *tensors[] = {
        b->qkv_in, b->qkv_in_scales, b->qkv_w, b->qkv_ws, b->q_norm,
        b->k_norm, b->rope_cos, b->rope_sin, b->query, b->key, b->value,
        b->heads, b->out_q, b->out_qs, b->out_w, b->out_ws, b->attn_out,
        b->mlp_q, b->mlp_qs, b->fc1_w, b->fc1_ws, b->fc2_w, b->fc2_ws,
        b->activated, b->mlp_out};
    for (size_t index = 0; index < sizeof(tensors) / sizeof(*tensors);
         index++) h3_gpu_tensor_free(tensors[index]);
    free(b->mlp_in_host);
    free(b->mlp_in_scales_host);
}

static void restore_mlp_input(h3_gpu *gpu, block_buffers *b) {
    if (!h3_gpu_tensor_write_i8(b->mlp_q, b->mlp_in_host,
                                (size_t)b->padded * HIDDEN) ||
        !h3_gpu_tensor_write_f32(b->mlp_qs, b->mlp_in_scales_host,
                                 b->padded)) die("restore MLP input", gpu);
}

static int encode_op(h3_gpu *gpu, block_buffers *b, int op) {
    if (op == 0)
        return h3_gpu_grouped_qkv_linear_rope_int8(
            gpu, b->query, b->key, b->value, b->qkv_in, b->qkv_in_scales,
            NULL, b->qkv_w, b->qkv_ws, b->q_norm, b->k_norm, b->rope_cos,
            b->rope_sin, b->rows, HIDDEN, HEADS, HEAD_DIM, ROPE_HALF, 1e-5f,
            1, 0, 0, 0);
    if (op == 1)
        return h3_gpu_linear_int8_head_major_bf16(
            gpu, b->attn_out, b->out_q, b->out_qs, b->heads, b->out_w,
            b->out_ws, b->rows, HEADS, HEAD_DIM, HIDDEN);
    static const char *parts[OPS] = {NULL, NULL, "a", "q", "b"};
    if (op >= 2) setenv("H3_BENCH_INT8_MLP_PARTS", parts[op], 1);
    else unsetenv("H3_BENCH_INT8_MLP_PARTS");
    int ok = h3_gpu_mlp_int8_bf16(
        gpu, b->mlp_out, b->activated, b->mlp_q, b->mlp_qs, NULL,
        b->fc1_w, b->fc1_ws, b->fc2_w, b->fc2_ws, NULL, NULL, b->rows,
        HIDDEN, FFN, HIDDEN, 0, 0, 0, 1);
    unsetenv("H3_BENCH_INT8_MLP_PARTS");
    return ok;
}

static void run_full_mlp(h3_gpu *gpu, block_buffers *b) {
    restore_mlp_input(gpu, b);
    if (!h3_gpu_begin(gpu) ||
        !h3_gpu_mlp_int8_bf16(
            gpu, b->mlp_out, b->activated, b->mlp_q, b->mlp_qs, NULL,
            b->fc1_w, b->fc1_ws, b->fc2_w, b->fc2_ws, NULL, NULL, b->rows,
            HIDDEN, FFN, HIDDEN, 0, 0, 0, 1) ||
        !h3_gpu_submit(gpu)) die("full MLP", gpu);
}

/* Returns GPU seconds for one submission holding just this op. */
static double time_op(h3_gpu *gpu, block_buffers *b, int op, double *wall) {
    h3_gpu_stats before, after;
    h3_gpu_get_stats(gpu, &before);
    double start = seconds();
    if (!h3_gpu_begin(gpu) || !encode_op(gpu, b, op) || !h3_gpu_submit(gpu))
        die(op_names[op], gpu);
    *wall = seconds() - start;
    h3_gpu_get_stats(gpu, &after);
    return after.gpu_seconds - before.gpu_seconds;
}

static uint64_t hash_tensor(const h3_gpu_tensor *tensor, size_t bytes) {
    uint8_t *host = malloc(bytes);
    if (!host) die("out of host memory", NULL);
    int ok = h3_gpu_tensor_dtype(tensor) == H3_GPU_I8 ?
        h3_gpu_tensor_read_i8(tensor, (int8_t *)host, bytes) :
        h3_gpu_tensor_read_bf16(tensor, (uint16_t *)host, bytes / 2);
    if (!ok) die("readback", NULL);
    uint64_t hash = UINT64_C(1469598103934665603);
    for (size_t index = 0; index < bytes; index++) {
        hash ^= host[index];
        hash *= UINT64_C(1099511628211);
    }
    free(host);
    return hash;
}

enum { CHECKS = 6 };
static const char *check_names[CHECKS] = {"query", "key", "value",
                                          "attn_out", "activated", "mlp_out"};

static void output_hashes(h3_gpu *gpu, block_buffers *b,
                          uint64_t hashes[CHECKS]) {
    size_t projected = (size_t)b->rows * INNER * 2;
    double unused;
    time_op(gpu, b, 0, &unused);
    time_op(gpu, b, 1, &unused);
    run_full_mlp(gpu, b);
    hashes[0] = hash_tensor(b->query, projected);
    hashes[1] = hash_tensor(b->key, projected);
    hashes[2] = hash_tensor(b->value, projected);
    hashes[3] = hash_tensor(b->attn_out, (size_t)b->rows * HIDDEN * 2);
    hashes[4] = hash_tensor(b->activated, (size_t)b->rows * FFN * 2);
    hashes[5] = hash_tensor(b->mlp_out, (size_t)b->rows * HIDDEN * 2);
}

/* Arithmetic (int8 MACs x 2) and two traffic figures per op: unique bytes
 * (every operand and output once) and tile-stream bytes (operands re-read
 * once per output tile, i.e. the L2/SLC demand with no cross-tile reuse). */
static void op_cost(int op, int variant, double rows, double padded,
                    double *ops, double *unique, double *stream) {
    const double hidden = HIDDEN, inner = INNER, ffn = FFN,
        groups = FC2_GROUPS;
    double m = padded, n, k, tile_n = 128, tile_m = 128;
    switch (op) {
    case 0: n = 3.0 * inner; k = hidden; break;
    case 1: n = hidden; k = inner; break;
    case 2: n = 2.0 * ffn; k = hidden; break;
    case 3:
        *ops = 0;
        *unique = rows * ffn * 2 + padded * ffn + padded * groups * 4;
        *stream = *unique;
        return;
    default: n = hidden; k = ffn; tile_n = variant >= 2 ? 128 : 64; break;
    }
    *ops = 2.0 * rows * n * k;
    double out_bytes = op == 2 ? rows * ffn * 2 : rows * n * 2;
    *unique = m * k + n * k + out_bytes;
    /* FC1 tiles 128 gate + 128 up columns and streams its A tile twice. */
    double a_reads = op == 2 ? 2.0 : 1.0;
    double tiles_n = op == 2 ? n / 256 : n / tile_n;
    *stream = (m / tile_m) * tiles_n *
              (a_reads * tile_m + (op == 2 ? 256 : tile_n)) * k + out_bytes;
    if (op == 1) {  /* head-major BF16 read + int8 write by the quantizer */
        *unique += rows * k * 2 + m * k;
        *stream += rows * k * 2 + m * k;
    }
}

static int cmp_double(const void *left, const void *right) {
    double a = *(const double *)left, b = *(const double *)right;
    return (a > b) - (a < b);
}

int main(int argc, char **argv) {
    int reps = 3, variants[4] = {0, 1, 2, 3}, variant_count = 4;
    uint32_t row_list[8];
    int row_count = 0;
    for (int index = 1; index < argc; index++) {
        if (!strcmp(argv[index], "--reps") && index + 1 < argc) {
            reps = atoi(argv[++index]);
        } else if (!strcmp(argv[index], "--variants") && index + 1 < argc) {
            variant_count = 0;
            for (char *token = strtok(argv[++index], ","); token &&
                 variant_count < 4; token = strtok(NULL, ","))
                variants[variant_count++] = atoi(token);
        } else if (row_count < 8) {
            row_list[row_count++] = (uint32_t)strtoul(argv[index], NULL, 10);
        }
    }
    if (reps < 1 || reps > MAX_REPS) die("--reps must be 1..32", NULL);
    if (!row_count) row_list[row_count++] = 25272;
    char error[1024];
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) { fprintf(stderr, "h3_int8_linear_bench: %s\n", error); return 1; }
    if (!h3_gpu_has_int8_mlp(gpu)) die("int8 TensorOps unavailable", NULL);

    for (int case_index = 0; case_index < row_count; case_index++) {
        uint32_t rows = row_list[case_index];
        if (rows <= 2048) die("rows must exceed 2048 (large-shape path)", NULL);
        block_buffers b;
        double start = seconds();
        allocate(gpu, &b, rows);
        printf("rows=%u padded=%u setup=%.1fs reps=%d\n", rows, b.padded,
               seconds() - start, reps);
        printf("%-7s %-10s %9s %9s %8s %9s %9s\n", "variant", "op", "gpu_ms",
               "wall_ms", "TOPS", "uniqGB/s", "tileGB/s");
        uint64_t reference[CHECKS] = {0};
        for (int v = 0; v < variant_count; v++) {
            char value[8];
            snprintf(value, sizeof(value), "%d", variants[v]);
            setenv("H3_INT8_MM_VARIANT", value, 1);
            uint64_t hashes[CHECKS];
            output_hashes(gpu, &b, hashes);  /* also warms every pipeline */
            double total_gpu = 0, total_ops = 0;
            for (int op = 0; op < OPS; op++) {
                double gpu_times[MAX_REPS], wall_times[MAX_REPS];
                for (int rep = 0; rep < reps; rep++)
                    gpu_times[rep] = time_op(gpu, &b, op, &wall_times[rep]);
                qsort(gpu_times, (size_t)reps, sizeof(double), cmp_double);
                qsort(wall_times, (size_t)reps, sizeof(double), cmp_double);
                double gpu_s = gpu_times[reps / 2];
                double wall_s = wall_times[reps / 2];
                double ops, unique, stream;
                op_cost(op, variants[v], rows, b.padded, &ops, &unique,
                        &stream);
                total_gpu += gpu_s;
                total_ops += ops;
                printf("%-7d %-10s %9.2f %9.2f %8.1f %9.0f %9.0f\n",
                       variants[v], op_names[op], gpu_s * 1e3, wall_s * 1e3,
                       ops / gpu_s * 1e-12, unique / gpu_s * 1e-9,
                       stream / gpu_s * 1e-9);
            }
            printf("%-7d %-10s %9.2f %9s %8.1f\n", variants[v], "block",
                   total_gpu * 1e3, "", total_ops / total_gpu * 1e-12);
            int identical = 1;
            for (int check = 0; check < CHECKS; check++) {
                if (v == 0) reference[check] = hashes[check];
                else if (hashes[check] != reference[check]) {
                    identical = 0;
                    printf("variant %d: %s DIFFERS from variant %d\n",
                           variants[v], check_names[check], variants[0]);
                }
            }
            if (v > 0)
                printf("variant %d: outputs %s variant %d\n", variants[v],
                       identical ? "byte-identical to" : "DIFFER from",
                       variants[0]);
            fflush(stdout);
        }
        unsetenv("H3_INT8_MM_VARIANT");
        release(&b);
    }
    h3_gpu_free(gpu);
    return 0;
}
