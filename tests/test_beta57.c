#include "h3_host.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int failures;

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        failures++; \
    } \
} while (0)

#define EVALS 8

static void build(const char *scheduler, const char *video, const char *audio,
                  h3_sigma_schedule *schedule) {
    if (scheduler) setenv("H3_SCHEDULER", scheduler, 1); else unsetenv("H3_SCHEDULER");
    setenv("H3_VIDEO_SHIFT", video, 1);
    setenv("H3_AUDIO_SHIFT", audio, 1);
    CHECK(h3_serving_schedule_build(EVALS, schedule));
}

static void check_grid(const float *values, const char *label) {
    for (int i = 0; i < EVALS; i++) {
        if (values[i + 1] > values[i]) {
            fprintf(stderr, "%s not monotonic at %d\n", label, i);
            failures++;
        }
    }
    CHECK(values[EVALS] == 0.0f);
}

int main(void) {
    unsetenv("H3_RENOISE");

    /* Quantile inversion: I_x(0.5, 0.7) at the returned x must reproduce q. */
    const double qs[] = {1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125};
    for (int i = 0; i < 8; i++) {
        if (qs[i] >= 1.0) continue;
        double x = h3_beta_ppf(qs[i], 0.5, 0.7);
        double err = fabs(h3_beta_cdf(x, 0.5, 0.7) - qs[i]);
        printf("q=%.3f ppf=%.9f |I_x-q|=%.2e\n", qs[i], x, err);
        CHECK(err < 1e-6);
    }

    h3_sigma_schedule base, video, audio;
    /* Base grid: shift 1 makes sigma equal the base value. */
    build("beta57", "1", "1", &base);
    printf("beta57 base N=%d:", EVALS);
    for (int i = 0; i <= EVALS; i++) printf(" %.3f", base.video[i]);
    printf("\n");
    CHECK(base.video[0] == 1.0f);
    check_grid(base.video, "beta57 base");

    build("beta57", "12", "3", &video);
    printf("beta57 video shift 12:");
    for (int i = 0; i <= EVALS; i++) printf(" %.4f", video.video[i]);
    printf("\nbeta57 audio shift 3:");
    for (int i = 0; i <= EVALS; i++) printf(" %.4f", video.audio[i]);
    printf("\n");
    CHECK(video.video[0] == 1.0f && video.audio[0] == 1.0f);
    check_grid(video.video, "beta57 video");
    check_grid(video.audio, "beta57 audio");

    /* Unset (or any other value) keeps the linear grid. */
    build(NULL, "12", "3", &audio);
    printf("linear video shift 12:");
    for (int i = 0; i <= EVALS; i++) printf(" %.4f", audio.video[i]);
    printf("\n");
    CHECK(fabs(audio.video[1] - 12.0 * (7.0 / 8.0) /
               (1.0 + 11.0 * 7.0 / 8.0)) < 1e-6);
    check_grid(audio.video, "linear video");

    unsetenv("H3_SCHEDULER");
    unsetenv("H3_VIDEO_SHIFT");
    unsetenv("H3_AUDIO_SHIFT");

    if (failures) {
        fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    printf("beta57 tests passed\n");
    return 0;
}
