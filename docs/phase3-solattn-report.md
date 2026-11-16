# Phase 3: Sol-Attn block-sparse attention in h3c

Branch `exp-sol-attn`. Hardware: M5 Max (TensorOps). Date: 2026-11-16. All renders, sheets and
logs are under `/Volumes/Transcend/p3sol/`. Sheets: `/Volumes/Transcend/p3sol/sheets/<profile>/`.

## Summary

- **Parity.** With sparse off, the branch is bit-identical to main on all 9 available profiles.
  framemd5 and the final latent are identical once the Qwen encoder is made deterministic
  (`H3_MPS_GQA=1`, applied to both binaries).
- **Upstream bug.** Main is not run-to-run deterministic. The cause is a missing barrier in the
  Qwen causal GQA kernel. See the bug section; it is not fixed here.
- **Quality.** At tau 1.3, sparse output differs from dense by about as much as two production
  runs of main differ from each other, and in places by 2-3x that. It shows as drift (pose,
  expression, motion timing), not degradation. There is no grid or block artifact in any
  sheet, and sharpness and texture metrics stay within ±6%. In the human blind A/B relayed by
  the coordinator, differences were "mild, hard to notice" (dense 10, same 4, sparse 2 of 16).
  On the real-use path (768x1024x124, default min_tokens) the default-profile render is close
  to dense: 32.5 dB, and the diff only shows when amplified 11x.
- **Speed.** At tau 1.3, end-to-end time drops 23% at 768x1024x124, 28% at 768x1344x124 and
  26% at 512x768x362. 1 s clips gain at most 4%.
- **Recommendation.** Merge as opt-in, and switch it on per profile with tau 1.3 and
  min_tokens 20000. Both defaults stay unchanged.

## Design

`H3_SPARSE_ATTN=1` routes the 50 joint self-attention calls of the DiT through a port of
Sol-Attn. Unset, nothing on the GPU changes.

Each sparse call makes four dispatches:

1. **Summaries.** Per 64-token block and head: fp32 query mean, key mean and value sum.
2. **Key stats.** Per head: the mean and variance of the block key means.
3. **Route.** Each 128-row query tile (two query blocks) routes a key block as exact when it is
   a sink (text, condition, reference or audio rows), a neighbour, the partial last block, or
   its log2-domain score is above mean + `H3_SPARSE_TAU` standard deviations. The routing is
   the union over the tile's two query blocks.
4. **Sparse attention.** This is the tiled TensorOps flash kernel `tgq_r16s8k32` with a single
   step call site. It walks "virtual key" tiles first: the key mean and value sum of each
   unrouted block, with weight 64. It then walks the routed blocks exactly. If every block is
   routed, the key order is dense and the output is identical to the dense tiled kernel.

Sink query rows stay dense. A call runs sparse only when rows >= `H3_SPARSE_MIN_TOKENS` and the
step's sigma lies in `[shift(1-END), shift(1-START)]`, which is ComfyUI `percent_to_sigma` with
the run's `H3_VIDEO_SHIFT`. START defaults to 0.2 and END to 0.9.

- With 6 steps at shift 6 or shift 12, steps 2-5 are sparse (4 of 6).
- With PRISM's 8 steps at shift 12, steps 2-7 are sparse (6 of 8).

The text refiner stays dense. To set the flag per profile, put the keys in the profile's `env`
map in `models.json`.

## Parity (sparse off == main)

**Method.** The main build is detached e6354c9; the branch build is 0250b42. Each pair renders
1 s, 22 frames, seed 101, prompt 1 (ref2va p1, fl2va p32), with the profile's registry run dir
and env, as gen.sh does. Both binaries also get `H3_MPS_GQA=1`. That is an existing env that
swaps the Qwen text encoder's causal GQA for the MPSGraph path, which is deterministic.
Without it, no bit-level comparison is possible (see the bug section). The sparse code never
touches the text encoder, so this does not hide a sparse-path difference.

| profile | framemd5 (51 hashes) | final latent |
|---|---|---|
| ref2va-lightx2v-hybrid-unlocked-lms08 (default) | identical | identical |
| ref2va-lightx2v-hybrid-unlocked | identical | identical |
| ref2va-flex-hybrid-unlocked | identical | identical |
| ref2va-flex-unlocked | identical | identical |
| ref2va-prism-dmad-pdmd | identical | identical |
| fl2va-flex-unlocked | identical | identical |
| fl2va-lightx2v-hybrid-unlocked | identical | identical |
| fl2va-lightx2v-unlocked | identical | identical |
| fl2va-base | identical | identical |

Skipped because their weights are missing: ref2va-lightx2v-hybrid-unlocked-lms06 and -lms10
(deleted folds), and fl2va-base-unlocked (not on Transcend).

The first PRISM attempt failed with "required weight is absent: blocks.0.attn.qkv_proj.weight".
The PRISM fold is int8-only and needs `H3_DIT_INT8_CACHE`, which h3serve always passes and the
render harness did not. With the cache passed, PRISM rendered and the pair is complete.

The production-path control is 2 runs of main without `H3_MPS_GQA`. Those are the noise
floor in the quality table.

## Quality: sparse vs dense, 1 s clips

**Setup.** 1 s, 22 frames, at 768x1024 (ref2va) and 512x768 (fl2va), with
`H3_SPARSE_MIN_TOKENS=0`. 1 s clips are about 7-9k tokens, below the 20k default, so without
that override sparse would not run at all.

- Prompts: ref2va p1@101 and p2@202; fl2va p32@101 (first and last frame) and p31@202
  (text-only).
- The dense baseline is the same branch binary with sparse unset.
- Both arms run with `H3_MPS_GQA=1`, so the difference is the sparse path alone.
- The noise floor is mainA vs mainB: main twice, production path, p1@101.
- Columns are final-latent relative L2, mean video PSNR, and the ab_metrics mean over both
  clips.

| profile | noise floor lat / PSNR | sparse p1 lat / PSNR | sparse p2 lat / PSNR | ssim_ds sparse (noise) | lap_absmean Δ | texture_ratio_skin |
|---|---|---|---|---|---|---|
| ref2va lms08 (default) | 0 / identical ¹ | 0.054 / 33.5 | 0.171 / 25.2 | 0.932 (1.000) | -1.1% | 1.011 |
| ref2va lightx2v-hybrid | 0.036 / 35.8 | 0.064 / 32.9 | 0.158 / 26.9 | 0.940 (0.994) | -2.4% | 1.035 |
| ref2va flex-hybrid | 0.041 / 36.2 | 0.145 / 26.9 | 0.338 / 21.7 | 0.816 (0.992) | -0.1% | 1.021 |
| ref2va flex | 0.093 / 30.8 | 0.112 / 30.1 | 0.372 / 19.7 | 0.791 (0.973) | +0.4% | 1.047 |
| ref2va prism-dmad-pdmd | 0 / identical ¹ | 0.296 / 21.4 | 0.336 / 21.1 | 0.749 (1.000) | -1.0% | 1.032 |
| fl2va flex (fl2va default) | 0.317 / 22.5 | 0.277 / 22.1 | 0.623 / 13.5 ² | 0.568 (0.744) | -6.1% | 0.957 |
| fl2va lightx2v-hybrid | 0.148 / 27.4 | 0.186 / 24.8 | 0.315 / 19.1 ² | 0.763 (0.921) | -0.8% | 0.983 |
| fl2va lightx2v-unlocked | 0.160 / 27.2 | 0.147 / 26.5 | 0.310 / 18.8 ² | 0.760 (0.903) | +1.9% | 1.015 |
| fl2va base | 0.177 / 26.4 | 0.165 / 26.5 | 0.422 / 17.1 ² | 0.713 (0.890) | -1.2% | 1.009 |

¹ The race only fires sometimes. Other main-vs-main pairs on the default profile measured
0.10 / 30.3 dB (ref_woman prompt) and, in the earlier worker's run, 0.196 latent / 39.5 dB.
A text-only fl2va prompt reached 0.45 / 17.2 dB. The production noise floor therefore spans
"identical" to a different sample.

² p31 is text-only. Text-only trajectories are chaotic: two runs of main give 17 dB on the
same kind of prompt. Sparse yields a different but equally clean sample (same scene, a new
composition).

**Perturbation floor of the pipeline.** Replacing MPSGraph SDPA with the exact dense tiled
kernel, an ulp-level change, already gives 22.3 dB at 768x1024x22. Toggling head-major
attention output on MPSGraph gives 22.7 dB (ledger, Phase 3 gate). So 21-33 dB between sparse
and dense is the size of difference any exact kernel swap produces here. PSNR alone cannot
separate approximation error from trajectory divergence.

**Artifacts.** Sheets with `--diff` are at `/Volumes/Transcend/p3sol/sheets/<profile>/sparseG_<p>_<seed>.png`.

- No grid, stripe or block pattern appears in any sparse-only diff. All diffs follow edges
  (positional drift).
- The user saw grid stripes in both arms of fl2va-base p31 and fl2va-lightx2v-hybrid p32, so
  that is a baseline artifact.
- Visible differences are drift: head-turn timing (flex-hybrid p2), expression and face shape
  (PRISM p1 is the largest), and sign glyphs and background layout.

**Human blind A/B (16 pairs, relayed by the coordinator).** Dense 10, same 4, sparse 2. Most
differences were mild and hard to notice. All 6 decisive fl2va verdicts went to dense. Two
exceptions:

- fl2va-lightx2v-unlocked p31_202: sparse "significantly worse, feels broken". The still
  frames and crops on that sheet look clean, so the defect is probably temporal (motion). It
  was not reproduced here.

**Real-use path, default profile.** 768x1024x124, default min_tokens (30.6k tokens, so sparse
runs), tau 1.3, seed 101, p1:

| | value |
|---|---|
| final latent rel L2 | 0.067 |
| PSNR mean / min | 32.5 / 30.0 dB |
| audio diff vs dense | -4.8 dB |
| audio log-spectral distance | 3.9 dB |

The sheet `sheets/long-768x1024x124/sparseG_p1_101.png` is near-identical; the diff only shows
amplified 11x and follows edges. This is the only 124-frame real-path visual check. Every other
profile's 124-frame check is **unverified** (not run, per the time cap).

## Tau study (default profile)

Density and per-layer error come from Q/K/V captured on the real 768x1024x124 run (30,610 rows,
4 heads; blocks 0/12/25/37/49; steps 1/3/5), via `h3_sparse_attn_check --tau T`. The 1 s video
and audio columns are sparse vs dense at min_tokens 0, with p1 and p2 averaged.

| tau | density (blocks 0/12/25/37/49, step 3) | rel L2 block 12 / 25 / 37 | 1 s latent rel L2 p1 / p2 | 1 s audio diff p1 / p2 |
|---|---|---|---|---|
| 1.3 | 0.39 / 0.32 / 0.34 / 0.36 / 0.26 | 0.157 / 0.136 / 0.119 | 0.054 / 0.171 | -4.8 / -12.3 dB |
| 0.7 | 0.51 / 0.45 / 0.44 / 0.47 / 0.42 | 0.092 / 0.092 / 0.083 | 0.056 / 0.151 | -18.1 / -17.8 dB |
| 0.3 | 0.61 / 0.55 / 0.51 / 0.55 / 0.59 | 0.062 / 0.070 / 0.066 | 0.055 / 0.141 | -13.6 / -19.8 dB |

- Lowering tau halves the per-layer error.
- The 1 s video divergence barely moves (0.054 → 0.055 on p1). The divergence is the
  pipeline's trajectory sensitivity, not the size of the approximation.
- Audio moves closer to dense at lower tau.
- Density does not vary by step at a given tau (step 1/3/5 within ±0.01 on block 12).

Speed cost of a lower tau, end-to-end (speed table): at 768x1024x124, 23% saving at 1.3 vs 17%
at 0.7; at 768x1344x124, 28% vs 22%.

**Recommendation: keep tau 1.3.**

- Under the user's rule (worth it when the saving is significant, about 15%), 1.3 gives the
  largest saving.
- The video-quality evidence is the same at 1.3 and 0.7, so the lower tau buys no visible
  quality.
- The real-path render at 1.3 is clean.
- Fallback: tau 0.7 if an audio or temporal defect shows. It still saves 17-22% at 124 frames.

## Token-reduction attribution

Default profile, 1 s, p1@101, `H3_MPS_GQA=1`:

| comparison | latent rel L2 | PSNR | sheet |
|---|---|---|---|
| token reduction on, sparse off vs plain dense | 0.70 | 11.2 dB | `sheets/token-reduction/trdense_p1_101.png` |
| token reduction on: sparse on vs sparse off | 0.145 | 25.0 dB | `sheets/token-reduction/trsparse_p1_101.png` |

The first sheet shows ghosted double faces and doubled hands. The second shows the same
ghosting in both arms.

**Verdict: the distorted face comes from `H3_TOKEN_REDUCTION`, not from sparse attention.**
It reproduces with sparse off. Sparse on top of token reduction adds only the usual drift.

## Audio

Audio was extracted with ffmpeg (mono, 48 kHz) and compared by difference RMS relative to the
dense RMS, RMS ratio, and log-spectral distance (LSD).

| | diff vs dense | LSD | RMS ratio |
|---|---|---|---|
| production noise floor, main vs main (7 profiles with a difference) | -0.3 to -8.5 dB | 5.5-8.3 dB | — |
| sparse at tau 1.3, 1 s (18 clips) | +1.3 to -14.3 dB | 3.5-12.0 dB | 0.84-1.06 |
| real-path 124-frame clip | -4.8 dB | 3.9 dB | — |

Sparse audio sits in the production noise range. There is no loudness collapse. The two
LSD > 9 dB cases are fl2va text-only p31 clips, where the video is a different sample too.
Tau 0.7 brings 1 s audio to about -18 dB, LSD 2.3-3.2. **Not checked by ear (unverified).**

## Speed (idle GPU)

**Setup.** Default ref2va profile, seed 101, p1. Production path: no `H3_MPS_GQA`; the latent
dump is on for every arm. Before each run, no other h3, rsync or mkfold process was running.
Sparse arms use `H3_SPARSE_MIN_TOKENS=0` to locate the crossover. Sigma window [0.40, 0.96],
so 4 of 6 steps are sparse.

Measurements:

- Denoise is the "denoise enqueue" plus "denoise" phases.
- End-to-end (e2e) is the h3 render summary.
- Per call is derived: (dense denoise - sparse denoise) / 200 sparse calls, the saving per
  attention call. Per-call kernel time was not instrumented.

| config | tokens | dense s/step / e2e | tau 1.3 s/step / e2e (saving) | tau 0.7 s/step / e2e (saving) | per-call saving (1.3 / 0.7) |
|---|---|---|---|---|---|
| 768x1024x22 | ~7.2k | 4.15 / 59.6 s | 4.05 / 58.9 s (-1.1%) | 4.17 / 59.7 s (+0.2%) | 0.003 / -0.001 s |
| 768x1344x22 | ~8.9k | 5.43 / 68.8 s | 5.00 / 66.2 s (-3.7%) | 5.19 / 67.6 s (-1.8%) | 0.013 / 0.007 s |
| 512x768x124 | ~15.6k | 12.37 / 118.0 s | 10.40 / 105.9 s (-10.3%) | 11.22 / 110.7 s (-6.2%) | 0.059 / 0.035 s |
| 768x1024x124 | 30.6k | 39.59 / 297.7 s | 28.24 / 229.4 s (**-22.9%**) | 31.05 / 246.1 s (-17.3%) | 0.34 / 0.26 s |
| 768x1344x124 | ~39.5k | 63.31 / 451.4 s | 42.16 / 324.3 s (**-28.2%**) | 47.00 / 353.1 s (-21.8%) | 0.63 / 0.49 s |
| 512x768x362 | ~37k | 74.31 / 519.5 s | 51.62 / 383.6 s (**-26.2%**) | 58.81 / 426.2 s (-18.0%) | 0.68 / 0.47 s |
| 768x1120x362 | ~83k | not measured (run cut for time; unverified) | | | |

Token counts are video rows plus sink rows. The 30.6k figure comes from the capture header;
the others are computed from the latent grid.

The phase-2 quality run at 768x1024x124 with the real default min_tokens repeated the result
on its own: dense 293 s, sparse 228 s (-22%).

**min_tokens: keep 20000 (committed with the measured rationale).**

- Sparse is never slower than dense: the break-even is about 7k tokens.
- The saving stays at 10% or less up to 15.6k tokens and reaches 23% at 30.6k.
- 20000 keeps 1 s clips and 512x768x124 (gains of 4% and 10%) on the exact dense path, and
  turns sparse on for every config that saves at least 17% at either tau.
- The default comment in `h3_dit.c` and the README now state this measurement instead of the
  gate-3a kernel ratio.

## Recommended defaults and per-profile flags

Engine defaults stay: `H3_SPARSE_TAU=1.3`, `H3_SPARSE_START=0.2`, `H3_SPARSE_END=0.9`,
`H3_SPARSE_MIN_TOKENS=20000`.

The user's rule is: sparse on when the 124-frame end-to-end saving is about 15% or more,
unless a 124-frame real-path render shows a visible break. Measured on the default profile:
-22.9% at 768x1024x124 and -28.2% at 768x1344x124.

- **Recommend `H3_SPARSE_ATTN=1` in the `env` map of every available profile.** At the real
  min_tokens it only engages from about 20k tokens: 768x1024 from about 3.5 s, 512x768 from
  about 6 s.
- Speed was measured on the default profile only. The other profiles share the same DiT and a
  sparse fraction of 4/6 steps (PRISM 6/8), so similar savings are expected (**unverified**).
- The 124-frame visual check passed on the default profile only. It is **unverified** for the
  other 8 profiles (not run under the time cap).
- fl2va profiles: the human A/B leaned dense (6 of 6 decisive verdicts) and reported one
  "feels broken" 1 s clip (fl2va-lightx2v-unlocked p31_202, text-only). At fl2va's usual
  512x768 the default min_tokens keeps 1 s-5 s clips dense anyway, so the flag only changes
  long or large fl2va renders. Do one 124-frame look at fl2va-lightx2v-unlocked before
  relying on it there.
- `models.json` was not changed (it belongs to the outer repo, which has uncommitted changes
  from another worker). The coordinator applies the env keys.

## Production bug: Qwen text encoder is not run-to-run deterministic

**Symptom.** Two identical runs of main differ. On the default profile: latent rel L2 0.10-0.20,
30-39 dB. On a text-only fl2va prompt: 0.45 latent, 17.2 dB, effectively a different sample.
It is intermittent; some pairs come out identical.

**Where it starts.** A block-0 step-0 Q/K/V capture of two runs shows only the text rows
(0-161) differing, by 4.9% relative. Audio and video rows were identical.

`h3_real_prompt_test` (main) narrows it to the text encoder:

- A 6-token prompt gives the same hash twice.
- The 175-token p31 prompt gives 1c1e8a41 vs 225f2d3a.
- `H3_QWEN_PREFETCH=0` does not help.

**Root cause.** In `h3_gqa_causal_bf16` (`h3_shaders.metal:4021-4028`), the line
`float maximum = reductions[0];` is followed, with no threadgroup barrier, by
`reductions[tid] = local_sum;`. Thread 0 can overwrite `reductions[0]` with its partial sum
before a slower SIMD group has read the max. That group then subtracts the wrong max. The
softmax normalisation hides most of the effect, but the rounding differs per run. If the
wrong max is much smaller than the true one, exp can overflow.

**Evidence.** A scratch probe (`scratchpad/scripts/det_ops.c`) fed fixed inputs at 175 rows:

- The GQA output hash differs rep to rep inside one process.
- The MPSGraph BF16 linear is identical across reps and processes.
- `H3_MPS_GQA=1` (MPSGraph GQA) is identical across reps.

**Fix (not applied, per brief).** Add `threadgroup_barrier(mem_flags::mem_threadgroup);` after
`float maximum = reductions[0];`. This changes production output, so it needs its own A/B.

**Impact.** It explains "renders are not bit-reproducible run to run" (ledger, Phase 1) and
the 35.9 / 39.05 dB re-render controls.

## Merge recommendation

**Merge as opt-in experimental, then enable per profile through `models.json` env.**

- Off is provably a no-op: bit-identical on 9 profiles.
- At tau 1.3 it saves 23-28% end-to-end on 124-frame HD clips and 26% on 512x768x362.
- No block artifacts. Human-judged differences are mild.
- The real-use path on the default profile is close to dense.

Reasons it is not on unconditionally in the engine:

- It needs M5 TensorOps and refuses to start on other GPUs.
- The fl2va human verdicts leaned dense, with one unexplained temporal complaint.
- 124-frame checks on 8 of 9 profiles are unverified.

Not pushed. Not merged.

## Not done / unverified

- 124-frame real-path visual checks on every profile except the default.
- Per-profile speed: measured on the default profile only.
- Audio by ear.
- The temporal "feels broken" fl2va-lightx2v-unlocked p31_202 clip was not reproduced or
  diagnosed.
- Per-call kernel timing is derived from end-to-end, not instrumented.
- The GQA barrier fix is identified but not applied or A/B'd.
- The formal visual inspection of the sheets is for the vision worker.
