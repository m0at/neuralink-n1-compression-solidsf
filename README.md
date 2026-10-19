# Neuralink N1 Brainwire Lossless Codec (3.47x) & All-6-Challenge Engineering Suite

**Author:** Andy Norris (`andy@solidsf.com`), Founder & CEO, solidSF  
**Formal Monograph DOI:** [10.5281/zenodo.23486374](https://doi.org/10.5281/zenodo.23486374)  
**Open-Source DataRepo PR:** [neuralinkcorp/datarepo#83](https://github.com/neuralinkcorp/datarepo/pull/83)

---

## 1. Track 1: Neuralink Brainwire Electrode Signal Compression (`3.47x` / `3.4692x` Exact)

Lossless C99 codec (`n1_codec.c`) for the 743-file (`73,383,917`-sample, `19,531 Hz`) Neuralink N1 extracellular recording dataset:

- **Exact 10-Bit Beatty Lattice Pullback (`Theorem 2`):** Inverts `y(k) = trunc(-32768 + 65535*k/1023)` into `k in {0..1023}` with zero residual across all `73,383,917` samples.
- **12-Tap Forgetting-Factor RLS + 3-Stage LMS Residual Cascade (`N1-SAR-RLS-Cascade-v2`):** Tracks non-stationary LFP drift and sharp action-potential troughs with sub-millisecond causal latency (`32 samples = 1.6384 ms`).
- **N1 Split-Capacitor SAR ADC DNL Prior + Causal Online Adaptation (`Theorem 19`):** Incorporates the empirical N1 SAR ADC carry-transition code density prior (`k mod 16 == 15` suppressed by `-30.9%`) blended with causal local counts (`online_cnt[c]`) conditioned on `k_{t-1} & 15` inside a 32-bit range coder.

### Verified Full-Corpus Benchmark (`743 / 743` files, `diff -q == 0`)

| Codec / Submission | Uncompressed Size | Compressed Size (incl. binaries) | Bits / Sample | Official `eval.sh` Ratio |
| :--- | :--- | :--- | :--- | :--- |
| **solidSF `N1-SAR-RLS-Cascade-v2` (Single-File Official)** | `146,800,526 B` | **`42,314,827 B`** | **`4.6120`** | **`3.47x` (`3.4692x`)** |
| **solidSF + Synchronous Dyadic Pair Mode (`Theorem 8`)** | `146,800,526 B` | **`41,444,487 B`** | **`4.5171`** | **`3.54x` (`3.5421x`)** |
| `hxrdxkxvxd/neuralink-codec` | `146,800,526 B` | `42,458,575 B` | `4.6276` | `3.4575x` |
| `phoboslab/neuralink_brainwire` | `146,800,526 B` | `43,821,052 B` | `4.7761` | `3.35x` |
| `xdsopl/brainwire` | `146,800,526 B` | `44,084,242 B` | `4.8048` | `3.33x` |
| `mikaelhaji/n1-codec` | `146,800,526 B` | `47,202,741 B` | `5.1447` | `3.11x` |
| `SanderGi/neuralink-compression` | `146,800,526 B` | `50,796,029 B` | `5.5363` | `2.89x` |
| `zip` baseline | `146,800,526 B` | `66,727,511 B` | `7.2727` | `2.20x` |

### Quick Reproduce
```bash
./build.sh
./eval.sh /path/to/neuralink/data
```

---

## 2. All 6 Neuralink Engineering Challenge Tracks Included

1. [`challenge1_compression/`](challenge1_compression/) — Brainwire Lossless Electrode Signal Codec (`3.4692x` single-file, `3.5421x` dyadic pair mode, `200.19x` Tier-2 Spike-Worldtube).
2. [`challenge2_implant_handwriting/`](challenge2_implant_handwriting/) — Implant Challenge (`mouse_velocities.csv` cursor velocity trajectory integration + Unicode Tag `U+E0000` prompt-injection stripping -> `MONKEY MINDPONG`, `SHA-256 = 8acb1410c5af0ff76da758b9a0178c8efe34ba3f9d80417c849b3f3911799586`).
3. [`challenge3_firmware_nlk_hw/`](challenge3_firmware_nlk_hw/) — Embedded Firmware Challenge (`nlk_hw.c` bare-metal C99 SPI Mode 0 temperature controller, JEDEC `0x36/0xC9` verification, `310 K +/- 1 K` hysteresis on GPIO 4, `14 uA` `0xB9` sleep, millikelvin alpha-beta filter).
4. [`challenge4_grpc_image_service/`](challenge4_grpc_image_service/) — Software Engineering Take-Home (`NLImageService` gRPC pipeline implementing `SO(2)` `RotateImage` and boundary-aware `3x3` `MeanFilter` on Grayscale & RGB).
5. [`challenge5_webgrid_bci_controller/`](challenge5_webgrid_bci_controller/) — Live Webgrid BCI Cursor-Control Benchmark (`SE(2)` Spike-Jet + Class I `PV+` PING `+29.8 deg / 0.690 ms` braking law lifting sustained throughput from `11.45 -> 20.77 BPS` on `30x30x1` and `5.21 -> 23.05 BPS` on `100x100x3`).
6. [`challenge6_neuralinkcorp_datarepo/`](challenge6_neuralinkcorp_datarepo/) — Open-Source DataRepo Issue #80 Fix ([PR #83](https://github.com/neuralinkcorp/datarepo/pull/83)).
7. [`formal_monograph_lean4/`](formal_monograph_lean4/) — 17-page LaTeX monograph + 24 machine-checked Lean 4 theorems (`NeuralinkPixeleanVoxeleanCohort.lean`, `#assert_zfc_axioms`) + 72-assertion Python verification oracle (`DOI: 10.5281/zenodo.23486374`).
