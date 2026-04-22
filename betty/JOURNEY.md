# facesplatt on Betty — the journey

A chronological account of what it actually took to get FaceLift (ICCV 2025) producing real 3D Gaussian splats for all 120 CFAD portraits on UPenn's Betty HPC cluster, from zero to shipped. Written for anyone running a similar ML inference batch on Betty, so they skip the ~20 walls we hit.

> **Already know the walls, just want the fixes?** Read `PLAYBOOK.md` — it has R1–R21 as a reference card. This file is the story.

---

## Phase 0 · What we were actually trying to do

- **Input:** 120 frontal/profile face photographs (CFAD — Chicago Face Database).
- **Model:** `weijielyu/FaceLift` (ICCV 2025, Adobe Research + UC Merced) — single 2D portrait → 3D Gaussian splat via multi-view diffusion + GS-LRM.
- **Compute:** UPenn PARCC's Betty cluster (NVIDIA B200 / Blackwell).
- **Output:** `.ply` 3DGS files, streamed back to a Mac, rendered in a browser.

Sounds simple on paper. It wasn't.

---

## Phase 1 · Getting INTO Betty

### Wall 1: SSH host key not trusted (R1)
First `ssh jvadala@login.betty.parcc.upenn.edu` → `Host key verification failed.`

**Fix:** `ssh-keyscan -T 10 login.betty.parcc.upenn.edu >> ~/.ssh/known_hosts`. Betty's current ED25519 fingerprint: `SHA256:talnzpFHiLmQR0xFrC8ZaPdQ9LxfDMb/iamK2pbBd7I`.

### Wall 2: Kerberos + Duo 2FA can't be scripted (R2, R3)
`ssh -vv` said: `Authentications that can continue: publickey,gssapi-with-mic` → after Kerberos: `Authenticated using "gssapi-with-mic" with partial success. Authentications that can continue: publickey,keyboard-interactive`.

In plain English: **Betty requires Kerberos AND a Duo push/passcode.** The Duo prompt is `keyboard-interactive`, which no batch tool can answer.

**Fix:** SSH ControlMaster. Open one interactive session, answer Duo once, all subsequent ssh/rsync rides the multiplexed master for 8 hours. See `~/.ssh/config`:
```
Host login.betty.parcc.upenn.edu
    User jvadala
    GSSAPIAuthentication yes
    GSSAPIDelegateCredentials yes
    PreferredAuthentications gssapi-with-mic,keyboard-interactive,publickey
    ControlMaster auto
    ControlPath ~/.ssh/cm/%r@%h:%p
    ControlPersist yes
    ServerAliveInterval 30
```

Bonus: launchd agent (`com.facesplatt.betty-keepalive.plist`) pings Betty every 4 minutes via the master to keep it warm and logs any failure to `/tmp/betty-keepalive.log`.

### Wall 3: No ColdFront project allocation (R4)
`mkdir /vast/projects/jvadala-facesplatt → Permission denied`. Project dirs must be provisioned via ColdFront (https://coldfront.parcc.upenn.edu). New accounts don't have any.

**Fix:** Run from the 50 GB home quota (`/vast/home/j/jvadala/`). All our scripts accept `BETTY_ROOT` env-var override, default to home. Tighter on space but gets you unblocked in 0 minutes instead of waiting 1 hour for ColdFront propagation.

---

## Phase 2 · Building the environment

### Wall 4: `module: command not found` in non-login SSH (R5)
`ssh host 'bash setup.sh'` runs bash as a **non-login shell**. Betty defines Lmod's `module` function in `/etc/profile.d/Z98-lmod.sh`, which is only sourced for login shells.

**Fix:** Source it explicitly at the top of every script that uses `module`:
```bash
if ! command -v module >/dev/null 2>&1; then
  [ -f /etc/profile.d/Z98-lmod.sh ] && source /etc/profile.d/Z98-lmod.sh
fi
```

### Wall 5: FaceLift's `setup_env.sh` calls `sudo apt install ffmpeg` (R8)
There's no `sudo` on shared HPC login nodes. The line kills the script. (We also don't need apt's ffmpeg — we install conda's later.)

**Fix:** Reimplemented `setup_env.sh` minus the sudo/video parts. Clean venv creation + pip installs only.

### Wall 6: `parcc_quota.py: command not found` (R6)
PARCC's helpers live at `/vast/parcc/sw/bin/` but that's not on default `$PATH`.

**Fix:** Always invoke by full path, or `export PATH=/vast/parcc/sw/bin:$PATH` in your `.bashrc`.

### Wall 7: `conda activate` fails, `source activate` works (R7)
Betty's anaconda3 module doesn't run `conda init` for every user. The legacy `source activate <env>` is the supported idiom.

**Fix:** Use `source activate`, never `conda activate`.

### Wall 8: PEP 517 build isolation breaks CUDA-extension installs (R14)
`pip install git+...diff-gaussian-rasterization` fails with `ModuleNotFoundError: No module named 'torch'`.

Why: pip 10+ runs the build in a fresh throwaway venv that only sees deps declared in `pyproject.toml`. The `setup.py` does `import torch` at top to grab `CUDAExtension`, but the build env has no torch.

**Fix:** Preinstall build tools *in the project env*, then pass `--no-build-isolation`:
```bash
pip install ninja setuptools wheel
pip install --no-build-isolation git+https://github.com/graphdeco-inria/diff-gaussian-rasterization
```

### Wall 9: CUDA compile can't auto-detect arch on login node (R15)
After fixing build isolation, the compile itself errored:
```
File "torch/utils/cpp_extension.py", line 1985, in _get_cuda_arch_flags
    arch_list[-1] += '+PTX'
IndexError: list index out of range
```

Why: PyTorch auto-detects the compute capability by querying the attached GPU. Login nodes have **no GPU** (by design — you're supposed to edit there, not compute). Empty list → IndexError.

**Fix:** Set `TORCH_CUDA_ARCH_LIST` explicitly before pip install:
```bash
export TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0;10.0+PTX"
```
This covers A100 / A40 / H100 natively; the `+PTX` fallback lets newer arches JIT at runtime.

### Wall 10: `diff-gaussian-rasterization` doesn't compile on CUDA 12.8 (R16)
After *all* of the above — fresh compile error:
```
namespace "std" has no member "uintptr_t"
identifier "uint32_t" is undefined
```

Why: the 2023-era headers don't `#include <cstdint>`. CUDA 12.4's STL headers transitively pulled it in; CUDA 12.8's stricter headers don't. Upstream hasn't patched it.

**Fix:** Clone, inject the include into every header and `.cu`, build from local path:
```bash
git clone --recursive https://github.com/graphdeco-inria/diff-gaussian-rasterization dgr
cd dgr
for f in cuda_rasterizer/*.h cuda_rasterizer/*.cu rasterize_points.{h,cu}; do
  grep -q "#include <cstdint>" "$f" || sed -i '/#pragma once/a #include <cstdint>' "$f"
done
pip install --no-build-isolation --no-cache-dir .
```

---

## Phase 3 · Running inference on Blackwell (B200)

### Wall 11: PyTorch 2.4 has no Blackwell kernels (R19)
First actual job run. Got to GPU. Loaded the diffusion pipeline. Then on the first real tensor op:
```
RuntimeError: CUDA error: no kernel image is available for execution on the device
NVIDIA B200 with CUDA capability sm_100 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90.
```

Why: FaceLift pins `torch==2.4.0` (cu124). Those wheels pre-date Blackwell (SM 10.0) — they only have kernels through Hopper (H100, SM 9.0).

**Fix:** Upgrade to PyTorch 2.7 + cu128 (Blackwell-supporting wheels). Rebuild `diff-gaussian-rasterization` against the new torch ABI. Also need matching xformers (0.0.30):
```bash
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install xformers==0.0.30
pip install --no-build-isolation --no-cache-dir --force-reinstall ./dgr
```

One subtlety: Betty has `cuda/12.8.1` as a module; torch cu128 wheels are a clean match.

### Wall 12: xformers 0.0.27 has no Blackwell support (R17)
Same root cause as Wall 11, different component. When using the torch 2.4 stack, diffusers' `enable_xformers_memory_efficient_attention()` failed on B200. Even torch 2.6+cu124 wheels didn't have sm_100.

**Fix now** (after going to torch 2.7/cu128): xformers 0.0.30 works. **Fix if stuck on torch 2.4**: comment out the `.enable_xformers_memory_efficient_attention()` call in `inference.py` and rely on PyTorch's native SDPA fallback (~20% slower but robust on any arch).

### Wall 13: `conda-forge ffmpeg` has no libx264 (R18)
Job ran, made first `.ply`, then died:
```
ValueError: Codec libx264 is not available in the installed ffmpeg version.
HINT: For conda users, run `conda remove ffmpeg` and `conda install ffmpeg x264 -c conda-forge`
```

The `turntable.mp4` step requires H.264; conda-forge ffmpeg on Betty was built without it. FaceLift's batch loop doesn't catch the exception — **one bad video call aborted all remaining faces**.

**Fix (best):** Skip video generation entirely — we only want `.ply`, the turntable is a demo artifact. `fix_skip_turntable.py` removes the `render_turntable()` + `imageseq2video()` calls from `inference.py`. Bonus: saves ~5 MB/face of quota AND skips 150 view renders of GPU time per subject.

### Wall 14: `inference.py`'s fallback path leaks RGBA into the diffusion VAE (R21)
Mid-batch, face 55 crashed the entire job:
```
RuntimeError: Given groups=1, weight of size [128, 3, 3, 3],
expected input[6, 4, 512, 512] to have 3 channels, but got 4 channels instead
```

Why: on `CFAD-WF-048-999-211`, MTCNN couldn't find a face. FaceLift's `process_single_image` fell back to `preprocess_image_without_cropping`, which returned an RGBA image (alpha channel included). The rest of the pipeline expected RGB, and the VAE's first conv choked on 4 channels.

**Fix:** Wrap the per-face call in `try/except`. One bad input now logs + skips, batch continues. Patched by `fix_robust.py`.

Result: 116/120 faces succeeded; 4 edge-case inputs (MTCNN no-face on unusual profile/lighting) were cleanly skipped.

---

## Phase 4 · Getting outputs back without blowing quota

### Wall 15: Home quota (50 GB) doesn't fit a full run (R20)
Math: env (~12 GB) + HF checkpoints (~5 GB) + 120 `.ply` × 38 MB ≈ 4.5 GB + intermediate PNGs ≈ 400 MB = peak ~22 GB of artifacts *on top of* the fixed ~17 GB of infrastructure. Leaves single-digit GB margin. Close.

**Fix:** Stream outputs off-cluster during the run with `betty/incremental_pull.sh`. It polls `/vast/home/.../outputs/` every 60 s, rsyncs any completed `gaussians.ply` to local `splats/`, then `rm -rf`'s the remote per-face directory. Betty never holds more than ~2 complete faces at a time — quota stayed flat at ~45.7 GB for the entire 2-hour run.

### Wall 16: Wrong partition = long queue (R12-ish)
First submit went to `b200-mig45` (cheaper MIG slices). 7 jobs ahead with higher priority. Could have sat there for hours.

**Fix:** Switched to `dgx-b200` (full B200 GPU, partition had 212/216 free). New submits ran within seconds. Billing weight is 4× but for a single 2-hour job that's negligible, and the time saved is huge.

---

## Phase 5 · Not-really-Betty gotchas that ate time anyway

### Wall 17: Silly bash/Python patch bug
One patch I wrote to wrap `imageseq2video` in try/except produced malformed Python where the closing `)` ended up *after* the `except:` clause. Script failed SyntaxError on import and the entire batch aborted. Rewrote as a line-by-line parser that walks paren depth.

**Lesson for future me:** when patching multi-line function calls, use a paren-depth tracker, not a regex.

### Wall 18: Accidentally committed 1 GB of `.ply` files to git
The `splats/` dir wasn't in `.gitignore` at the time; the auto-commit swept them in. Pushed 29 large blobs before I caught it.

**Fix:** `.gitignore` now excludes `splats/`, `previews/`, `*.ply`. The repo still has them in history (234 MB of pack) but won't accept new ones. If you care, `git filter-repo` can excise them.

---

## The timeline, in minutes

| minutes | phase | event |
| ---: | --- | --- |
| 0 | auth | kinit, VPN, SSH-host-key add, Duo |
| 5 | auth | ControlMaster stanza, keepalive plist loaded |
| 10 | env | conda env created, pip installs started |
| 25 | env | `diff-gaussian-rasterization` build failed (R14) |
| 28 | env | `--no-build-isolation` fixed that, next failure (R15) |
| 32 | env | `TORCH_CUDA_ARCH_LIST` set, still fails (R16) |
| 38 | env | cstdint-patched clone, finally installs |
| 42 | run | first job submitted → Blackwell failure (R19/R12) |
| 55 | run | torch 2.7 cu128 + xformers 0.0.30 installed |
| 60 | run | xformers+Blackwell works; first `.ply` produced |
| 62 | run | ffmpeg/libx264 kills the batch (R18) |
| 65 | run | `fix_skip_turntable.py` applied |
| 68 | run | 5-face smoke succeeds — 5 `.ply` at 38 MB each |
| 72 | run | full 120 job submitted on `dgx-b200` |
| 85 | run | face 55 crashes the batch (R21) |
| 88 | run | `fix_robust.py` adds try/except wrap |
| 90 | run | resume job for the remaining 65 |
| 140 | run | resume job completes — 116/120 locally, 4.2 GB |

Total: ~2 hours 20 minutes of clock time, maybe 40 minutes of actual thinking; the rest was round-trip on queue/compile.

---

## Reusable lessons for *any* ML batch job on Betty

1. **`doctor.sh` first, always.** Run the pre-flight before every session. It catches the 13 most common walls in 5 seconds and tells you exactly what to fix.

2. **Use `BETTY_ROOT` + home as default.** Don't hardcode `/vast/projects/<x>`. Project allocations come and go.

3. **Put `module` source-guards in every script.** `ssh host 'bash …'` is not a login shell.

4. **Build CUDA extensions with explicit arch list.** Login nodes have no GPU.

5. **Assume ABI churn between torch minor versions.** Any CUDA extension needs rebuilding when you bump torch. Keep the clone, run `--force-reinstall` from it.

6. **For Blackwell (B200), you need cu128 + torch ≥ 2.6.** cu124 wheels stop at Hopper.

7. **Wrap the per-item loop in try/except.** In a 100+ item batch, *something* will be weird. Don't let one bad input kill the whole job.

8. **Stream outputs off-cluster during the run.** Quota constraints disappear. `incremental_pull.sh` is the template.

9. **Prefer a less-congested partition over a cheaper one for short jobs.** Waiting 6 hours in queue to save $0.40 is not a good trade.

10. **Add a `Why:` comment next to every weird env var / patch.** Future you won't remember which of the 5 conflicting conda installs set `TORCH_CUDA_ARCH_LIST`.

---

## Files produced

| path | what |
| --- | --- |
| `betty/doctor.sh` | pre-flight validator; run first |
| `betty/setup_facelift.sh` | one-shot env build (login node safe) |
| `betty/run_facelift.sbatch` | Slurm job script |
| `betty/fix_skip_turntable.py` | idempotent patch to skip video output |
| `betty/fix_robust.py` | adds per-face try/except wrap |
| `betty/sync_up.sh` | rsync cfad + betty/ → Betty |
| `betty/sync_down.sh` | rsync everything back after |
| `betty/incremental_pull.sh` | stream `.ply` files back during the run |
| `betty/stage_smoke.sh` | copy N faces to `cfad_smoke/` for test run |
| `betty/PLAYBOOK.md` | reference-card view of all roadblocks |
| `betty/JOURNEY.md` | this file |

If you're a future UPenn student running FaceLift or a similar model on Betty: clone the repo, read PLAYBOOK.md, run `doctor.sh`, and most of these walls just don't exist for you anymore. That's what all this was for.
