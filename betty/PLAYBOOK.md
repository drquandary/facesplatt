# Betty engineering harness — playbook & roadblocks log

A repeatable recipe for running GPU inference/training jobs on PARCC's **Betty** cluster from a non-interactive agent (e.g. Claude Code on your Mac), plus a chronological log of every roadblock hit while bringing up **facesplatt / FaceLift** so future runs skip the same walls.

Skim this once. Then run `doctor.sh` and it'll re-verify everything mechanically.

---

## What Betty is

- Penn PARCC GPU/CPU HPC cluster at `login.betty.parcc.upenn.edu`
- Slurm scheduler, Lmod modules, Ubuntu 24.04, CUDA 12.8+
- Auth: **Kerberos + Duo 2FA** (no plain password SSH)
- Storage: 50 GB home quota at `/vast/home/<letter>/<pennkey>`; project allocations at `/vast/projects/<name>` (via ColdFront)

---

## Prerequisites (one-time, on your Mac)

| | |
|---|---|
| 1 | **Kerberos client.** macOS ships with Heimdal; `kinit` / `klist` just work. |
| 2 | **Penn VPN or campus network.** Login node is only reachable from inside Penn. |
| 3 | **ColdFront membership.** If you want `/vast/projects/<name>`: request it via https://coldfront.parcc.upenn.edu and your PI approves. Propagation takes up to 1 h. Without a project you can still run jobs from home dir — see "no project allocation" below. |

---

## End-to-end recipe (happy path)

```bash
# 1. Auth
kinit <pennkey>@UPENN.EDU         # password; may prompt Duo

# 2. First SSH in to establish a ControlMaster (Duo prompt once)
ssh login.betty.parcc.upenn.edu 'hostname; date'
# Approve Duo. Master persists 8 h (see ~/.ssh/config).

# 3. Push code + inputs
export BETTY_ROOT=/vast/home/<letter>/<pennkey>/<project>   # or /vast/projects/<name>/<project>
bash betty/sync_up.sh

# 4. One-time remote env setup (login node, no GPU)
ssh login.betty.parcc.upenn.edu \
  "cd ${BETTY_ROOT}/betty && bash setup_facelift.sh"

# 5. Submit job
ssh login.betty.parcc.upenn.edu \
  "cd ${BETTY_ROOT}/betty && sbatch run_facelift.sbatch"

# 6. Watch + retrieve
ssh login.betty.parcc.upenn.edu 'squeue -u $USER'
# when done:
ssh login.betty.parcc.upenn.edu "cd ${BETTY_ROOT}/betty && bash collect_plys.sh"
bash betty/sync_down.sh
```

Run `bash betty/doctor.sh` before each of steps 3–5; it's a 5-second sanity check.

---

## Roadblocks log (chronological, with fixes)

Each one cost 5-15 minutes the first time. The scripts in this directory now avoid each of them.

### R1. SSH host key not trusted on first connect

**Symptom:** `Host key verification failed.`

**Cause:** Betty not in `~/.ssh/known_hosts`.

**Fix** (trust-on-first-use; verify fingerprint separately if strict):

```bash
ssh-keyscan -T 10 login.betty.parcc.upenn.edu >> ~/.ssh/known_hosts
```

Betty's ED25519 fingerprint (as of 2026-04-21): `SHA256:talnzpFHiLmQR0xFrC8ZaPdQ9LxfDMb/iamK2pbBd7I`.

### R2. SSH refuses GSSAPI, asks for password

**Symptom:** `Permission denied (publickey,gssapi-with-mic).`

**Cause:** macOS `ssh` doesn't send Kerberos by default.

**Fix:** `~/.ssh/config` stanza:

```
Host login.betty.parcc.upenn.edu
    User <pennkey>
    GSSAPIAuthentication yes
    GSSAPIDelegateCredentials yes
    PreferredAuthentications gssapi-with-mic,keyboard-interactive,publickey
```

### R3. GSSAPI succeeds but Betty still asks for a second factor

**Symptom:** SSH debug log says `Authenticated using "gssapi-with-mic" with partial success.` followed by `Authentications that can continue: publickey,keyboard-interactive`.

**Cause:** Betty requires Duo 2FA as a mandatory second factor even after Kerberos. `keyboard-interactive` is the Duo prompt. **Non-interactive tools can't answer this.**

**Fix:** SSH **ControlMaster** multiplexing. You open one interactive session (answer Duo once); all subsequent ssh/rsync/scp commands ride the persistent master for 8 h.

```
# Append to the Host block in ~/.ssh/config:
    ControlMaster auto
    ControlPath ~/.ssh/cm/%r@%h:%p
    ControlPersist 8h
```

```bash
mkdir -p ~/.ssh/cm
ssh login.betty.parcc.upenn.edu 'echo ok'   # answer Duo; master starts
ssh -O check login.betty.parcc.upenn.edu    # should say "Master running"
```

All later `ssh`/`rsync`/`scp` automatically reuse it with no re-auth.

### R4. No ColdFront project allocation yet

**Symptom:** `mkdir: cannot create directory ‘/vast/projects/<name>’: Permission denied`

**Cause:** ColdFront membership not yet propagated (or never requested).

**Fix:** Two options.

- **Short-term**: run out of home dir (`/vast/home/<letter>/<pennkey>`). 50 GB quota. All our scripts accept `BETTY_ROOT` override — default is already home dir.
- **Long-term**: request via https://coldfront.parcc.upenn.edu → your PI approves → wait up to 1 h.

Check your current allocation with `/vast/parcc/sw/bin/parcc_quota.py`.

### R5. `module: command not found` in non-interactive SSH

**Symptom:** `setup_facelift.sh: line 20: module: command not found` immediately on script start.

**Cause:** Lmod's `module` shell function is defined by `/etc/profile.d/Z98-lmod.sh`, which is **only sourced in login shells**. `ssh host 'bash script.sh'` is non-login.

**Fix** (now baked into `setup_facelift.sh` and `run_facelift.sbatch`):

```bash
if ! command -v module >/dev/null 2>&1; then
  [ -f /etc/profile.d/Z98-lmod.sh ] && source /etc/profile.d/Z98-lmod.sh
fi
```

Alternative: invoke with `ssh host 'bash -l script.sh'`.

### R6. `parcc_quota.py: command not found`

**Symptom:** Running `parcc_quota.py` returns "not found".

**Cause:** PARCC helper scripts live in `/vast/parcc/sw/bin/` but that dir isn't in the default `$PATH`.

**Fix:** Full path — `/vast/parcc/sw/bin/parcc_quota.py`. All helpers live there:

```
parcc_quota.py  parcc_du.py  parcc_free.py  parcc_sfree.py
parcc_sqos.py   parcc_sreport.py  parcc_sdebug.py
interact        betty-jupyter.sh
```

Add to shell rc if you use them often:
```bash
echo 'export PATH=/vast/parcc/sw/bin:$PATH' >> ~/.bashrc
```

### R7. `conda activate` errors, `source activate` works

**Symptom:** `conda activate envname` fails with init errors.

**Cause:** Betty's shared anaconda3 module doesn't run `conda init` for every user — the legacy `activate` script is the supported path.

**Fix:** Always use `source activate <env-path-or-name>`, never `conda activate`. Scripts in this dir do this.

### R8. FaceLift's `setup_env.sh` does `sudo apt install ffmpeg`

**Symptom:** Would fail on Betty — no sudo on login or compute nodes.

**Cause:** Upstream script assumes a user-owned Linux box.

**Fix:** `setup_facelift.sh` reimplements `setup_env.sh` without the sudo/video steps. We don't need ffmpeg for `.ply` output anyway.

### R9. `diff-gaussian-rasterization` fails to compile

**Symptom:** `pip install git+...diff-gaussian-rasterization` fails with `nvcc: command not found` or `gcc: command not found`.

**Cause:** CUDA toolkit + matching gcc not on PATH when pip runs the extension build.

**Fix:** Load the modules **before** `pip install`, inside the same shell:

```bash
module load cuda/12.8.1
module load gcc/13.3.0
pip install git+https://github.com/graphdeco-inria/diff-gaussian-rasterization
```

### R10. `rembg` / HuggingFace / facenet-pytorch download models to home

**Symptom:** Home quota fills silently during first run.

**Cause:** Default cache paths are `~/.u2net/`, `~/.cache/huggingface/`, `~/.cache/torch/`.

**Fix** (set before any Python runs):

```bash
export HF_HOME=${BETTY_ROOT}/hf_cache
export TRANSFORMERS_CACHE=${BETTY_ROOT}/hf_cache
export HF_DATASETS_CACHE=${BETTY_ROOT}/hf_cache/datasets
export U2NET_HOME=${BETTY_ROOT}/hf_cache/u2net
```

`run_facelift.sbatch` exports all four.

### R11. Login node runs the actual job

**Symptom:** Job killed with `Signal 9` mid-inference, or hammer complaint from sysadmins.

**Cause:** Ran `python inference.py` directly on `login.betty.parcc.upenn.edu`. Login nodes have CPU limits and no GPU.

**Fix:** Only `pip install` / `git clone` / `sbatch` on login. GPU work goes through `sbatch run_facelift.sbatch`.

### R12. Slurm job dies with "Out of memory" or "CUDA OOM"

**Symptom:** Job exits near start; `logs/*.err` mentions OOM.

**Cause:** Wrong partition or insufficient `--mem`.

**Fix:** Right-size partition and memory:

| job type | partition | mem | gpus |
|---|---|---|---|
| FaceLift inference (≤20 GB VRAM) | `b200-mig45` qos `mig` | 64 GB | 1 |
| Full-model fine-tuning | `dgx-b200` qos `dgx` | 256+ GB | 1–8 |
| Large multi-node | `dgx-b200` qos `gpu-max` | as needed | >8 |

Billing weight of `b200-mig45` is 250 vs 1000 for full B200 — use MIG when you can.

### R14. `diff-gaussian-rasterization` pip build fails with `ModuleNotFoundError: No module named 'torch'`

**Symptom:** `Getting requirements to build wheel: finished with status 'error'` → Traceback pointing at `import torch` inside setup.py → `ModuleNotFoundError`.

**Cause:** PEP 517 build isolation. pip runs the build in a throwaway `/tmp/pip-build-env-*` venv that only contains what's declared in `pyproject.toml`. The project's `setup.py` does `import torch` at module top to read `torch.utils.cpp_extension.CUDAExtension`, but the build env has no torch.

**Fix** (baked into `setup_facelift.sh`):
```bash
pip install --no-build-isolation ninja setuptools wheel
pip install --no-build-isolation git+...diff-gaussian-rasterization
```

### R15. CUDA-extension build fails with `IndexError: list index out of range` in `_get_cuda_arch_flags`

**Symptom:** Build gets past `--no-build-isolation`, compiler starts, then:
```
File ".../torch/utils/cpp_extension.py", line 1985, in _get_cuda_arch_flags
    arch_list[-1] += '+PTX'
    ~~~~~~~~~^^^^
IndexError: list index out of range
```

**Cause:** Login nodes have **no GPU**. PyTorch auto-detects arch by querying the device; with no device, `arch_list` is empty, and `arch_list[-1]` raises. The build script has no fallback.

**Fix:** Set `TORCH_CUDA_ARCH_LIST` explicitly before pip install. For B200 clusters:
```bash
export TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0+PTX"
```
This compiles native code for A100/H100/L40 and a PTX fallback that newer arches (Blackwell B200 sm_100) JIT-compile at runtime. Baked into `setup_facelift.sh`.

### R16. `diff-gaussian-rasterization` compile fails with `namespace "std" has no member "uintptr_t"` / `identifier "uint32_t" is undefined`

**Symptom:** nvcc compiles, then errors in `cuda_rasterizer/rasterizer_impl.h`:
```
error: namespace "std" has no member "uintptr_t"
error: identifier "uint32_t" is undefined
```

**Cause:** The 2023-era dgr headers don't `#include <cstdint>`. CUDA 12.4's STL headers transitively pulled it in; CUDA 12.8's stricter headers don't. Upstream hasn't patched it.

**Fix** (baked into `setup_facelift.sh`): clone locally, inject the include in every header/cu file that uses uint32_t / uintptr_t, install from the patched path:

```bash
git clone --recursive https://github.com/graphdeco-inria/diff-gaussian-rasterization dgr
cd dgr
for f in cuda_rasterizer/*.h cuda_rasterizer/*.cu rasterize_points.{h,cu}; do
  grep -q "#include <cstdint>" "$f" || sed -i '/#pragma once/a #include <cstdint>' "$f"
done
pip install --no-build-isolation --no-cache-dir .
```

### R13. Kerberos ticket expires mid-job

**Symptom:** After ~10 h, `rsync`/`ssh` starts prompting for password.

**Cause:** Default ticket lifetime ~10 h.

**Fix:** `kinit -R` to renew (if ticket is renewable), or `kinit <pennkey>@UPENN.EDU` again. Check with `klist`.

---

## The doctor

`bash betty/doctor.sh` verifies all of the above in seconds. Run it before each fresh session. It checks:

1. Kerberos ticket (klist)
2. Penn VPN / Betty reachability
3. Host key in known_hosts
4. SSH ControlMaster socket live
5. Lmod sourcing from non-login shell
6. `$BETTY_ROOT` writable + quota remaining
7. conda / module / nvidia-smi availability once on Betty

---

## Adding a new job to the harness

1. Drop your job script into `betty/` as `run_<whatever>.sbatch`. Copy the header from `run_facelift.sbatch` (module source + env paths + HF cache).
2. If you need extra Python deps, extend `setup_facelift.sh` or clone a new env next to it.
3. Before submitting, run `doctor.sh`.
4. Log any new roadblock here under a new R-number. **This doc is the harness's tribal-knowledge database.**

---

## Related

- Upstream Betty agent docs: https://github.com/drquandary/betty-agent
- PARCC onboarding: https://parcc.upenn.edu/training/getting-started/
- FaceLift upstream: https://github.com/weijielyu/FaceLift
