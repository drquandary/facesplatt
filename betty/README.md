# facesplatt · Betty job package

Run FaceLift inference on all 120 CFAD faces on the PARCC Betty cluster, then pull the `.ply` Gaussian-splat files back to your Mac.

## What this package contains

| file | purpose |
| --- | --- |
| `sync_up.sh` | rsync `cfad/` + `betty/` from your Mac up to Betty |
| `setup_facelift.sh` | run once on Betty: clone FaceLift, create conda env |
| `run_facelift.sbatch` | Slurm job: processes all `cfad/*.png` → `outputs/<img>/gaussians.ply` |
| `collect_plys.sh` | flatten `outputs/*/gaussians.ply` → `splats/<id>.ply` for download |
| `sync_down.sh` | rsync `splats/` + sample renders back to your Mac |

## One-time per-project setup

**On your Mac — edit these two variables at the top of every `.sh` / `.sbatch` file:**

```bash
BETTY_PROJECT=jvadala-facesplatt         # your /vast/projects/<name> allocation
BETTY_USER=jvadala                       # your PennKey
```

The scripts all read the same pair. Use a ColdFront project you have write access to.

## Step-by-step

### 1. Push code + faces to Betty (from your Mac)

```bash
kinit jvadala@UPENN.EDU                  # Kerberos ticket
cd ~/facesplatt
bash betty/sync_up.sh
```

### 2. One-time environment setup (on Betty)

```bash
ssh jvadala@login.betty.parcc.upenn.edu
cd /vast/projects/$BETTY_PROJECT/facesplatt/betty
bash setup_facelift.sh                    # ~20 min; creates conda env + clones FaceLift
```

### 3. Submit the job

```bash
sbatch run_facelift.sbatch                # submits to b200-mig45 (QOS=mig)
squeue -u $USER                           # watch it
```

Expected runtime: **~2 hours** for 120 faces on a single B200 MIG-45 slice.
First run also downloads ~5 GB of model weights from HuggingFace on the compute node.

### 4. Collect + pull back (on Mac, after job finishes)

```bash
# On Betty, after the job succeeds:
bash collect_plys.sh

# Back on Mac:
bash betty/sync_down.sh
# → ./splats/*.ply lands locally
```

## Gotchas

- **Login nodes are for editing only.** `setup_facelift.sh` runs package installs on the login node (no GPU needed). The actual inference runs via `sbatch`. Don't `python inference.py` on the login node — you'll get killed.
- **`source activate` not `conda activate`** on Betty (it's in the cluster guide — their conda init expects this).
- **HF cache goes to project storage.** 50 GB home quota fills instantly otherwise. The Slurm script sets `HF_HOME` for you.
- **Kerberos ticket expires after 10 h.** If `sync_down.sh` prompts for password, re-run `kinit jvadala@UPENN.EDU`.
- **`diff-gaussian-rasterization` compiles CUDA kernels** during setup. Must load `cuda/12.8.1` and `gcc/13.3.0` modules first — `setup_facelift.sh` does this.
- **`rembg` downloads a U²-Net model on first run.** We redirect `U2NET_HOME` to project storage.

## If the job fails

```bash
parcc_sdebug.py --job <JOBID>             # inspect Slurm-side issues
tail -100 logs/facelift_<JOBID>.err       # FaceLift error output
```

Most common cause: `diff-gaussian-rasterization` fails to compile because `cuda/12.8.1` module wasn't loaded when you ran `setup_facelift.sh`. Re-source it and re-run just the `pip install` lines.

## Cost estimate (billing weight)

- `b200-mig45` billing weight: 250 per GPU-hour
- Job length: ~2 h × 1 GPU = ~500 SU
- Well under `mig` QOS quota (`max_gpus=8`)
