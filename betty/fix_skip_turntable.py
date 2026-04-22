"""Skip render_turntable() + imageseq2video() in FaceLift's inference.py.

Why:
  - We only need gaussians.ply for holographic rendering.
  - turntable.mp4 is a demo rendering, not used downstream.
  - Betty's conda-forge ffmpeg has no libx264 → videoio raises ValueError.
  - render_turntable costs real GPU time (150 views).

Idempotent: finds the block bounded by `vis_image = render_turntable(` through
the closing ) of `imageseq2video(`. Replaces with a comment. Safe to re-run.
"""
import os
import pathlib


def main():
    root = os.environ.get("FACELIFT_REPO") or \
        os.environ.get("FP_ROOT", "/vast/home/j/jvadala/facesplatt") + "/FaceLift"
    p = pathlib.Path(root) / "inference.py"
    if not p.exists():
        print(f"SKIP — {p} not found")
        return

    lines = p.read_text().splitlines(keepends=True)

    # Find starting line of `    vis_image = render_turntable(`
    start = None
    for i, ln in enumerate(lines):
        if "vis_image = render_turntable(" in ln:
            start = i
            break

    if start is None:
        print("ALREADY_PATCHED — no render_turntable() call remains")
        return

    # Walk forward to the closing ) of imageseq2video(
    i = start
    found_imageseq = False
    depth = 0
    end = None
    while i < len(lines):
        if "imageseq2video(" in lines[i]:
            found_imageseq = True
            depth = lines[i].count("(") - lines[i].count(")")
            i += 1
            continue
        if found_imageseq:
            depth += lines[i].count("(") - lines[i].count(")")
            if depth <= 0:
                end = i
                break
        i += 1

    if end is None:
        print("FAIL — couldn't find end of imageseq2video call")
        return

    new_lines = (
        lines[:start]
        + ["    # patched: skip turntable video (ffmpeg libx264 missing on Betty;\n",
           "    # we only need gaussians.ply)\n"]
        + lines[end + 1:]
    )
    p.write_text("".join(new_lines))
    print(f"PATCHED — removed lines {start + 1}..{end + 1}")


if __name__ == "__main__":
    main()
