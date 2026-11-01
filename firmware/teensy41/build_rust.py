"""PlatformIO pre-build script: build rust/ (bench-decide) and link it.

The schedule decisions in the firmware come from contingency-rs, built
without std for the Teensy 4.1 core (Cortex-M7, hard float:
`thumbv7em-none-eabihf`). rust/Cargo.toml selects the features
(`default-features = false`, `c-runtime`: the allocator and panic handler
call the C library's malloc/free/abort).

Set CONTINGENCY_RS_DIR to a local contingency-rs checkout to build against
it instead of the git dependency in rust/Cargo.toml.
"""

import os
import subprocess

Import("env")  # noqa: F821  (provided by PlatformIO)

TARGET = "thumbv7em-none-eabihf"
GIT_URL = "https://github.com/OperantKit/contingency-rs"

project_dir = env.subst("$PROJECT_DIR")  # noqa: F821
crate_dir = os.path.join(project_dir, "rust")
lib_dir = os.path.join(crate_dir, "target", TARGET, "release")

cmd = ["cargo", "build", "--release", "--target", TARGET]
local = os.environ.get("CONTINGENCY_RS_DIR")
if local:
    cmd += ["--config", f'patch."{GIT_URL}".contingency.path="{os.path.abspath(local)}"']

print("build_rust.py:", " ".join(cmd))
result = subprocess.run(cmd, cwd=crate_dir)
if result.returncode != 0:
    env.Exit(result.returncode)  # noqa: F821

env.Append(LIBPATH=[lib_dir], LIBS=["bench_decide"])  # noqa: F821
