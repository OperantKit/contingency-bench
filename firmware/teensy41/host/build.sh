#!/usr/bin/env bash
# Build the firmware for the host: src/main.cpp unchanged, the Arduino API of
# this directory, and bench-decide from rust/, linked into one executable.
#
# Standard output is the path of the executable and nothing else; the compilers
# write to standard error. CXX selects the C++ compiler (default: c++).
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
firmware="$(dirname "$here")"
crate="$firmware/rust"
out="$here/build"
cxx="${CXX:-c++}"

for tool in cargo rustc "$cxx"; do
    if ! command -v "$tool" >/dev/null; then
        echo "build.sh: $tool not found" >&2
        exit 1
    fi
done

# One target for both compilers: on an arm64 Mac, cargo may be an x86_64 one.
cd "$crate"
target="$(rustc -vV | sed -n 's/^host: //p')"
arch_flags=()
link_flags=()
case "$target" in
    aarch64-apple-darwin) arch_flags=(-arch arm64) ;;
    x86_64-apple-darwin) arch_flags=(-arch x86_64) ;;
    *-linux-*) link_flags=(-pthread -ldl -lm) ;;
esac

target_dir="${CARGO_TARGET_DIR:-$crate/target}"
cargo build --target "$target" --target-dir "$target_dir" >&2

mkdir -p "$out"
"$cxx" ${arch_flags[@]+"${arch_flags[@]}"} -std=c++17 -O2 -Wall -Wextra \
    -I "$here" -I "$firmware/include" \
    "$firmware/src/main.cpp" "$here/host_main.cpp" \
    "$target_dir/$target/debug/libbench_decide.a" \
    ${link_flags[@]+"${link_flags[@]}"} \
    -o "$out/fw_host" >&2
echo "$out/fw_host"
