# Source before running headless Chromium tests on this machine (no system browser deps, no GPU).
D="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/.deps/root"
export LD_LIBRARY_PATH=$D/usr/lib/x86_64-linux-gnu
export VK_ICD_FILENAMES=$D/usr/share/vulkan/icd.d/lvp_icd.json
