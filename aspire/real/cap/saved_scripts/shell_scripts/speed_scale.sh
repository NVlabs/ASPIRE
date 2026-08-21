# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Shared YAM demo speed helpers. Keep this file source-only.

yam_speed_scale() {
  awk -v raw="${YAM_FULL_DEMO_SPEED_SCALE:-1.0}" 'BEGIN {
    scale = raw + 0.0
    if (scale <= 0.0) {
      scale = 1.0
    }
    if (scale < 0.25) {
      scale = 0.25
    }
    if (scale > 2.0) {
      scale = 2.0
    }
    printf "%.3f", scale
  }'
}

yam_speed_mul() {
  awk -v base="$1" -v scale="$(yam_speed_scale)" 'BEGIN {
    printf "%.3f", base * scale
  }'
}

yam_speed_div() {
  awk -v base="$1" -v scale="$(yam_speed_scale)" 'BEGIN {
    printf "%.3f", base / scale
  }'
}
