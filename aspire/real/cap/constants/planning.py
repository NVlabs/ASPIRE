# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

# Shared planning defaults for the main freespace/curobo path.
DEFAULT_IK_POSITION_THRESHOLD_M = 0.005
DEFAULT_IK_ROT_THRESHOLD_RAD = 0.05
DEFAULT_IK_ROT_THRESHOLD_DEG = float(np.degrees(DEFAULT_IK_ROT_THRESHOLD_RAD))
