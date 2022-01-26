#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


def _try_register_locomotion_envs():
    import habitat.tasks.locomotion.locomotion_base_env
    import habitat.tasks.locomotion.energy_locomotion
    import habitat.tasks.locomotion.stand_locomotion

