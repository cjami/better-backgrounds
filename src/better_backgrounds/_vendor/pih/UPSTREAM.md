# Adobe PIH inference runtime

- Upstream: https://github.com/adobe/PIH
- Revision: `2823cccf0778c6ea213a3d366f03864ac8ab82e6`
- Authors: Ke Wang, Michaël Gharbi, He Zhang, Zhihao Xia, and Eli Shechtman.
- Paper: *Semi-supervised Parametric Real-world Image Harmonization*, CVPR 2023.
- License: Apache License 2.0, https://www.apache.org/licenses/LICENSE-2.0
- Included: an adapted inference-only ResNet-50 curve predictor and gain-map U-Net.
- Excluded: training code, discriminators, demos, datasets, UI, example media, and checkpoint.

Better Backgrounds supplies the official checkpoint externally. The adapted model preserves the
official state-dictionary names and inference equations while returning RGB curves and the gain map
explicitly instead of mutating model attributes. The expanded 32-cubed upstream curve table is
replaced by its mathematically equivalent separable one-dimensional interpolation.
