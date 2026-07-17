# Harmonizer inference runtime

- Upstream: https://github.com/ZHKKKe/Harmonizer
- Revision: `48ecd70becbff50ccaf576db0e64212dbc494e26`
- Authors: Zhanghan Ke, Chunyi Sun, Lei Zhu, Ke Xu, and Rynson W. H. Lau.
- Paper: *Harmonizer: Learning to Perform White-Box Image and Video Harmonization*, ECCV 2022.
- License: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International,
  https://creativecommons.org/licenses/by-nc-sa/4.0/
- Included: an adapted inference-only backbone and cascade argument regressor.
- Excluded: training code, validation code, demos, datasets, example media, filters, and checkpoint.

Better Backgrounds supplies the official checkpoint externally and uses this subset only for its
non-commercial hackathon build. The implementation has been modernised for the current runtime,
uses the separately packaged EfficientNet-PyTorch backbone, and supplies six global controls to the
application's session-compiled native renderer. That renderer preserves the trained filter order and
equations within bounded 8-bit interpolation error without retaining a per-frame Torch workload.
