# Apple SHARP inference runtime

- Upstream: https://github.com/apple/ml-sharp
- Revision: `1eaa046834b81852261262b41b0919f5c1efdd2e`
- Software license: Apple sample-code license; see `LICENSE`.
- Model license: Apple Machine Learning Research Model License; see `LICENSE_MODEL`.
- Included: predictor models and the utilities required for prediction and PLY export.
- Excluded: Click CLI, trajectory rendering, gsplat integration, video/image writers,
  visualisation, training assets, examples, and demo data.

Better Backgrounds imports this code only inside the dedicated SHARP build worker.
