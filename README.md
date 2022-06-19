# Comparing CMA-ES with IPOP-MAES

The objective of this project is to investigate the performance of the MA-ES algorithm extended by the IPOP heuristics. The modified algorithm should be compared with the CMA-ES algorithm.
For this purpose we use the BBOB benchmark from the [coco](https://github.com/numbbo/coco) framework.

## Requirements
The code was tested on Windows and Linux. We recommend using [poetry](https://python-poetry.org/) for package versioning and installation.
Coco must be built locally, refer to coco documentation for instructions.

## Reproducing results
1. Install dependencies with `make install`
2. Activate poetry shell with `make shell`
3. Run benchmarks with `make run`
4. Run post processing with `make postprocessing`
5. The results should be visible in your browser opened automatically.

## References
- A. Auger, N. Hansen. “A restart CMA evolution strategy with increasing
population size”. W: 2005 IEEE Congress on Evolutionary Computation.
T. 2. 2005, 1769–1776 Vol. 2. doi: 10.1109/CEC.2005.1554902
- Hans-Georg Beyer, Bernhard Sendhoff. “Simplify Your Covariance Matrix
Adaptation Evolution Strategy”. W: IEEE Transactions on Evolutionary
Computation 21.5 (2017), p. 746–759. doi: 10.1109/TEVC.2017.2680320.