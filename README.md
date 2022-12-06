# ADCB

This repo is the simulator and environment proposed in the paper ["ADCB: An Alzheimer’s disease simulator for benchmarking
observational estimators of causal effects"](https://proceedings.mlr.press/v174/kinyanjui22a/kinyanjui22a.pdf) from the Conference on Health, Inference, and Learning (CHIL) 2022


## Abstract
Abstract Simulators make unique benchmarks for causal effect estimation as they do not rely on unverifiable assumptions or the ability to intervene on real-world systems. This is especially important for estimators targeting healthcare applications as possibilities for experimentation are limited with good reason. We develop a simulator of clinical variables associated with Alzheimer’s disease, aimed to serve as a benchmark for causal effect estimation while modeling intricacies of healthcare data. We fit the system to the Alzheimer’s Disease Neuroimaging Initiative (ADNI) dataset and ground hand-crafted components in results from comparative treatment trials and observational treatment patterns. The simulator includes parameters which alter the nature and difficulty of the causal inference tasks, such as latent variables, effect heterogeneity, length of observed subject history, behavior policy and sample size. We use the simulator to compare standard estimators of average and conditional treatment effects.
