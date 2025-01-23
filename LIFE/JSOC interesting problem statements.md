### [Support for AMGDPU and Apple Silicon](https://julialang.org/jsoc/gsoc/gnn/#support_for_amgdpu_and_apple_silicon)

We currently support scatter/gather operation only on CPU and CUDA hardware. We aim to extend this to AMDGPU and Apple Silicon leveraging KernelAbstractions.jl, AMDGPU.jl, and Metal.jl.

**Duration**: 175h.

**Expected difficulty**: medium.

**Expected outcome**: Graph convolution speedup for AMD GPU and Apple hardware, performance roughly on par with CUDA.

## [Mentors](https://julialang.org/jsoc/gsoc/gnn/#mentors) 

[Carlo Lucibello](https://github.com/CarloLucibello) (author of [GraphNeuralNetworks.jl](https://github.com/JuliaGraphs/GraphNeuralNetworks.jl)). Feel free to contact me on the [Julia Slack Workspace](https://julialang.slack.com/) or by opening an issue in the GitHub repo.

## [Writing Julia-native kernels for common NN operations](https://fluxml.ai/gsoc/#writing_julia-native_kernels_for_common_nn_operations)

Implement optimized kernels for common neural network operations for which we don't already have Julia-native implementations. This project will require experience with GPU kernel writing and performance optimizations.

**Difficulty.** Hard. **Duration.** 350 hours

### [Description](https://fluxml.ai/gsoc/#description)

Many ML frameworks are making the move away from vendor-specific libraries (like CUBLAS, CUDNN, etc.) towards more generic, JIT-compiled implementations of ML-related kernels, like BLAS, softmax, ReLU, etc. The reasons for this move are many-fold:

- Vendor-provided libraries often only work on that vendor's hardware and software
    
- These libraries only support certain element types, tensor shapes/sizes, and limited array view/stride/transpose support
    
- These libraries often expect to be executed from the host, without a device-side launchable equivalent
    
- These libraries have unreliable build systems or are binary blobs
    

Improving this state of affairs for Flux will involve using Julia's existing GPU and compute kernel libraries (e.g KernelAbstractions.jl) to implement various accelerated, cross-vendor routines. These kernels should be both composable and performance competitive with Flux's current generic code paths. Examples of routines specifically useful for implementing Neural Networks include:

- GEMM and GEMV
    
- Softmax
    
- Batchnorm and Layernorm
    
- ReLU
    
- Convolution/correlation
    

The ideal candidate should have experience with what operations are used in popular ML models and how they are commonly implemented on GPU. This includes experience writing and benchmarking high performance GPU kernels. Because kernels will be required for both training and inference, an understanding of automatic differentiation (AD) is also highly recommended.

**Mentors.** [Julian Samaroo](https://github.com/jpsamaroo), [Kyle Daruwalla](https://github.com/darsnack), [Brian Chen](https://github.com/ToucheSir)

### [Prerequisites](https://fluxml.ai/gsoc/#prerequisites)

- Julia language fluency is essential.
    
- Experience with low-level GPU kernel programming is strongly recommended.
    
- Experience with common primitive machine learning ops (forwards and backwards passes) and their interaction is recommended.
    
- Familiarity with existing prior art such as [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) is preferred.
    

### [Your contributions](https://fluxml.ai/gsoc/#your_contributions)

- A new package containing the optimized kernels and any supporting code for integration into Flux/Flux's operation library [NNlib.jl](https://github.com/FluxML/NNlib.jl).
    
- Tests on CI and a simple benchmark harness for the new NN kernel library.
    
- A proof-of-concept example showing the kernels being used with kernel fusion on device (GPU).
### resources
https://clang.llvm.org/get_started.html
https://llvm.org/docs/
https://arxiv.org/pdf/2201.11811
https://compress-pdf-free.obar.info/download/compresspdf
https://cdrdv2-public.intel.com/830231/oneapi_programming-guide_2025.0-771723-830231.pdf
https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf
https://www.openacc.org/sites/default/files/inline-files/OpenACC_Programming_Guide_0_0.pdf

## [GPU support for NormalizingFlows.jl and Bijectors.jl](https://julialang.org/jsoc/gsoc/turing/#gpu_support_for_normalizingflowsjl_and_bijectorsjl)

**Mentors:** Tor Fjelde, Tim Hargreaves, Xianda Sun, Kai Xu, Hong Ge

**Project difficulty:** Hard

**Project length:** 175 hrs or 350 hrs

**Description:** Bijectors.jl, a package that facilitates transformations of distributions within Turing.jl, currently lacks full GPU compatibility. This limitation stems partly from the implementation details of certain bijectors and also from how some distributions are implemented in the Distributions.jl package. NormalizingFlows.jl, a newer addition to the Turing.jl ecosystem built atop Bijectors.jl, offers a user-friendly interface and utility functions for training normalizing flows but shares the same GPU compatibility issues.

The aim of this project is to enhance GPU support for both Bijectors.jl and NormalizingFlows.jl.

## [Targets for Benchmarking Samplers with vectorization, GPU and high-order derivative supports](https://julialang.org/jsoc/gsoc/turing/#targets_for_benchmarking_samplers_with_vectorization_gpu_and_high-order_derivative_supports)

**Mentors:** Kai Xu, Hong Ge

**Project difficulty:** Medium

**Project length:** 175 hrs

**Description:** The project aims to develop a comprehensive collection of target distributions designed to study and benchmark Markov Chain Monte Carlo (MCMC) samplers in various computational environments. This collection will be an extension and enhancement of the existing Julia package, [VecTargets.jl](https://github.com/xukai92/VecTargets.jl), which currently offers limited support for vectorization, GPU acceleration, and high-order derivatives. The main objectives of this project include:

- Ensuring that the target distributions fully support vectorization and GPU acceleration
    
- Making high-order derivatives (up to 3rd order) seamlessly integrable with the target distributions
    
- Creating a clear and comprehensive documentation that outlines the capabilities and limitations of the project, including explicit details on cases where vectorization, GPU acceleration, or high-order derivatives are not supported.
    
- Investigating and documenting how different Automatic Differentiation (AD) packages available in Julia can be combined or utilized to achieve efficient and accurate computation of high-order derivatives.
    

By achieving these goals, the project aims to offer a robust framework that can significantly contribute to the research and development of more efficient and powerful MCMC samplers, thereby advancing the field of computational statistics and machine learning.

# [Event-chain Monte Carlo methods – Summer of Code](https://julialang.org/jsoc/gsoc/pdmp/#event-chain_monte_carlo_methods_summer_of_code)

## [Massive parallel factorized bouncy particle sampler](https://julialang.org/jsoc/gsoc/pdmp/#massive_parallel_factorized_bouncy_particle_sampler)

At JuliaCon 2021 a new sampler Monte Carlo method (for example as sampling algorithm for the posterior in Bayesian inference) was introduced [1]. The method exploits the factorization structure to sample _a single_ continuous time Markov chain targeting a joint distribution in parallel. In contrast to parallel Gibbs sampling in the method at no time a subset of coordinates is kept fixed. In Gibbs sampling keeping a subset fixed is the main device to achieve massive parallelism: given a separating set of coordinates, the conditional posterior factorizes into independent subproblems. In the presented method, a particle representing a parameter vector sampled from the posterior never ceases to move, and it is only the decisions about changes of the direction of the movement which happen in parallel on subsets of coordinates.

There are already two implementations available which make use of Julias multithreading capabilities. Starting from that, the contributor implements a version of the algorithm using GPU computing techniques as the methods is are suitable for these approaches.

**Expected Results**: Implement massive parallel factorized bouncy particle sampler [1,2] using GPU computing.

**Recommended Skills**: GPU computing, Markov processes, Bayesian inference.

**Mentors**: [Moritz Schauer](https://github.com/mschauer)

**Rating**: Hard, 350 hours

[1] Moritz Schauer: ZigZagBoomerang.jl - parallel inference and variable selection. JuliaCon 2021 contribution [https://pretalx.com/juliacon2021/talk/LUVWJZ/], Youtube: [https://www.youtube.com/watch?v=wJAjP_I1BnQ], 2021.

[2] Joris Bierkens, Paul Fearnhead, Gareth Roberts: The Zig-Zag Process and Super-Efficient Sampling for Bayesian Analysis of Big Data. The Annals of Statistics, 2019, 47. Vol., Nr. 3, pp. 1288-1320. [https://arxiv.org/abs/1607.03188].

