# ML Performance Playground

My hands-on experiments with JAX, Pallas, TPUs, and transformer performance, alongside the [Scaling book](https://jax-ml.github.io/scaling-book/). I’m exploring how the code we write maps to the compiler, memory system, and hardware, and what that means for performance.

## 1. JAX and transformers

[jax_transformers_playground.ipynb](https://github.com/Superfawll/ml_performance_playground/blob/main/jax_transformers_playground.ipynb) · [Open in Colab](https://colab.research.google.com/github/Superfawll/ml_performance_playground/blob/main/jax_transformers_playground.ipynb)

**What’s there**

- A small decoder-only transformer example: embeddings, causal attention, an MLP, normalization, vocabulary projection, next-token loss, and a training step.
- JAX transformations, including `jit`, `value_and_grad`, `jvp`, `vjp`, `vmap`, and `jacfwd`.
- Inspection of Jaxpr, StableHLO, and compiled HLO to follow the lowering process.
- An initial comparison of eager and JIT-compiled execution, and experiments with `lax.scan` and Python loops.

**What’s to come**

More profiling and short performance comparisons to connect transformer operations and JAX transformations to their execution on hardware.

## 2. Pallas and TPU kernels

[pallas_playground.ipynb](https://github.com/Superfawll/ml_performance_playground/blob/main/pallas_playground.ipynb) · [Open in Colab](https://colab.research.google.com/github/Superfawll/ml_performance_playground/blob/main/pallas_playground.ipynb)

**What’s there**

- Pallas basics through vector operations and tiled matrix-vector multiplication.
- Explicit movement between HBM and VMEM, using synchronous copies, asynchronous DMA, and semaphores.
- Pipelining and double buffering to explore latency hiding and dependencies between loads, computation, and stores.
- Experiments with input mutation and custom JVP rules.
- Profiling hooks for inspecting kernel execution with XProf.

These experiments use a TPU runtime; the notebook includes setup cells for JAX, libtpu, and XProf.

**What’s to come**

- Larger matrix-multiplication experiments, including matrices that exceed one device’s HBM capacity and require sharding and ICI communication.
- Transformer inference and KV caching.
- Ring Attention, PagedAttention, FlashAttention, and KV-cache sharing.
- Profiling and performance analysis of transformer inference.

## 3. Sharded matrix multiplication

[sharded_matmul_playground.ipynb](https://github.com/Superfawll/ml_performance_playground/blob/main/sharded_matmul_playground.ipynb) · [Open in Colab](https://colab.research.google.com/github/Superfawll/ml_performance_playground/blob/main/sharded_matmul_playground.ipynb)

**What’s there**

- A sequence of matrix multiplications and softmax operations, expressed with `lax.scan`.
- Experiments with replicated arrays, activation sharding, weight sharding, and combined layouts using `NamedSharding` and `PartitionSpec`.
- FSDP-style and tensor-parallel layouts to explore how placement changes execution and communication.
- XProf tracing and compiled HLO inspection, including the resulting collective operations.

The notebook uses four TPU devices in a 2×2 mesh.

**What’s to come**

More advanced performance experiments with transformer MLPs and attention dot products, building on the sharding and profiling experiments here.
