# MLIR Beginner-Friendly Tutorial 2

This is a continuation of my last beginner-friendly tutorial on MLIR, found on GitHub here: https://github.com/AlexandreSinger/mlir-beginner-friendly-tutorial

I wanted to build on my previous tutorial by discussing how MLIR is used in the context of programming hardware accelerators.
Although the first tutorial is not required to follow along with this tutorial, it is recommended that you watch / read the
prior tutorial before this one. The key things from the last tutorial you will need is how to read MLIR's Intermediate Representation
and what multi-level means in the context of MLIR and how different levels are lowered in the MLIR framework.

TODO: Maybe change the order to put torch-mlir first. It fits more into the them of the tutorial and makes serves as a better hook.
        Story: Torch-mlir is an entry point into MLIR, people like to use it, it can be used to target CPUs, but now I want to target an accelerator
                - What does my accelerator look like and how is it usually programmed. It sucks to program in, id rather use PyTorch, but I do not want to lose performance.
                - How can we modify the programming model to make this easier to target from torch-mlir? We can write high-level host-side kernels that do the work for us.
                - Final demo modfies torch-mlir to use CIFace to emit high-level kernels that allow us to target our device easily and get performance.

Idea: Get people warmed up with Torch-MLIR and remind them of the last tutorial, then describe the hw accel architecture.
    - Leave as a question for next week: How can we make use of MLIR's infrastructure to program this hw accelerator from PyTorch.

# Demo 0: TBD

In this repository, you will find a Git submodule of torch-mlir. This was a recent version of torch-mlir that was available when I wrote this tutorial. There is nothing special about it, I just verified that it worked on the code in this tutorial, so I provide it here such that the results of the tutorial will always match in the future. Make sure you have initialized the submodule using:
```
git submodule update --init --recursive
```
This command may take some time since it also initializes the submodules of torch-mlir, which includes LLVM.

Unlike the last tutorial, which used Bash scripts, this tutorial uses Python scripts to lower code through MLIR. Python makes it much more convenient to work with the torch-mlir passes. The following commands will set up a Python virtual environment and install the necessary python libraries to run the scripts in the demos. It is also required to build LLVM correctly in the next step.
```
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

This tutorial uses the Ninja generator to build torch-mlir, this can be installed using:
```
apt-get install ninja-build
```

After initializing the submodules and installing the necessary Python dependencies, we can build torch-mlir using the following commands.
```
cmake -GNinja -Bbuild \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_EXTERNAL_PROJECTS="torch-mlir" \
    -DLLVM_EXTERNAL_TORCH_MLIR_SOURCE_DIR="$(pwd)/external/torch-mlir" \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DTORCH_MLIR_ENABLE_STABLEHLO=ON \
    -DLLVM_TARGETS_TO_BUILD=host \
    external/torch-mlir/externals/llvm-project/llvm

cmake --build build --target tools/torch-mlir/all mlir-opt mlir-translate llc mlir_runner_utils mlir_c_runner_utils
```

Finally, we will need to update our Python Path to point to the torch-mlir python packages. I think this needs to be done for every terminal instance. There are ways to do this permanently with your python virtual environment, but this is beyond the scope of this tutorial (and is annoying). For now, every session that you want to run these demos, run:
```
export PYTHONPATH="$(pwd)/build/tools/torch-mlir/python_packages/torch_mlir:$PYTHONPATH"
```
<!-- TODO: Verify if this is true, or if there is an easy repeatable way to do this. -->

These instructions were graciously donated by Robert Luo from his WaferScapeMapper project: https://github.com/robluo/WaferScapeMapper/tree/main

# Demo 1: Compiling PyTorch Through MLIR Using Torch-MLIR

TODO: Show PyTorch->MLIR->CPU, then try to show targetting a GPU (big feature of MLIR), then ask the question about what to do if I want to target my own hardware accelerator.

# Demo 2: Programming Hardware Accelerators

Although MLIR is an incredibly powerful framework, and can be used to power compiling towards any target, it is not commonly
used to directly target hardware (to emit assembly instructions). For that, prior frameworks or custom compiler flows are
still far more popular. Where MLIR is most valuable is in the context of translating high-level compute graphs (such as
neural networks) to low-level kernels which are then compiled by a lower-level compiler (such as LLVM). As such, for this
demo, I want to motivate why MLIR is used for this purpose by applying it to a hypothetical hardware accelerator.

## 2.1 Hypothetical Hardware Accelerator Architecture

There are many hardware accelerator architectures out there, all with different pros and cons, but often as compiler engineers
we do not get a choice about what the architecture looks like; we leave that to the hardware engineers, although we often beg
them not to make our lives too difficult. So we are going to assume that the hardware accelerator we are going to be targetting
will look something like the figure below:

TODO: Add figure of hardware accelerator.

This hardware accelerator architecture is based on the [Huawei DaVinci AI chip](https://doi.org/10.1016/j.jpdc.2023.01.008) that I used to work with, but I have simplified it slightly for brevity (and obfuscated a few things). The architecture is made of publically available information on the DaVinci chip, where I filled in some gaps using the [Google TPU architecture](https://doi-org.myaccess.library.utoronto.ca/10.1145/3079856.3080246). Frankly, the exact details of the accelerator is not important, I just wanted to keep things realistic for this tutorial.

This accelerator has three compute units: a systolic array for accelerating matrix multiply operations, a vector unit
for accelerating SIMD (Single Instruction Multiple Data) operations, and a scalar unit to handle computing offset and moving
data around. The systolic array is directly connected to three buffers: Buffer A and B (both 256 KB) are for the A and B matrix inputs
and Buffer C (256 KB) for the output. Buffer C is directly connected to the vector unit to optimize applying activation functions after
performing a matrix multiply. The vector and scalar units are connected to a 24 MB UB (Unified Buffer). Connected to the accelerator is an 8
GB L1 cache which holds the input and outputs transferred to and from the host computer. A DMA (Direct Memory Access) controller is
used to move memory between the different buffers (for example moving data from the L1 to the UB, UB to Buffer A, etc.).

The overall system is shown below:

TODO: Add a figure of the system bus and how it connects the host computer to the L2 cache and the hardware accelerator.

This basic system consists of a host computer (imagine you PC or laptop CPU), connected to a large 64 GB L2 cache and the hardware accelerator
through a bus. This bus allows the host computer to "launch kernels" on the hardware accelerator by passing the kernel instructions and kerenl
input data to the accelerator and reading the output data after the kernel completes..
In the context of this tutorial, a kernel is a set of instructions which are executed by the hardware accelerators.

As a worked example of the overall system, suppose the host computer was running a program and the program wanted to accelerate a matrix multiply
between two large matrices. The host would store the matrix operands into the L1 cache of the hardware accelerator and then launch a matrix multiply
kernel on the device. The host would then wait for the kernel to complete and copy the matrix result back into main memory. On the device-side,
the hardware accelerator would perform DMA operations to move the matrix operands from the L1 cache into Buffer A and Buffer B, compute the
matrix multiply, copy the result from Buffer C to the UB, and then finally copy the result from UB to the L1 cache.

For this tutorial, we will largely be ignoring the complixities of how the host CPU and device communicate as well as how
memory is syncronized on the devices. It is far out of scope for this tutorial and should not change the overall design of
our compiler, but do be aware that I am hand-waving some of these details away to keep this tutorial approachable. For more
information, I encourage the reader to look into Heterogeneous Compilation.

## 2.2 Domain-Specific Languages

So, now that we have the architecture details out of the way, lets discuss how we program a system like this. Most often, you will
not be programming these devices (hardware accelerators) using assembly language. The act of programming these devices are so tedious,
that often the designers of the device build an extremely low-level language to communicate with the device. This language is often
built on top of C and use "intrinsic" instructions to perform operations on the device. For example, an intrinsic may exist to perform
a matrix multiply between two matrices or copy data from one buffer to another. This is a step higher than assembly and makes it easier
to program on the device.

This process of making a low-level language to comminicate more directly with a specific hardware-accelerator is sometimes called a
Domain-Specific Language (DSL). An example of a DSL for my example hardware accelerator is shown below:

TODO: Add an example of a 256x256 matrix multiply kernel.

This examples shows a basic 256x256xf32 matrix multiply kernel. It shows...

These DSL are often very simple and close to the hardware, such that the compiler is more easily written and directly translates to
deterministic machine instructions.

## 2.3 Host Kernel Launch Code

Although this is a very important concept in Heterogeneous Compilation, for this tutorial I will be abstracting away how the host
actually launches the kernel on the device. In general, this process requires vendor-provided drivers which allow a host program
to execute a kernel on the device. In the host program, this will look something like a function call which carries with it the
kernel code and pointers to the arguments / results. For this tutorial, I will simply just write this as the following function
call:
```cpp
__HostKernelLaunch(char* kernel_file_path, void** args, void** res, uint32_t config);
```

Where `kernel_file_path` is a path to the binary file of the compiled kernel (to be executed on the device), args is a pointer
to the arguments of the kernel, res is an allocated pointer to where in memory the result should be stored, and config is a
configuration struct that contains information on the size and number of args and results.

## 2.4 Full Host and Device Example

Putting all of this together, we can create a basic example which will compute a matrix multiply on the system.

Let's start with the device code, and call it `matmul_256_256_f32_kernel.c`:
```c
void matmul_256_256_f32_kernel(float* c, float* a, float* b) {
    // Create a pointers to the bottom of Buffers A, B, and C as target addresses.
    // NOTE: In this architecture, we are assuming that we manage our own memory. This is not always the case.
    float* buf_a_ptr = (float *)(0);
    float* buf_b_ptr = (float *)(0);
    float* buf_c_ptr = (float *)(0);

    // Copy 162,144 bytes (256x256 f32 elements) for each input.
    __dma_L1_to_buffer_a(buf_a_ptr, a, 162144);
    __dma_L1_to_buffer_b(buf_b_ptr, b, 162144);

    // Syncronize the memory. Often these operations are asyncronous to allow multiple DMAs to happen in parallel,
    // here we wait for them to finish before starting the matmul.
    __sync_mem();

    // Perform the matrix multiply and store the result in buf_c_ptr.
    // NOTE: The last argument is configuration to tell the systolic array control unit the size and shape of the inputs.
    //       I am far too lazy to make up a fake control language for this, so I am using DEADBEEF for this example.
    __matmul(buf_c_ptr, buf_a_ptr, buf_b_ptr, 0xDEADBEEF);

    // Wait for the systolic array to finish.
    __sync_systolic_array();

    // Copy the result out into L1.
    __dma_buffer_c_to_L1(c, buf_c_ptr, 162144);

    // Syncronize the memory before completing.
    __sync_mem();
}
```
This kernel would then be compiled using a custom compiler to create a binary that can be executed on the device: `matmul_256_256_f32_kernel.bin`.


```c
int main(void) {
    // Allocate the 256x256xf32 input matrices. We assume that they are filled with interesting data.
    float matrix_a[65536] = {...};
    float matrix_b[65536] = {...};
    // Allocate the output matrix.
    float matrix_c[65536] = {0};

    // Launch the kernel.
    float *args[2] = {matrix_a, matrix_b};
    float *res[1] = {matrix_c};
    // NOTE: Similar to the kernel above, the last argument is configuration to tell the OS the size and shape
    //       of args / res. I am just using DEADBEEF as a placeholder to save time.
    __HostKernelLaunch("matmul_256_256_f32_kernel.bin", (void **)args, (void **)res, 0xDEADBEEF);

    // matrix_c now holds the result of the matrix multiply.
    // ... (print the result to the screen).
}
```

NOTE: Often-times the kernel and the host code are written in the same source file, and a specialized heterogeneous compiler will then separate
the kernel from the host code and manage the launches for you (CUDA is a good example of this). For this example, I wanted to keep things as simple
as possible, so I decided to be explicit about this. What you are seeing above is a more written-out example of how this is done.

As shown in the example above, programming for devices like this at this low of a level is often very difficult and requires low-level knowledge
of the system and device. Many users do not like writing programs like this, even though it allows you to achieve the highest performance for the
system. This leads to a very unfortunate trade-off: we need to be at this level to get amazing performance, but users do not want to write at
this low-level. But users also want performance. So, in order for people to use your hardware accelerator, the software stack must be approachable
and yield high-performance; or else they will just buy a GPU instead.

In the following demos, we will discuss how we can still achieve excellent performance while moving up to higher levels of abstraction using PyTorch.

# Demo 3: Picking a Target Level for our Hardware Accelerator

Coming Soon!

<!-- Level 0: Direct hardware intrinsics (no abstraction); Level 1: Higher-level abstraction of hardware; Level 2: Hardware Kernel abstraction; Level 3: Linalg abstraction (dynamic Conv Layer, dynamic FC layer, etc.), Linalg; Level 4: Application abstraction (Pytorch Model). Each level is a dialect which can be translated to a lower level. You choose which level to enter on. Need to show with an example how we can go from level 3 down to level 0. -->

# Demo 4: Modifying Torch-MLIR to Target our Hardware Accelerator

Coming Soon!

<!-- Use Linalg for Level 3. Use CiFace for Level 2 (maybe make simple pass). Assume Level 2 lowering is done in a custom library for simplicity. -->

