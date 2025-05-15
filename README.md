<!-- MIT License
  --
  -- Modifications Copyright (C) 2024-2025 Advanced Micro Devices, Inc.
  --
  -- Permission is hereby granted, free of charge, to any person obtaining a copy
  -- of this software and associated documentation files (the "Software"), to deal
  -- in the Software without restriction, including without limitation the rights
  -- to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  -- copies of the Software, and to permit persons to whom the Software is
  -- furnished to do so, subject to the following conditions:
  --
  -- The above copyright notice and this permission notice shall be included in all
  -- copies or substantial portions of the Software.
  --
  -- THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  -- IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  -- FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  -- AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  -- LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  -- OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  -- SOFTWARE.
-->

# HIP Memory Manager for AMD GPUs (hipMM)

> [!CAUTION]
> This release is an *early-access* software technology preview. Running production workloads is *not* recommended.

> [!NOTE]
> This ROCm&trade; port is derived from the NVIDIA RAPIDS&reg; RMM project (**version 23.12**). It aims to
follow the latter's directory structure, file naming and API naming as closely as possible to minimize porting friction for users that are interested in using both projects.

## RMM Resources

- [RMM Reference Documentation](https://docs.rapids.ai/api/rmm/stable/): Python API reference, tutorials, and topic guides.
- [librmm Reference Documentation](https://docs.rapids.ai/api/librmm/stable/): C/C++ library API reference.
- [Getting Started](https://rapids.ai/start.html): Instructions for installing RMM.
- [RAPIDS Community](https://rapids.ai/community.html): Get help, contribute, and collaborate.
- [GitHub repository](https://github.com/rapidsai/rmm): Download the RMM source code.
- [Issue tracker](https://github.com/rapidsai/rmm/issues): Report issues or request features.

## Overview

Achieving optimal performance in GPU-centric workflows frequently requires customizing how host and
device memory are allocated. For example, using "pinned" host memory for asynchronous
host <-> device memory transfers, or using a device memory pool sub-allocator to reduce the cost of
dynamic device memory allocation.

The goal of the HIP Memory Manager (hipMM) is to provide:
- A common interface that allows customizing [device](#device_memory_resource) and
  [host](#host_memory_resource) memory allocation
- A collection of [implementations](#available-resources) of the interface
- A collection of [data structures](#device-data-structures) that use the interface for memory allocation

For information on the interface hipMM provides and how to use hipMM in your C++ code, see
[below](#using-rmm-in-c).

For a walkthrough about the design of the HIP Memory Manager, read [Fast, Flexible Allocation for NVIDIA with RAPIDS Memory Manager](https://developer.nvidia.com/blog/fast-flexible-allocation-for-cuda-with-rapids-memory-manager/) on the NVIDIA Developer Blog.

## Installation

> [!NOTE]
> We support only AMD GPUs. Use the NVIDIA RAPIDS package for NVIDIA GPUs.

> [!NOTE]
> Currently, it is not possible to install hipMM via `conda`.

<!-- ### Conda

hipMM can be installed with Conda ([miniconda](https://conda.io/miniconda.html), or the full
[Anaconda distribution](https://www.anaconda.com/download)) from the `rapidsai` channel:

```bash
# NOTE: Conda installation not supported for hipMM for AMD GPUs.
# conda install -c rapidsai -c conda-forge -c nvidia rmm cuda-version=11.8
```

We also provide [nightly Conda packages](https://anaconda.org/rapidsai-nightly) built from the HEAD
of our latest development branch.

Note: hipMM is supported only on Linux, and only tested with Python versions 3.9 and 3.10.


Note: The hipMM package from Conda requires building with GCC 9 or later. Otherwise, your application may fail to build.

See the [Get RAPIDS version picker](https://rapids.ai/start.html) for more OS and version info. -->

## Building from Source

### Get hipMM Dependencies

Compiler requirements:

* `gcc`                    version 9.3+
* ROCm HIP SDK compilers   version 5.6.0+, recommended is 6.3.0+
* `cmake`                  version 3.26.4+

GPU requirements:

* ROCm HIP SDK 5.6.0+, recommended is 6.3.0+

Python requirements:
* `scikit-build`
* `hip-python`
* `hip-python-as-cuda`
* `cython`

For more details, see [pyproject.toml](python/pyproject.toml)


### Script to build hipMM from source

To install hipMM from source, ensure the dependencies are met and follow the steps below:

- Clone the repository and submodules
```bash
$ git clone --recurse-submodules https://github.com/ROCm-DS/hipMM.git
$ cd hipMM
```

- Create the conda development environment `rmm_dev`
```bash
# create the conda environment (assuming in base `rmm` directory)
$ conda env create --name rmm_dev --file conda/environments/all_rocm_arch-x86_64.yaml
# activate the environment
$ conda activate rmm_dev
```

- Install ROCm dependencies that are not yet distributed via a conda channel. We install HIP Python and the optional Numba HIP dependency via the Github-distributed `numba-hip` package. We select dependencies of Numba HIP that agree with our ROCm installation by providing a parameter `rocm-${ROCM_MAJOR}-${ROCM-MINOR}-${ROCM-PATCH}`
(example: `rocm-6-1-2`) in square brackets:

> [!IMPORTANT]
> Some hipMM dependencies are currently distributed via: https://test.pypi.org/simple
> We need to specify 'https://test.pypi.org/simple' as additional global extra index URL.
> To append the URL and not overwrite what else is specified already, we combine `pip
> config set` and `pip config get` as shown below. We further restore the original URLs.
> (Note that specifying the `--extra-index-url` command line option does not have
> the same effect.)

```bash
(rmm_dev) $ pip install --upgrade pip
(rmm_dev) $ previous_urls=$(pip config get global.extra-index-url)
(rmm_dev) $ pip config set global.extra-index-url "${previous_urls} https://test.pypi.org/simple"
(rmm_dev) $ pip install numba-hip[rocm-${ROCM_MAJOR}-${ROCM-MINOR}-${ROCM-PATCH}]@git+https://github.com/rocm/numba-hip.git
# example: pip install numba-hip[rocm-6-1-2]@git+https://github.com/rocm/numba-hip.git
(rmm_dev) $ pip config set global.extra-index-url "${previous_urls}" # restore urls
```

> [!IMPORTANT]
> When compiling for AMD GPUs, we always need to set the environment variable `CXX` before building so that the Cython build process uses a HIP C++ compiler.
>
> Example:
>
> `(rmm_dev) $ export CXX=hipcc`
>
> We further need to provide the location of the ROCm CMake scripts to CMake via the `CMAKE_PREFIX_PATH` CMake or environment variable. We append via the `:` char to not modify configurations performeed by the active Conda environment.
>
> Example:
>
> `(rmm_dev) $ export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}:/opt/rocm/lib/cmake"`
>

- Build and install `librmm` using cmake & make.

```bash

(rmm_dev) $ export CXX="hipcc"                                # Cython CXX compiler, adjust according to your setup.
(rmm_dev) $ export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}:/opt/rocm/lib/cmake" # ROCm CMake packages
(rmm_dev) $ mkdir build                                       # make a build directory
(rmm_dev) $ cd build                                          # enter the build directory
(rmm_dev) $ cmake .. -DCMAKE_INSTALL_PREFIX=/install/path     # configure cmake ... use $CONDA_PREFIX if you're using Anaconda
(rmm_dev) $ make -j                                           # compile the library librmm.so ... '-j' will start a parallel job using the number of physical cores available on your system
(rmm_dev) $ make install                                      # install the library librmm.so to '/install/path'
```

- Building and installing `librmm` and `rmm` using build.sh. Build.sh creates build dir at root of
  git repository.

```bash

(rmm_dev) $ export CXX="hipcc"                                # Cython CXX compiler, adjust according to your setup.
(rmm_dev) $ export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}:/opt/rocm/lib/cmake" # ROCm CMake packages
(rmm_dev) $ ./build.sh -h                                     # Display help and exit
(rmm_dev) $ ./build.sh -n librmm                              # Build librmm without installing
(rmm_dev) $ ./build.sh -n rmm                                 # Build rmm without installing
(rmm_dev) $ ./build.sh -n librmm rmm                          # Build librmm and rmm without installing
(rmm_dev) $ ./build.sh librmm rmm                             # Build and install librmm and rmm
```

> [!Note]
> Before rebuilding, it is recommended to remove previous build files.
> When you are using the `./build.sh` script, this can be accomplished
> by additionally specifying `clean` (example: `./build.sh clean rmm`).

- To run tests (Optional):
```bash
(rmm_dev) $ cd build (if you are not already in build directory)
$ make test
```

- Build, install, and test the `rmm` python package, in the `python` folder:
```bash
(rmm_dev) $ export CXX="hipcc" # Cython CXX compiler, adjust according to your setup.
(rmm_dev) $ export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}:/opt/rocm/lib/cmake" # ROCm CMake packages
(rmm_dev) $ python setup.py build_ext --inplace
(rmm_dev) $ python setup.py install
(rmm_dev) $ pytest -v
```

- Build the `rmm` python package and create a binary wheel, in the `python` folder:
```bash
(rmm_dev) $ export CXX="hipcc" # Cython CXX compiler, adjust according to your setup.
(rmm_dev) $ export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}:/opt/rocm/lib/cmake" # ROCm CMake packages
(rmm_dev) $ python3 setup.py bdist_wheel
```

Done! You are ready to develop for the hipMM OSS project.

### Installing a hipMM Python wheel

When you install the hipMM-ROCm Python wheel, you can again specify the ROCm version of the dependencies via the optional dependency key `rocm-${ROCM_MAJOR}_${ROCM_MINOR}-${ROCM-PATCH}`. Again, you need to specify an extra `pip` index URL to make it possible for `pip` to find some dependencies.

```bash
$ previous_urls=$(pip config get global.extra-index-url)
$ pip config set global.extra-index-url "${previous_urls} https://test.pypi.org/simple"

$ pip install ${path_to_wheel}.whl[rocm-${ROCM_MAJOR}_${ROCM_MINOR}-${ROCM-PATCH}]
# example: pip install ${path_to_wheel}.whl[rocm-6-1-2]
```

> [!IMPORTANT]
> Each hipMM-ROCm wheel has been built against a particular ROCm version.
> The ROCm dependency key helps you to install hipMM dependencies for this
> particular ROCm version. Using the wheel with an incompatible
> ROCm installation or specifying dependencies that are not compatible
> with the ROCm installation assumed by the hipMM wheel,
> will likely result in issues.

### Caching third-party dependencies

hipMM uses [CPM.cmake](https://github.com/TheLartians/CPM.cmake) to
handle third-party dependencies like spdlog, Thrust, GoogleTest,
GoogleBenchmark. In general you won't have to worry about it. If CMake
finds an appropriate version on your system, it uses it (you can
help it along by setting `CMAKE_PREFIX_PATH` to point to the
installed location). Otherwise those dependencies will be downloaded as
part of the build.

If you frequently start new builds from scratch, consider setting the
environment variable `CPM_SOURCE_CACHE` to an external download
directory to avoid repeated downloads of the third-party dependencies.

## Using hipMM in a downstream CMake project

The installed hipMM library provides a set of config files that makes it easy to
integrate hipMM into your own CMake project. In your `CMakeLists.txt`, just add

```cmake
find_package(rmm [VERSION])
# ...
target_link_libraries(<your-target> (PRIVATE|PUBLIC) rmm::rmm)
```

Since hipMM is a header-only library, this does not actually link hipMM,
but it makes the headers available and pulls in transitive dependencies.
If hipMM is not installed in a default location, use
`CMAKE_PREFIX_PATH` or `rmm_ROOT` to point to its location.

One of hipMM's dependencies is the Thrust library, so the above
automatically pulls in `Thrust` by means of a dependency on the
`rmm::Thrust` target. By default it uses the standard configuration of
Thrust. If you want to customize it, you can set the variables
`THRUST_HOST_SYSTEM` and `THRUST_DEVICE_SYSTEM`; see
[Thrust's CMake documentation](https://github.com/NVIDIA/thrust/blob/main/thrust/cmake/README.md).

# Using hipMM in C++

The first goal of hipMM is to provide a common interface for device and host memory allocation.
This allows both _users_ and _implementers_ of custom allocation logic to program to a single
interface.

To this end, hipMM defines two abstract interface classes:
- [`rmm::mr::device_memory_resource`](#device_memory_resource) for device memory allocation
- [`rmm::mr::host_memory_resource`](#host_memory_resource) for host memory allocation

These classes are based on the
[`std::pmr::memory_resource`](https://en.cppreference.com/w/cpp/memory/memory_resource) interface
class introduced in C++17 for polymorphic memory allocation.

## `device_memory_resource`

`rmm::mr::device_memory_resource` is the base class that defines the interface for allocating and
freeing device memory.

It has two key functions:

1. `void* device_memory_resource::allocate(std::size_t bytes, cuda_stream_view s)`
   - Returns a pointer to an allocation of at least `bytes` bytes.

2. `void device_memory_resource::deallocate(void* p, std::size_t bytes, cuda_stream_view s)`
   - Reclaims a previous allocation of size `bytes` pointed to by `p`.
   - `p` *must* have been returned by a previous call to `allocate(bytes)`, otherwise behavior is
     undefined

It is up to a derived class to provide implementations of these functions. See
[available resources](#available-resources) for example `device_memory_resource` derived classes.

Unlike `std::pmr::memory_resource`, `rmm::mr::device_memory_resource` does not allow specifying an
alignment argument. All allocations are required to be aligned to at least 256B. Furthermore,
`device_memory_resource` adds an additional `cuda_stream_view` argument to allow specifying the stream
on which to perform the (de)allocation.

## `cuda_stream_view` and `cuda_stream`

`rmm::cuda_stream_view` is a simple non-owning wrapper around a `hipStream_t`. This wrapper's
purpose is to provide strong type safety for stream types. (`hipStream_t` is an alias for a pointer,
which can lead to ambiguity in APIs when it is assigned `0`.)  All hipMM stream-ordered APIs take a
`rmm::cuda_stream_view` argument.

`rmm::cuda_stream` is a simple owning wrapper around a `hipStream_t`. This class provides
RAII semantics (constructor creates the stream, destructor destroys it). An `rmm::cuda_stream`
can never represent the default stream or per-thread default stream; it only ever represents
a single non-default stream. `rmm::cuda_stream` cannot be copied, but can be moved.

## `cuda_stream_pool`

`rmm::cuda_stream_pool` provides fast access to a pool of streams. This class can be used to
create a set of `cuda_stream` objects whose lifetime is equal to the `cuda_stream_pool`. Using the
stream pool can be faster than creating the streams on the fly. The size of the pool is configurable.
Depending on this size, multiple calls to `cuda_stream_pool::get_stream()` may return instances of
`rmm::cuda_stream_view` that represent identical streams.

### Thread Safety

All current device memory resources are thread safe unless documented otherwise. More specifically,
calls to memory resource `allocate()` and `deallocate()` methods are safe with respect to calls to
either of these functions from other threads. They are _not_ thread safe with respect to
construction and destruction of the memory resource object.

Note that a class `thread_safe_resource_adapter` is provided which can be used to adapt a memory
resource that is not thread safe to be thread safe (as described above). This adapter is not needed
with any current hipMM device memory resources.

### Stream-ordered Memory Allocation

`rmm::mr::device_memory_resource` is a base class that provides stream-ordered memory allocation.
This allows optimizations such as re-using memory deallocated on the same stream without the
overhead of synchronization.

A call to `device_memory_resource::allocate(bytes, stream_a)` returns a pointer that is valid to use
on `stream_a`. Using the memory on a different stream (say `stream_b`) is Undefined Behavior unless
the two streams are first synchronized, for example by using `hipStreamSynchronize(stream_a)` or by
recording a event on `stream_a` and then calling `hipStreamWaitEvent(stream_b, event)`.

The stream specified to `device_memory_resource::deallocate` should be a stream on which it is valid
to use the deallocated memory immediately for another allocation. Typically this is the stream
on which the allocation was *last* used before the call to `deallocate`. The passed stream may be
used internally by a `device_memory_resource` for managing available memory with minimal
synchronization, and it may also be synchronized at a later time, for example using a call to
`hipStreamSynchronize()`.

For this reason, it is Undefined Behavior to destroy a stream that is passed to
`device_memory_resource::deallocate`. If the stream on which the allocation was last used has been
destroyed before calling `deallocate` or it is known that it will be destroyed, it is likely better
to synchronize the stream (before destroying it) and then pass a different stream to `deallocate`
(e.g. the default stream).

Note that device memory data structures such as `rmm::device_buffer` and `rmm::device_uvector`
follow these stream-ordered memory allocation semantics and rules.

For further information about stream-ordered memory allocation semantics, read
[Using the NVIDIA Stream-Ordered Memory
Allocator](https://developer.nvidia.com/blog/using-cuda-stream-ordered-memory-allocator-part-1/)
on the NVIDIA Developer Blog.

### Available Resources

hipMM provides several `device_memory_resource` derived classes to satisfy various user requirements.
For more detailed information about these resources, see their respective documentation.

#### `cuda_memory_resource`

Allocates and frees device memory using `hipMalloc` and `hipFree`.

#### `managed_memory_resource`

Allocates and frees device memory using `hipMallocManaged` and `hipFree`.

Note that `managed_memory_resource` cannot be used with NVIDIA Virtual GPU Software (vGPU, for use
with virtual machines or hypervisors) because [NVIDIA Unified Memory is not supported by
NVIDIA vGPU](https://docs.nvidia.com/grid/latest/grid-vgpu-user-guide/index.html#cuda-open-cl-support-vgpu).

#### `pool_memory_resource`

A coalescing, best-fit pool sub-allocator.

#### `fixed_size_memory_resource`

A memory resource that can only allocate a single fixed size. Average allocation and deallocation
cost is constant.

#### `binning_memory_resource`

Configurable to use multiple upstream memory resources for allocations that fall within different
bin sizes. Often configured with multiple bins backed by `fixed_size_memory_resource`s and a single
`pool_memory_resource` for allocations larger than the largest bin size.

### Default Resources and Per-device Resources

hipMM users commonly need to configure a `device_memory_resource` object to use for all allocations
where another resource has not explicitly been provided. A common example is configuring a
`pool_memory_resource` to use for all allocations to get fast dynamic allocation.

To enable this use case, hipMM provides the concept of a "default" `device_memory_resource`. This
resource is used when another is not explicitly provided.

Accessing and modifying the default resource is done through two functions:
- `device_memory_resource* get_current_device_resource()`
   - Returns a pointer to the default resource for the current device.
   - The initial default memory resource is an instance of `cuda_memory_resource`.
   - This function is thread safe with respect to concurrent calls to it and
     `set_current_device_resource()`.
   - For more explicit control, you can use `get_per_device_resource()`, which takes a device ID.

- `device_memory_resource* set_current_device_resource(device_memory_resource* new_mr)`
   - Updates the default memory resource pointer for the current device to `new_mr`
   - Returns the previous default resource pointer
   - If `new_mr` is `nullptr`, then resets the default resource to `cuda_memory_resource`
   - This function is thread safe with respect to concurrent calls to it and
     `get_current_device_resource()`
   - For more explicit control, you can use `set_per_device_resource()`, which takes a device ID.

#### Example

```c++
rmm::mr::cuda_memory_resource cuda_mr;
// Construct a resource that uses a coalescing best-fit pool allocator
rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool_mr{&cuda_mr};
rmm::mr::set_current_device_resource(&pool_mr); // Updates the current device resource pointer to `pool_mr`
rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource(); // Points to `pool_mr`
```

#### Multiple Devices

A `device_memory_resource` should only be used when the active device is the same device
that was active when the `device_memory_resource` was created. Otherwise behavior is undefined.

If a `device_memory_resource` is used with a stream associated with a different device than the
device for which the memory resource was created, behavior is undefined.

Creating a `device_memory_resource` for each device requires care to set the current device before
creating each resource, and to maintain the lifetime of the resources as long as they are set as
per-device resources. Here is an example loop that creates `unique_ptr`s to `pool_memory_resource`
objects for each device and sets them as the per-device resource for that device.

```c++
std::vector<unique_ptr<pool_memory_resource>> per_device_pools;
for(int i = 0; i < N; ++i) {
  hipSetDevice(i); // set device i before creating MR
  // Use a vector of unique_ptr to maintain the lifetime of the MRs
  per_device_pools.push_back(std::make_unique<pool_memory_resource>());
  // Set the per-device resource for device i
  set_per_device_resource(cuda_device_id{i}, &per_device_pools.back());
}
```

Note that the device that is current when creating a `device_memory_resource` must also be
current any time that `device_memory_resource` is used to deallocate memory, including in a
destructor. This affects RAII classes like `rmm::device_buffer` and `rmm::device_uvector`. Here's an
(incorrect) example that assumes the above example loop has been run to create a
`pool_memory_resource` for each device. A correct example adds a call to `hipSetDevice(0)` on the
line of the error comment.

```c++
{
  RMM_CUDA_TRY(hipSetDevice(0));
  rmm::device_buffer buf_a(16);

  {
    RMM_CUDA_TRY(hipSetDevice(1));
    rmm::device_buffer buf_b(16);
  }

  // Error: when buf_a is destroyed, the current device must be 0, but it is 1
}
```

### Allocators

C++ interfaces commonly allow customizable memory allocation through an [`Allocator`](https://en.cppreference.com/w/cpp/named_req/Allocator) object.
hipMM provides several `Allocator` and `Allocator`-like classes.

#### `polymorphic_allocator`

A [stream-ordered](#stream-ordered-memory-allocation) allocator similar to [`std::pmr::polymorphic_allocator`](https://en.cppreference.com/w/cpp/memory/polymorphic_allocator).
Unlike the standard C++ `Allocator` interface, the `allocate` and `deallocate` functions take a `cuda_stream_view` indicating the stream on which the (de)allocation occurs.

#### `stream_allocator_adaptor`

`stream_allocator_adaptor` can be used to adapt a stream-ordered allocator to present a standard `Allocator` interface to consumers that may not be designed to work with a stream-ordered interface.

Example:
```c++
rmm::cuda_stream stream;
rmm::mr::polymorphic_allocator<int> stream_alloc;

// Constructs an adaptor that forwards all (de)allocations to `stream_alloc` on `stream`.
auto adapted = rmm::mr::make_stream_allocator_adaptor(stream_alloc, stream);

// Allocates 100 bytes using `stream_alloc` on `stream`
auto p = adapted.allocate(100);
...
// Deallocates using `stream_alloc` on `stream`
adapted.deallocate(p,100);
```

#### `thrust_allocator`

`thrust_allocator` is a device memory allocator that uses the strongly typed `thrust::device_ptr`, making it usable with containers like `thrust::device_vector`.

See [below](#using-rmm-with-thrust) for more information on using hipMM with Thrust.

## Device Data Structures

### `device_buffer`

An untyped, uninitialized RAII class for stream ordered device memory allocation.

#### Example

```c++
cuda_stream_view s{...};
// Allocates at least 100 bytes on stream `s` using the *default* resource
rmm::device_buffer b{100,s};
void* p = b.data();                   // Raw, untyped pointer to underlying device memory

kernel<<<..., s.value()>>>(b.data()); // `b` is only safe to use on `s`

rmm::mr::device_memory_resource * mr = new my_custom_resource{...};
// Allocates at least 100 bytes on stream `s` using the resource `mr`
rmm::device_buffer b2{100, s, mr};
```

### `device_uvector<T>`
A typed, uninitialized RAII class for allocation of a contiguous set of elements in device memory.
Similar to a `thrust::device_vector`, but as an optimization, does not default initialize the
contained elements. This optimization restricts the types `T` to trivially copyable types.

#### Example

```c++
cuda_stream_view s{...};
// Allocates uninitialized storage for 100 `int32_t` elements on stream `s` using the
// default resource
rmm::device_uvector<int32_t> v(100, s);
// Initializes the elements to 0
thrust::uninitialized_fill(thrust::cuda::par.on(s.value()), v.begin(), v.end(), int32_t{0});

rmm::mr::device_memory_resource * mr = new my_custom_resource{...};
// Allocates uninitialized storage for 100 `int32_t` elements on stream `s` using the resource `mr`
rmm::device_uvector<int32_t> v2{100, s, mr};
```

### `device_scalar`
A typed, RAII class for allocation of a single element in device memory.
This is similar to a `device_uvector` with a single element, but provides convenience functions like
modifying the value in device memory from the host, or retrieving the value from device to host.

#### Example
```c++
cuda_stream_view s{...};
// Allocates uninitialized storage for a single `int32_t` in device memory
rmm::device_scalar<int32_t> a{s};
a.set_value(42, s); // Updates the value in device memory to `42` on stream `s`

kernel<<<...,s.value()>>>(a.data()); // Pass raw pointer to underlying element in device memory

int32_t v = a.value(s); // Retrieves the value from device to host on stream `s`
```

## `host_memory_resource`

`rmm::mr::host_memory_resource` is the base class that defines the interface for allocating and
freeing host memory.

Similar to `device_memory_resource`, it has two key functions for (de)allocation:

1. `void* host_memory_resource::allocate(std::size_t bytes, std::size_t alignment)`
   - Returns a pointer to an allocation of at least `bytes` bytes aligned to the specified
     `alignment`

2. `void host_memory_resource::deallocate(void* p, std::size_t bytes, std::size_t alignment)`
   - Reclaims a previous allocation of size `bytes` pointed to by `p`.


Unlike `device_memory_resource`, the `host_memory_resource` interface and behavior is identical to
`std::pmr::memory_resource`.

### Available Resources

#### `new_delete_resource`

Uses the global `operator new` and `operator delete` to allocate host memory.

#### `pinned_memory_resource`

Allocates "pinned" host memory using `cuda(Malloc/Free)Host`.

## Host Data Structures

hipMM does not currently provide any data structures that interface with `host_memory_resource`.
In the future, hipMM will provide a similar host-side structure like `device_buffer` and an allocator
that can be used with STL containers.

## Using hipMM with Thrust

ROCm-DS and other libraries make heavy use of Thrust. Thrust uses device memory in two
situations:

 1. As the backing store for `thrust::device_vector`, and
 2. As temporary storage inside some algorithms, such as `thrust::sort`.

hipMM provides `rmm::mr::thrust_allocator` as a conforming Thrust allocator that uses
`device_memory_resource`s.

### Thrust Algorithms

To instruct a Thrust algorithm to use `rmm::mr::thrust_allocator` to allocate temporary storage, you
can use the custom Thrust device execution policy: `rmm::exec_policy(stream)`.

```c++
thrust::sort(rmm::exec_policy(stream, ...);
```

The first `stream` argument is the `stream` to use for `rmm::mr::thrust_allocator`.
The second `stream` argument is what should be used to execute the Thrust algorithm.
These two arguments must be identical.

## Logging

hipMM includes two forms of logging. Memory event logging and debug logging.

### Memory Event Logging and `logging_resource_adaptor`

Memory event logging writes details of every allocation or deallocation to a CSV (comma-separated
value) file. In C++, Memory Event Logging is enabled by using the `logging_resource_adaptor` as a
wrapper around any other `device_memory_resource` object.

Each row in the log represents either an allocation or a deallocation. The columns of the file are
"Thread, Time, Action, Pointer, Size, Stream".

The CSV output files of the `logging_resource_adaptor` can be used as input to `REPLAY_BENCHMARK`,
which is available when building hipMM from source, in the `gbenchmarks` folder in the build directory.
This log replayer can be useful for profiling and debugging allocator issues.

The following C++ example creates a logging version of a `cuda_memory_resource` that outputs the log
to the file "logs/test1.csv".

```c++
std::string filename{"logs/test1.csv"};
rmm::mr::cuda_memory_resource upstream;
rmm::mr::logging_resource_adaptor<rmm::mr::cuda_memory_resource> log_mr{&upstream, filename};
```

If a file name is not specified, the environment variable `RMM_LOG_FILE` is queried for the file
name. If `RMM_LOG_FILE` is not set, then an exception is thrown by the `logging_resource_adaptor`
constructor.

In Python, memory event logging is enabled when the `logging` parameter of `rmm.reinitialize()` is
set to `True`. The log file name can be set using the `log_file_name` parameter. See
`help(rmm.reinitialize)` for full details.

### Debug Logging

hipMM includes a debug logger which can be enabled to log trace and debug information to a file. This
information can show when errors occur, when additional memory is allocated from upstream resources,
etc. The default log file is `rmm_log.txt` in the current working directory, but the environment
variable `RMM_DEBUG_LOG_FILE` can be set to specify the path and file name.

There is a CMake configuration variable `RMM_LOGGING_LEVEL`, which can be set to enable compilation
of more detailed logging. The default is `INFO`. Available levels are `TRACE`, `DEBUG`, `INFO`,
`WARN`, `ERROR`, `CRITICAL` and `OFF`.

The log relies on the [spdlog](https://github.com/gabime/spdlog.git) library.

Note that to see logging below the `INFO` level, the application must also set the logging level at
run time. C++ applications must must call `rmm::logger().set_level()`, for example to enable all
levels of logging down to `TRACE`, call `rmm::logger().set_level(spdlog::level::trace)` (and compile
librmm with `-DRMM_LOGGING_LEVEL=TRACE`). Python applications must call `rmm.set_logging_level()`,
for example to enable all levels of logging down to `TRACE`, call `rmm.set_logging_level("trace")`
(and compile the hipMM Python module with `-DRMM_LOGGING_LEVEL=TRACE`).

Note that debug logging is different from the CSV memory allocation logging provided by
`rmm::mr::logging_resource_adapter`. The latter is for logging a history of allocation /
deallocation actions which can be useful for replay with hipMM's replay benchmark.

## hipMM and Memory Bounds Checking

Memory allocations taken from a memory resource that allocates a pool of memory (such as
`pool_memory_resource` and `arena_memory_resource`) are part of the same low-level memory
allocation. Therefore, out-of-bounds or misaligned accesses to these allocations are not likely to
be detected by tools such as
[Compute Sanitizer](https://docs.nvidia.com/cuda/compute-sanitizer/index.html) memcheck.

Exceptions to this are `cuda_memory_resource`, which wraps `hipMalloc`, and
`cuda_async_memory_resource`, which uses `hipMallocAsync` with the device runtime's built-in memory pool
functionality (11.2 or later required). Illegal memory accesses to memory allocated by these
resources are detectable with Compute Sanitizer Memcheck.

It may be possible in the future to add support for memory bounds checking with other memory
resources using NVTX APIs.

## Using hipMM in Python Code

There are two ways to use hipMM in Python code:

1. Using the `rmm.DeviceBuffer` API to explicitly create and manage
   device memory allocations
2. Transparently via external libraries such as CuPy and Numba

hipMM provides a `MemoryResource` abstraction to control _how_ device
memory is allocated in both the above uses.

### DeviceBuffers

A DeviceBuffer represents an **untyped, uninitialized device memory
allocation**.  DeviceBuffers can be created by providing the
size of the allocation in bytes:

```python
>>> import rmm
>>> buf = rmm.DeviceBuffer(size=100)
```

The size of the allocation and the memory address associated with it
can be accessed via the `.size` and `.ptr` attributes respectively:

```python
>>> buf.size
100
>>> buf.ptr
140202544726016
```

DeviceBuffers can also be created by copying data from host memory:

```python
>>> import rmm
>>> import numpy as np
>>> a = np.array([1, 2, 3], dtype='float64')
>>> buf = rmm.DeviceBuffer.to_device(a.tobytes())
>>> buf.size
24
```

Conversely, the data underlying a DeviceBuffer can be copied to the
host:

```python
>>> np.frombuffer(buf.tobytes())
array([1., 2., 3.])
```

### MemoryResource objects

`MemoryResource` objects are used to configure how device memory allocations are made by
hipMM.

By default if a `MemoryResource` is not set explicitly, hipMM uses the `CudaMemoryResource`, which
uses `hipMalloc` for allocating device memory.

`rmm.reinitialize()` provides an easy way to initialize hipMM with specific memory resource options
across multiple devices. See `help(rmm.reinitialize)` for full details.

For lower-level control, the `rmm.mr.set_current_device_resource()` function can be
used to set a different MemoryResource for the current device.  For
example, enabling the `ManagedMemoryResource` tells hipMM to use
`hipMallocManaged` instead of `hipMalloc` for allocating memory:

```python
>>> import rmm
>>> rmm.mr.set_current_device_resource(rmm.mr.ManagedMemoryResource())
```

> :warning: The default resource must be set for any device **before**
> allocating any device memory on that device.  Setting or changing the
> resource after device allocations have been made can lead to unexpected
> behaviour or crashes. See [Multiple Devices](#multiple-devices)

As another example, `PoolMemoryResource` allows you to allocate a
large "pool" of device memory up-front. Subsequent allocations will
draw from this pool of already allocated memory.  The example
below shows how to construct a PoolMemoryResource with an initial size
of 1 GiB and a maximum size of 4 GiB. The pool uses
`CudaMemoryResource` as its underlying ("upstream") memory resource:

```python
>>> import rmm
>>> pool = rmm.mr.PoolMemoryResource(
...     rmm.mr.CudaMemoryResource(),
...     initial_pool_size=2**30,
...     maximum_pool_size=2**32
... )
>>> rmm.mr.set_current_device_resource(pool)
```
Other MemoryResources include:

* `FixedSizeMemoryResource` for allocating fixed blocks of memory
* `BinningMemoryResource` for allocating blocks within specified "bin" sizes from different memory
resources

MemoryResources are highly configurable and can be composed together in different ways.
See `help(rmm.mr)` for more information.

## Using hipMM with third-party libraries

> [!WARNING]
> The contents of this section have not been tested explicitly with **hipMM**.

### Using hipMM with CuPy

You can configure [CuPy](https://cupy.dev/) to use hipMM for memory
allocations by setting the CuPy allocator to
`rmm_cupy_allocator`:

```python
>>> from rmm.allocators.cupy import rmm_cupy_allocator
>>> import cupy
>>> cupy.cuda.set_allocator(rmm_cupy_allocator)
```


**Note:** This only configures CuPy to use the current hipMM resource for allocations.
It does not initialize nor change the current resource, e.g., enabling a memory pool.
See [here](#memoryresource-objects) for more information on changing the current memory resource.

### Using hipMM with Numba

You can configure Numba to use hipMM for memory allocations using the
Numba [EMM Plugin](https://numba.readthedocs.io/en/stable/cuda/external-memory.html#setting-emm-plugin).

This can be done in two ways:

1. Setting the environment variable `NUMBA_CUDA_MEMORY_MANAGER`:

  ```python
  $ NUMBA_CUDA_MEMORY_MANAGER=rmm.allocators.numba python (args)
  ```

2. Using the `set_memory_manager()` function provided by Numba:

  ```python
  >>> from numba import cuda
  >>> from rmm.allocators.numba import RMMNumbaManager
  >>> cuda.set_memory_manager(RMMNumbaManager)
  ```

**Note:** This only configures Numba to use the current hipMM resource for allocations.
It does not initialize nor change the current resource, e.g., enabling a memory pool.
See [here](#memoryresource-objects) for more information on changing the current memory resource.

### Using hipMM with PyTorch

[PyTorch](https://pytorch.org/docs/stable/notes/cuda.html) can use hipMM
for memory allocation.  For example, to configure PyTorch to use an
hipMM-managed pool:

```python
import rmm
from rmm.allocators.torch import rmm_torch_allocator
import torch

rmm.reinitialize(pool_allocator=True)
torch.cuda.memory.change_current_allocator(rmm_torch_allocator)
```

PyTorch and hipMM will now share the same memory pool.

You can, of course, use a custom memory resource with PyTorch as well:

```python
import rmm
from rmm.allocators.torch import rmm_torch_allocator
import torch

# note that you can configure PyTorch to use hipMM either before or
# after changing hipMM's memory resource.  PyTorch will use whatever
# memory resource is configured to be the "current" memory resource at
# the time of allocation.
torch.cuda.change_current_allocator(rmm_torch_allocator)

# configure hipMM to use a managed memory resource, wrapped with a
# statistics resource adaptor that can report information about the
# amount of memory allocated:
mr = rmm.mr.StatisticsResourceAdaptor(rmm.mr.ManagedMemoryResource())
rmm.mr.set_current_device_resource(mr)

x = torch.tensor([1, 2]).cuda()

# the memory resource reports information about PyTorch allocations:
mr.allocation_counts
Out[6]:
{'current_bytes': 16,
 'current_count': 1,
 'peak_bytes': 16,
 'peak_count': 1,
 'total_bytes': 16,
 'total_count': 1}
```
