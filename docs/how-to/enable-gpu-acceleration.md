# How to Configure GPU/CPU Backend

Set up GPU acceleration or force CPU execution.

## Prerequisites

GPU acceleration requires:

1. NVIDIA GPU with CUDA support
2. CUDA toolkit installed
3. CuPy library installed

## Install CuPy

Install CuPy matching your CUDA version:

!!! example "Install CuPy for GPU support"

    ```bash
    # For CUDA 12.x
    pip install cupy-cuda12x

    # For CUDA 11.x
    pip install cupy-cuda11x

    # Or install with osipy GPU extras
    pip install osipy[gpu]
    ```

## Verify GPU Availability

Check if GPU is available:

!!! example "Check GPU availability and info"

    ```python
    import osipy

    # Check GPU status
    print(f"GPU available: {osipy.is_gpu_available()}")
    print(f"Current backend: {osipy.get_backend()}")

    # Detailed GPU info
    if osipy.is_gpu_available():
        import cupy as cp
        device = cp.cuda.Device()
        print(f"GPU: {device.compute_capability}")
        print(f"Memory: {device.mem_info[1] / 1e9:.1f} GB")
    ```

## Automatic GPU Usage

When GPU is available, osipy uses it automatically:

!!! example "Automatic GPU usage with numpy input"

    ```python
    import numpy as np
    import osipy

    # Data on CPU (numpy array)
    concentration = np.random.rand(64, 64, 32, 60)
    time = np.linspace(0, 300, 60)
    aif = osipy.ParkerAIF()(time)

    # Fitting automatically uses GPU if available
    result = osipy.dce_fit_model("extended_tofts", concentration, aif, time)

    # Result is returned as numpy (CPU) array
    print(f"Result type: {type(result.parameter_maps['Ktrans'].values)}")  # numpy.ndarray
    ```

## Explicit GPU Arrays

For manual control, use CuPy arrays:

!!! example "Explicit GPU arrays with CuPy"

    ```python
    import cupy as cp
    import osipy

    # Move data to GPU
    concentration_gpu = cp.asarray(concentration)
    time_gpu = cp.asarray(time)

    # Process on GPU
    result = osipy.dce_fit_model("extended_tofts", concentration_gpu, aif, time_gpu)

    # Result stays on GPU (CuPy array)
    print(f"Result type: {type(result.parameter_maps['Ktrans'].values)}")  # cupy.ndarray

    # Move back to CPU when needed
    ktrans_cpu = osipy.to_numpy(result.parameter_maps['Ktrans'].values)
    ```

## Memory Management

Monitor and manage GPU memory:

!!! example "Monitor and free GPU memory"

    ```python
    import cupy as cp

    # Check available memory
    mempool = cp.get_default_memory_pool()
    print(f"Used: {mempool.used_bytes() / 1e9:.2f} GB")
    print(f"Total: {mempool.total_bytes() / 1e9:.2f} GB")

    # Free unused memory
    mempool.free_all_blocks()
    ```

## Process Large Datasets

For datasets larger than GPU memory, process in chunks:

!!! example "Process large datasets in chunks"

    ```python
    import numpy as np
    import osipy

    def fit_chunked(concentration, aif, time, chunk_size=10000):
        """Fit model in chunks to manage GPU memory."""
        # Reshape to (n_voxels, n_timepoints)
        shape = concentration.shape
        data_2d = concentration.reshape(-1, shape[-1])
        n_voxels = data_2d.shape[0]

        # Initialize results
        results = {
            'Ktrans': np.zeros(n_voxels),
            've': np.zeros(n_voxels),
            'vp': np.zeros(n_voxels),
            'r_squared': np.zeros(n_voxels),
        }

        # Process in chunks
        for start in range(0, n_voxels, chunk_size):
            end = min(start + chunk_size, n_voxels)
            chunk = data_2d[start:end]

            # Fit chunk (will use GPU)
            chunk_result = osipy.dce_fit_model(
                "extended_tofts",
                chunk[..., np.newaxis, np.newaxis, :].transpose(1, 2, 0, 3),
                aif, time
            )

            # Store results
            results['Ktrans'][start:end] = chunk_result.parameter_maps['Ktrans'].values.flatten()
            results['ve'][start:end] = chunk_result.parameter_maps['ve'].values.flatten()
            results['vp'][start:end] = chunk_result.parameter_maps['vp'].values.flatten()
            results['r_squared'][start:end] = chunk_result.r_squared_map.flatten()

        # Reshape back to 3D
        for key in results:
            results[key] = results[key].reshape(shape[:-1])

        return results

    # Use for large datasets
    result = fit_chunked(concentration, aif, time)
    ```

## Configure GPU Settings

Configure GPU behavior:

!!! example "Configure GPU settings"

    ```python
    import osipy

    # Configure GPU settings
    config = osipy.GPUConfig(
        device_id=0,                # GPU device to use
        force_cpu=False,            # Don't force CPU
        memory_limit_fraction=0.9   # Use up to 90% of available GPU memory
    )

    osipy.set_backend(config)
    ```

## Multi-GPU Processing

Use specific GPU devices:

!!! example "Multi-GPU device selection"

    ```python
    import cupy as cp

    # Select GPU device
    with cp.cuda.Device(0):
        # Processing uses GPU 0
        result_0 = osipy.dce_fit_model("extended_tofts", data_0, aif, time)

    with cp.cuda.Device(1):
        # Processing uses GPU 1
        result_1 = osipy.dce_fit_model("extended_tofts", data_1, aif, time)
    ```

## Performance Tips

### 1. Batch Your Data

!!! example "Process entire volume at once"

    ```python
    # Process entire volume at once (better GPU utilization)
    result = osipy.dce_fit_model("extended_tofts", concentration, aif, time)

    # NOT: loop over slices (inefficient)
    # for z in range(n_slices):
    #     result[z] = osipy.dce_fit_model("extended_tofts", concentration[:,:,z,:], aif, time)
    ```

### 2. Use Appropriate Data Types

!!! example "Use float32 for faster computation"

    ```python
    # float32 is faster and uses less memory
    concentration_f32 = concentration.astype(np.float32)
    ```

### 3. Pre-allocate Arrays

!!! example "Pre-allocate GPU arrays"

    ```python
    import cupy as cp

    # Pre-allocate output arrays on GPU
    ktrans = cp.zeros(shape[:3], dtype=cp.float32)
    ve = cp.zeros(shape[:3], dtype=cp.float32)
    ```

## Troubleshooting

### CUDA Out of Memory

!!! example "Free GPU memory and retry"

    ```python
    # Free memory and retry
    import cupy as cp
    cp.get_default_memory_pool().free_all_blocks()

    # Or process in smaller chunks
    result = fit_chunked(concentration, aif, time, chunk_size=5000)
    ```

### CuPy Import Error

!!! example "Fix CuPy installation"

    ```bash
    # Check CUDA installation
    nvidia-smi

    # Reinstall CuPy for correct CUDA version
    pip uninstall cupy-cuda12x
    pip install cupy-cuda11x  # if CUDA 11
    ```

### Slow First Run

!!! example "Warmup CuPy kernel compilation"

    ```python
    # CuPy compiles kernels on first use
    # Run a small warmup:
    warmup = np.random.rand(8, 8, 8, 10)
    _ = osipy.dce_fit_model("extended_tofts", warmup, aif, time[:10])

    # Subsequent runs will be faster
    ```

## Force CPU Execution

Force CPU even when GPU is available -- useful for debugging, reproducibility testing,
or when GPU memory is insufficient.

### Global CPU Mode

!!! example "Force CPU execution globally"

    ```python
    import osipy

    osipy.set_backend(osipy.GPUConfig(force_cpu=True))
    backend = osipy.get_backend()  # Returns a GPUConfig object, not a string
    print(f"Backend: {backend}")
    print(f"Force CPU: {backend.force_cpu}")  # True
    ```

### Environment Variable

!!! example "Force CPU via environment variable"

    ```bash
    export OSIPY_FORCE_CPU=1
    python your_script.py
    ```

### Compare CPU vs GPU Results

!!! example "Compare CPU and GPU results"

    ```python
    import numpy as np
    import osipy

    osipy.set_backend(osipy.GPUConfig(force_cpu=True))
    result_cpu = osipy.dce_fit_model("extended_tofts", concentration, aif, time)

    osipy.set_backend(osipy.GPUConfig(force_cpu=False))
    result_gpu = osipy.dce_fit_model("extended_tofts", concentration, aif, time)

    ktrans_cpu = osipy.to_numpy(result_cpu.parameter_maps['Ktrans'].values)
    ktrans_gpu = osipy.to_numpy(result_gpu.parameter_maps['Ktrans'].values)
    print(f"Max difference: {np.abs(ktrans_cpu - ktrans_gpu).max()}")
    ```

### Reset to Default

!!! example "Reset backend to default"

    ```python
    osipy.set_backend(osipy.GPUConfig(force_cpu=False))
    ```

## See Also

- [The xp Abstraction Pattern](../explanation/xp-abstraction.md)
