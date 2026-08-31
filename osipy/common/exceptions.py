"""Custom exceptions for osipy.

This module defines the exception hierarchy for error handling across
all osipy modules. Following Constitution Principle VI, all exceptions
provide explicit, informative error messages.
"""


class OsipyError(Exception):
    """Base exception for all osipy errors.

    All custom exceptions in osipy inherit from this base class,
    allowing users to catch all osipy-specific errors with a
    single except clause.

    Examples
    --------
    >>> try:
    ...     # osipy operations
    ...     pass
    ... except OsipyError as e:
    ...     print(f"osipy error: {e}")
    """

    pass


class DataValidationError(OsipyError):
    """Raised when input data fails validation.

    This exception is raised when:
    - Data has invalid shape or dimensions
    - Data contains unexpected NaN or infinite values
    - Data type is incompatible with expected operations
    - Array shapes are inconsistent

    Examples
    --------
    >>> raise DataValidationError("Data must be 4D, got 3D array")
    Traceback (most recent call last):
        ...
    osipy.common.exceptions.DataValidationError: Data must be 4D, got 3D array
    """

    pass


class FittingError(OsipyError):
    """Raised when model fitting fails.

    This exception is raised when:
    - Fitting algorithm fails to converge
    - All voxels fail fitting (complete failure)
    - Invalid fitting parameters are provided
    - Numerical issues prevent fitting

    Note that individual voxel fitting failures do NOT raise this
    exception; instead, those voxels are marked in the quality mask.
    This exception is reserved for complete/catastrophic failures.

    Examples
    --------
    >>> raise FittingError("Fitting failed to converge after 1000 iterations")
    Traceback (most recent call last):
        ...
    osipy.common.exceptions.FittingError: Fitting failed to converge after 1000 iterations
    """

    pass


class AIFError(OsipyError):
    """Raised when AIF extraction or validation fails.

    This exception is raised when:
    - Automatic AIF detection fails
    - AIF ROI contains insufficient voxels
    - AIF curve has unexpected shape or timing
    - Population AIF parameters are invalid

    Examples
    --------
    >>> raise AIFError("Automatic AIF detection found no suitable voxels")
    Traceback (most recent call last):
        ...
    osipy.common.exceptions.AIFError: Automatic AIF detection found no suitable voxels
    """

    pass


class GPUTransferError(OsipyError):
    """Raised when transferring an array to GPU fails.

    Raised by ``to_gpu()`` when a GPU is available, ``force_cpu`` is not
    set, and the transfer itself fails for any reason (e.g. out of
    memory). Callers that want a graceful CPU fallback instead of
    surfacing this error (e.g. batch processing retrying on CPU after a
    GPU out-of-memory error) must catch it explicitly.

    Examples
    --------
    >>> raise GPUTransferError("GPU transfer failed (CUDA out of memory)")
    Traceback (most recent call last):
        ...
    osipy.common.exceptions.GPUTransferError: GPU transfer failed (CUDA out of memory)
    """

    pass


class IOError(OsipyError):
    """Raised when file I/O operations fail.

    This exception is raised when:
    - File format is not recognized
    - File cannot be read or written
    - BIDS export validation fails
    - Required files are missing

    Examples
    --------
    >>> raise IOError("Unsupported file format: .xyz")
    Traceback (most recent call last):
        ...
    osipy.common.exceptions.IOError: Unsupported file format: .xyz
    """

    pass
