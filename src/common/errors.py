"""Unwraps the ExceptionGroup that anyio raises when an MCP task group fails."""


def root_cause(exc: BaseException) -> str:
    """Returns 'TypeName: message' for the innermost exception in a nested group."""
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return f"{type(exc).__name__}: {exc}"
