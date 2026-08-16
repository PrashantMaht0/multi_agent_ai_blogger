"""
src/agents/errors.py
Unwraps the ExceptionGroup that anyio raises when an MCP task group fails.

Its str() is always "unhandled errors in a TaskGroup (N sub-exceptions)", which says
nothing about the actual cause - the useful error is nested several groups deep.
"""


def root_cause(exc: BaseException) -> str:
    """Returns 'TypeName: message' for the innermost exception in a (nested) group."""
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return f"{type(exc).__name__}: {exc}"
