"""External model bindings and authenticated runtime adapters."""

from .dsh_native_runtime import (
    DSH_NATIVE_EXECUTION_PROTOCOL,
    DshNativeAgentRuntimeClient,
)

__all__ = ["DSH_NATIVE_EXECUTION_PROTOCOL", "DshNativeAgentRuntimeClient"]
