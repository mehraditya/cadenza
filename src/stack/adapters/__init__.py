"""World-model adapters — base ABC + registry only.

Cadenza ships no concrete adapters. Define yours in your project (e.g. an
``ai_models/`` folder) by subclassing ``WorldModelAdapter`` and decorating
with ``@register_adapter`` (or pass the instance directly via
``go1.setup(model=...)``).
"""

from cadenza.stack.adapters.base import (
    WorldModelAdapter,
    AdapterReply,
    ProposedAction,
    register_adapter,
    get_adapter,
    list_adapters,
)
from cadenza.stack.adapters.mock import MockAdapter

__all__ = [
    "WorldModelAdapter",
    "AdapterReply",
    "ProposedAction",
    "register_adapter",
    "get_adapter",
    "list_adapters",
    "MockAdapter",
]
