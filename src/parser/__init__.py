"""cadenza.parser — text/decision -> action library bridges.

Two backends live here:

  * :class:`CommandParser` — regex-and-aliases parser for human-typed
    commands. Used by the CLI and ``sim`` for things like
    ``"walk forward 2m then jump"``.
  * :mod:`cadenza.parser.lora_action_head` — LoRA-adapted neural head for
    bridging an AI model's decisions into action-library sequences.
    Used by the VLA adapters.
"""

from cadenza.parser.translator import CommandParser

__all__ = ["CommandParser"]
