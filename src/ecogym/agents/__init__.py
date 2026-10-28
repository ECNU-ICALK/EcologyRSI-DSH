"""LLM research agents: chat transports, research tools and the ``llm`` proposer.

Importing this package registers the ``llm`` proposer with the arena.
"""

from ..arena.proposers import register_proposer
from .clients import AgentError, ChatClient, DshHeadlessClient, OpenAICompatibleClient, Reply, make_client
from .roles import LLMProposer
from .tools import ResearchTools

register_proposer("llm", LLMProposer)

__all__ = [
    "AgentError",
    "ChatClient",
    "DshHeadlessClient",
    "LLMProposer",
    "OpenAICompatibleClient",
    "Reply",
    "ResearchTools",
    "make_client",
]
