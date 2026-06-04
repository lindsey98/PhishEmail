"""Reference-DB package.

Heavy submodules (``IdentityMatcher``/``CharacterBert``, which pull in torch,
faiss and transformers) are imported lazily via :pep:`562` ``__getattr__`` so that
the dependency-light agent submodules (``agent_constants``/``agent_helpers``) can
be imported — and unit-tested — without the full ML stack.
"""

from typing import TYPE_CHECKING

__all__ = [
    "BaseFaissIPRetriever",
    "CharacterBERT",
    "CharacterBertModel",
    "IdentityMatcher",
]

if TYPE_CHECKING:  # for type checkers / IDEs only; not executed at runtime
    from .CharacterBert import CharacterBertModel
    from .IdentityMatcher import BaseFaissIPRetriever, CharacterBERT, IdentityMatcher


def __getattr__(name):
    if name == "CharacterBertModel":
        from .CharacterBert import CharacterBertModel

        return CharacterBertModel
    if name in {"IdentityMatcher", "BaseFaissIPRetriever", "CharacterBERT"}:
        from .IdentityMatcher import BaseFaissIPRetriever, CharacterBERT, IdentityMatcher

        return {
            "IdentityMatcher": IdentityMatcher,
            "BaseFaissIPRetriever": BaseFaissIPRetriever,
            "CharacterBERT": CharacterBERT,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
