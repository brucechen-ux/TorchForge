"""Model patch examples that wire common components without copying them."""

from .deepseek_v3_mla import DSV3MLAPatch
from .deepseek_v4_mla import DSV4MLAPatch

__all__ = ["DSV3MLAPatch", "DSV4MLAPatch"]

