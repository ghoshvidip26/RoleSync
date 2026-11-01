from abc import ABC,abstractmethod
from typing import Any

class BaseResumeRenderer(ABC):
    @abstractmethod
    def render(
        self,
        profile: dict[str,Any],
        draft: dict[str,Any],
        layout_metadata: dict[str,Any]
    ) -> bytes:
        pass