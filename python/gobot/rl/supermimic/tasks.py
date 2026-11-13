"""Backend-neutral goal encoder dimensions; motion extraction stays with the task."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EncoderSpec:
    name: str
    goal_dim: int

    def __post_init__(self):
        if not self.name or "." in self.name:
            raise ValueError("encoder name must be nonempty and contain no periods")
        if (
            isinstance(self.goal_dim, bool)
            or not isinstance(self.goal_dim, int)
            or self.goal_dim < 1
        ):
            raise ValueError("goal_dim must be a positive integer")
