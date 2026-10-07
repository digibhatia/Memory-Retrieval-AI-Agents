"""
memory_store.py — conversation memory, treated as an operational
component with its own failure mode, not as "the AI remembering things."

Memory here is nothing magical: it's a Python dictionary, keyed by
session_id, holding a list of {role, content} turns. That's it. Treating
it as an operational component means asking the same questions you'd ask
of a cache or a database: what happens when it's empty? What happens when
it overflows? Who can inspect it? Who can clear it?

THE FAILURE MODE THIS FILE DEMONSTRATES: a conversation has no limit
unless you impose one. Left unbounded, every turn gets resent to the
model on every single call, so cost and latency climb with every message
— quietly, with no error, no warning, right up until the context window
itself overflows. truncate() is the fix, and it is also where "memory"
genuinely forgets things, on purpose, the same way a human skimming back
through a long thread does.
"""

from __future__ import annotations

from dataclasses import dataclass, field

MAX_TURNS_DEFAULT = 8  # turns kept per session; each user+assistant exchange is 2 turns


@dataclass
class Session:
    session_id: str
    turns: list[dict] = field(default_factory=list)   # [{"role": "user"/"assistant", "content": "..."}]
    truncated_count: int = 0                            # how many turns have been dropped, lifetime

    def append(self, role: str, content: str) -> None:
        self.turns.append({"role": role, "content": content})

    def truncate(self, max_turns: int = MAX_TURNS_DEFAULT) -> int:
        """Drop the oldest turns beyond max_turns. Returns how many were dropped just now."""
        overflow = len(self.turns) - max_turns
        if overflow <= 0:
            return 0
        self.turns = self.turns[overflow:]
        self.truncated_count += overflow
        return overflow


class MemoryStore:
    """
    An in-process store, one Session per session_id. Restarting app.py
    clears everything — there is nothing durable here, which is itself
    worth noticing: "memory" in a single-process demo is not the same
    operational guarantee as memory backed by a real database.
    """

    def __init__(self):
        self._sessions: dict[str, Session] = {}

    def get_or_create(self, session_id: str) -> Session:
        if session_id not in self._sessions:
            self._sessions[session_id] = Session(session_id=session_id)
        return self._sessions[session_id]

    def inspect(self, session_id: str) -> dict | None:
        """Read-only view of a session's current state — no side effects."""
        s = self._sessions.get(session_id)
        if s is None:
            return None
        return {
            "session_id": s.session_id,
            "turn_count": len(s.turns),
            "turns": s.turns,
            "truncated_count_lifetime": s.truncated_count,
        }

    def reset(self, session_id: str) -> bool:
        existed = session_id in self._sessions
        self._sessions.pop(session_id, None)
        return existed

    def list_sessions(self) -> list[str]:
        return list(self._sessions.keys())


# One shared instance for app.py's whole process — this IS the "memory," and it's
# nothing more than this object. Printing it, inspecting it, or resetting it is
# exactly as mundane as doing the same to any other in-process Python object.
store = MemoryStore()


if __name__ == "__main__":
    s = store.get_or_create("demo-session")
    for i in range(12):
        s.append("user" if i % 2 == 0 else "assistant", f"turn {i}")
    print(f"Before truncate: {len(s.turns)} turns")
    dropped = s.truncate(max_turns=8)
    print(f"Dropped {dropped} oldest turns. Now: {len(s.turns)} turns, "
          f"{s.truncated_count} dropped lifetime.")
    print("Oldest remaining turn:", s.turns[0])
