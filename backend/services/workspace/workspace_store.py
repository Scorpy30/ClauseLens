"""
In-memory workspace store.

Workspaces and Conversations are stored by ID in plain dicts.
Design is intentionally storage-agnostic: swap these dicts for a
database client later without changing any other service code.
"""

from backend.models.workspace import Workspace, Conversation

_workspaces: dict[str, Workspace] = {}
_conversations: dict[str, Conversation] = {}


# ── Workspace ──────────────────────────────────────────────────────────────

def save_workspace(workspace: Workspace) -> None:
    _workspaces[workspace.workspace_id] = workspace


def get_workspace(workspace_id: str) -> Workspace | None:
    return _workspaces.get(workspace_id)


def list_workspaces() -> list[Workspace]:
    return list(_workspaces.values())


# ── Conversation ───────────────────────────────────────────────────────────

def save_conversation(conversation: Conversation) -> None:
    _conversations[conversation.conversation_id] = conversation


def get_conversation(conversation_id: str) -> Conversation | None:
    return _conversations.get(conversation_id)


def get_conversations_for_workspace(workspace_id: str) -> list[Conversation]:
    return [
        c for c in _conversations.values()
        if c.workspace_id == workspace_id
    ]
