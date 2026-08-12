from __future__ import annotations

import json
import logging
import os
import re
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX platforms
    fcntl = None

try:
    import msvcrt
except ImportError:  # pragma: no cover - non-Windows platforms
    msvcrt = None


STALE_KEYS = ["story", "characters", "script", "storyboard", "shot_descriptions", "camera_tree", "frames", "clips", "final_video"]


def _synchronized(method):
    """Hold the index file lock across a read-modify-write cycle.

    Every mutator loads the whole sessions file, edits it, and saves it back;
    without a lock, two concurrent writers (threads or processes) silently
    drop each other's updates.
    """

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._locked():
            return method(self, *args, **kwargs)

    return wrapper


class SessionIndex:
    def __init__(self, workspace_root: str | Path) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.vimax_dir = self.workspace_root / ".vimax"
        self.sessions_path = self.vimax_dir / "sessions.json"
        self.memory_path = self.vimax_dir / "memory.md"
        self.logs_dir = self.vimax_dir / "logs"
        self.working_root = self.workspace_root / ".working_dir"
        self.vimax_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.working_root.mkdir(parents=True, exist_ok=True)
        if not self.memory_path.exists():
            self.memory_path.write_text("# User Preferences\n", encoding="utf-8")
        if not self.sessions_path.exists():
            self.save({"active_session_id": "", "sessions": {}})
        if msvcrt is not None:
            # msvcrt.locking locks a byte range, so the file needs at least
            # one byte before it can be locked. Written once here, outside
            # any concurrent access, since Windows enforces this lock as
            # mandatory: a bare read() on the lock file while another thread
            # holds it would itself raise PermissionError.
            lock_path = self.vimax_dir / "sessions.lock"
            if not lock_path.exists():
                lock_path.write_bytes(b"\0")

    @contextmanager
    def _locked(self):
        if fcntl is None and msvcrt is None:
            yield
            return
        lock_path = self.vimax_dir / "sessions.lock"
        with open(lock_path, "a+b") as handle:
            if fcntl is not None:
                fcntl.flock(handle, fcntl.LOCK_EX)
            else:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(handle, fcntl.LOCK_UN)
                else:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)

    def load(self) -> dict[str, Any]:
        try:
            return json.loads(self.sessions_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {"active_session_id": "", "sessions": {}}
        except json.JSONDecodeError:
            # A corrupt file usually means a crash mid-write. Returning empty
            # state is fine for this call, but the next save() would overwrite
            # the file and destroy every session — keep the evidence first.
            backup = self.sessions_path.with_name(f"sessions.json.corrupt-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}")
            try:
                os.replace(self.sessions_path, backup)
                logging.error("sessions.json was corrupt; preserved it at %s and starting with empty state", backup)
            except OSError:
                logging.error("sessions.json is corrupt and could not be backed up; starting with empty state")
            return {"active_session_id": "", "sessions": {}}

    def save(self, data: dict[str, Any]) -> None:
        tmp_path = self.sessions_path.with_name("sessions.json.tmp")
        tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_path, self.sessions_path)

    def active(self) -> dict[str, Any] | None:
        data = self.load()
        session_id = str(data.get("active_session_id", ""))
        if not session_id:
            return None
        record = data.get("sessions", {}).get(session_id)
        return self._with_session_defaults(record) if isinstance(record, dict) else None

    def get(self, session_id: str) -> dict[str, Any] | None:
        normalized = self._normalize_session_id(session_id)
        record = self.load().get("sessions", {}).get(normalized)
        return self._with_session_defaults(record) if isinstance(record, dict) else None

    @_synchronized
    def create(
        self,
        idea: str = "",
        user_requirement: str = "",
        style: str = "",
        session_id: str | None = None,
        project_name: str = "",
    ) -> dict[str, Any]:
        data = self.load()
        sessions = data.setdefault("sessions", {})
        clean_project_name = str(project_name or "").strip()[:64]
        final_id = self._normalize_session_id(session_id) if session_id else self._new_session_id(clean_project_name or idea or user_requirement or "vimax", sessions)
        if final_id in sessions:
            final_id = self._dedupe_session_id(final_id, sessions)
        now = datetime.now().isoformat(timespec="seconds")
        working_dir = self._working_dir_for_id(final_id)
        (working_dir / "idea2video").mkdir(parents=True, exist_ok=True)
        (working_dir / "script2video").mkdir(parents=True, exist_ok=True)
        record = {
            "session_id": final_id,
            "project_name": clean_project_name,
            "working_dir": str(working_dir.relative_to(self.workspace_root)),
            "idea": idea,
            "user_requirement": user_requirement,
            "style": style,
            "stage": "created",
            "summary": "",
            "stale": {key: False for key in STALE_KEYS},
            "recent_turn_records": [],
            "compacted_summary": "",
            "compacted_turns": 0,
            "compaction_snapshots": [],
            "last_compaction_reason": "",
            "last_compaction_at": "",
            "created_at": now,
            "updated_at": now,
        }
        sessions[final_id] = record
        data["active_session_id"] = final_id
        self.save(data)
        return record

    def get_or_create_active(self, idea: str = "", user_requirement: str = "", style: str = "") -> dict[str, Any]:
        active = self.active()
        if active is not None:
            return active
        return self.create(idea=idea, user_requirement=user_requirement, style=style)

    @_synchronized
    def set_active(self, session_id: str) -> dict[str, Any]:
        normalized = self._normalize_session_id(session_id)
        data = self.load()
        if normalized not in data.get("sessions", {}):
            raise KeyError(f"Unknown session_id: {session_id}")
        data["active_session_id"] = normalized
        self.save(data)
        return dict(data["sessions"][normalized])

    @_synchronized
    def update_stage(self, session_id: str, stage: str, summary: str = "") -> None:
        data = self.load()
        record = data.get("sessions", {}).get(session_id)
        if not isinstance(record, dict):
            raise KeyError(f"Unknown session_id: {session_id}")
        record["stage"] = stage
        if summary:
            record["summary"] = summary
        record["updated_at"] = datetime.now().isoformat(timespec="seconds")
        self.save(data)

    @_synchronized
    def mark_stale(self, session_id: str, keys: list[str]) -> None:
        data = self.load()
        record = data.get("sessions", {}).get(session_id)
        if not isinstance(record, dict):
            raise KeyError(f"Unknown session_id: {session_id}")
        stale = record.setdefault("stale", {key: False for key in STALE_KEYS})
        for key in keys:
            stale[key] = True
        record["updated_at"] = datetime.now().isoformat(timespec="seconds")
        self.save(data)

    @_synchronized
    def update_compaction(self, session_id: str, result: dict[str, Any]) -> None:
        data = self.load()
        session = data.get("sessions", {}).get(session_id)
        if not isinstance(session, dict):
            raise KeyError(f"Unknown session_id: {session_id}")
        summary = str(result.get("summary", "") or "")
        compacted_count = int(result.get("compacted_message_count", 0) or 0)
        snapshot = {
            "level": len(session.get("compaction_snapshots", []) or []) + 1,
            "reason": str(result.get("reason", "manual") or "manual"),
            "mode": str(result.get("mode", "unknown") or "unknown"),
            "summary": summary,
            "preserved_messages": int(result.get("preserved_message_count", 0) or 0),
            "compacted_message_count": compacted_count,
            "estimated_tokens_before": int(result.get("estimated_tokens_before", 0) or 0),
            "estimated_tokens_after": int(result.get("estimated_tokens_after", 0) or 0),
            "created_at": str(result.get("created_at", "") or datetime.now().isoformat(timespec="seconds")),
        }
        session["compacted_summary"] = summary
        session["compacted_turns"] = int(session.get("compacted_turns", 0) or 0) + max(1, compacted_count // 2)
        snapshots = list(session.get("compaction_snapshots", []) or [])
        snapshots.append(snapshot)
        session["compaction_snapshots"] = snapshots[-8:]
        session["last_compaction_reason"] = snapshot["reason"]
        session["last_compaction_at"] = snapshot["created_at"]
        session["updated_at"] = datetime.now().isoformat(timespec="seconds")
        self.save(data)
        self.append_log("loop_history", {"session_id": session_id, "event": "context_compacted", "compaction": snapshot})

    def compacted_summary(self, session_id: str | None = None) -> str:
        record = self.get(session_id) if session_id else self.active()
        return str((record or {}).get("compacted_summary", "") or "")

    @_synchronized
    def append_turn_record(self, session_id: str, record: dict[str, Any]) -> None:
        data = self.load()
        session = data.get("sessions", {}).get(session_id)
        if isinstance(session, dict):
            recent = session.setdefault("recent_turn_records", [])
            recent.append({
                "turn_id": record.get("turn_id", ""),
                "status": record.get("status", ""),
                "tool_round_count": len(record.get("tool_rounds", [])),
                "final_preview": str(record.get("final_assistant_text", ""))[:240],
                "created_at": record.get("created_at", ""),
            })
            session["recent_turn_records"] = recent[-6:]
            session["updated_at"] = datetime.now().isoformat(timespec="seconds")
            self.save(data)
        self.append_log("loop_history", {"session_id": session_id, **record})

    def working_dir(self, session_id: str | None = None) -> Path:
        record = self.get(session_id) if session_id else self.active()
        if record is None:
            record = self.create()
        path = (self.workspace_root / str(record["working_dir"])).resolve()
        if path != self.working_root and self.working_root not in path.parents:
            raise ValueError(f"Session working_dir escapes .working_dir: {record.get('working_dir')}")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def artifact_checklist(self, session_id: str | None = None) -> dict[str, bool]:
        root = self.working_dir(session_id)
        idea_dir = root / "idea2video"
        idea_scene_dirs = sorted(path for path in idea_dir.glob("scene_*") if path.is_dir()) if idea_dir.exists() else []
        idea_scene_storyboards = [path / "storyboard.json" for path in idea_scene_dirs]
        idea_scene_camera_trees = [path / "camera_tree.json" for path in idea_scene_dirs]
        idea_scene_shot_desc_groups = [list((scene / "shots").glob("*/shot_description.json")) for scene in idea_scene_dirs]
        idea_scene_selector_outputs = [output for scene in idea_scene_dirs for output in (scene / "shots").glob("*/*_selector_output.json")]

        script_shots = root / "script2video" / "shots"
        script_shot_descs = list(script_shots.glob("*/shot_description.json")) if script_shots.exists() else []
        script_selector_outputs = list(script_shots.glob("*/*_selector_output.json")) if script_shots.exists() else []

        novel_dir = root / "novel2video"
        novel_events = list((novel_dir / "events").glob("event_*.json")) if novel_dir.exists() else []
        novel_relevant_chunks = [path for path in (novel_dir / "relevant_chunks").glob("event_*/*") if path.is_file()] if novel_dir.exists() else []
        novel_scenes = list((novel_dir / "scenes").glob("event_*/scene_*.json")) if novel_dir.exists() else []
        novel_event_chars = list((novel_dir / "global_information" / "characters" / "event_level").glob("event_*_characters.json")) if novel_dir.exists() else []
        novel_level_chars = list((novel_dir / "global_information" / "characters" / "novel_level").glob("novel_characters_after_event_*.json")) if novel_dir.exists() else []
        return {
            "idea2video/story.txt": (idea_dir / "story.txt").exists(),
            "idea2video/characters.json": (idea_dir / "characters.json").exists(),
            "idea2video/script.json": (idea_dir / "script.json").exists(),
            "idea2video/scene_*/storyboard.json": bool(idea_scene_storyboards) and all(path.exists() for path in idea_scene_storyboards),
            "idea2video/scene_*/camera_tree.json": bool(idea_scene_camera_trees) and all(path.exists() for path in idea_scene_camera_trees),
            "idea2video/scene_*/shots/*/shot_description.json": bool(idea_scene_shot_desc_groups) and all(idea_scene_shot_desc_groups),
            "idea2video/scene_*/shots/*/*_selector_output.json": bool(idea_scene_selector_outputs),
            "idea2video/final_video.mp4": (idea_dir / "final_video.mp4").exists(),
            "script2video/script.txt": (root / "script2video" / "script.txt").exists(),
            "script2video/characters.json": (root / "script2video" / "characters.json").exists(),
            "script2video/storyboard.json": (root / "script2video" / "storyboard.json").exists(),
            "script2video/shots/*/shot_description.json": bool(script_shot_descs),
            "script2video/camera_tree.json": (root / "script2video" / "camera_tree.json").exists(),
            "script2video/shots/*/*_selector_output.json": bool(script_selector_outputs),
            "script2video/final_video.mp4": (root / "script2video" / "final_video.mp4").exists(),
            "novel2video/novel/novel.txt": (novel_dir / "novel" / "novel.txt").exists(),
            "novel2video/novel/novel_compressed.txt": (novel_dir / "novel" / "novel_compressed.txt").exists(),
            "novel2video/events/event_*.json": bool(novel_events),
            "novel2video/relevant_chunks/event_*": bool(novel_relevant_chunks),
            "novel2video/scenes/event_*/scene_*.json": bool(novel_scenes),
            "novel2video/global_information/characters/event_level/*.json": bool(novel_event_chars),
            "novel2video/global_information/characters/novel_level/*.json": bool(novel_level_chars),
        }

    def memory_text(self) -> str:
        return self.memory_path.read_text(encoding="utf-8") if self.memory_path.exists() else ""

    def write_memory(self, text: str) -> None:
        self.memory_path.write_text(text, encoding="utf-8")

    def append_log(self, name: str, payload: dict[str, Any]) -> None:
        event = {"timestamp": datetime.now().isoformat(timespec="seconds"), **payload}
        path = self.logs_dir / f"{name}.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

    def snapshot(self) -> dict[str, Any]:
        active = self.active()
        if active is None:
            return {"active_session_id": "", "session": None}
        return {"active_session_id": active["session_id"], "session": active, "artifact_checklist": self.artifact_checklist(active["session_id"])}

    def _with_session_defaults(self, record: dict[str, Any]) -> dict[str, Any]:
        item = dict(record)
        item.setdefault("project_name", "")
        item.setdefault("compacted_summary", "")
        item.setdefault("compacted_turns", 0)
        item.setdefault("compaction_snapshots", [])
        item.setdefault("last_compaction_reason", "")
        item.setdefault("last_compaction_at", "")
        item.setdefault("recent_turn_records", [])
        return item

    def _new_session_id(self, source: str, sessions: dict[str, Any]) -> str:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        slug = (re.sub(r"[^a-zA-Z0-9]+", "-", source.lower()).strip("-")[:32].strip("-") or "vimax")
        return self._dedupe_session_id(f"{stamp}-{slug}", sessions)

    def _dedupe_session_id(self, base: str, sessions: dict[str, Any]) -> str:
        candidate = base
        counter = 2
        while candidate in sessions:
            candidate = f"{base}-{counter}"
            counter += 1
        return candidate

    def _normalize_session_id(self, session_id: str | None) -> str:
        raw = str(session_id or "").strip()
        if not raw:
            raise ValueError("session_id cannot be empty")
        normalized = re.sub(r"[^a-zA-Z0-9]+", "-", raw).strip("-")[:96]
        if not normalized:
            raise ValueError(f"Invalid session_id: {session_id}")
        return normalized

    def _working_dir_for_id(self, session_id: str) -> Path:
        path = (self.working_root / session_id).resolve()
        if path != self.working_root and self.working_root not in path.parents:
            raise ValueError(f"Session path escapes .working_dir: {session_id}")
        return path
