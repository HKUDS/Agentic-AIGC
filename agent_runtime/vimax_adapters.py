from __future__ import annotations

import asyncio
from datetime import datetime
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import json
import logging
import os
from pathlib import Path
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from tenacity import RetryError

from interfaces import CharacterInScene
from agents.event_extractor import EventExtractor
from agents.global_information_planner import GlobalInformationPlanner
from agents.novel_compressor import NovelCompressor
from agents.scene_extractor import SceneExtractor
from pipelines.novel2movie_pipeline import Novel2MoviePipeline
from pipelines.idea2video_pipeline import Idea2VideoPipeline
from pipelines.script2video_pipeline import Script2VideoPipeline
from tools.image_generator_nanobanana_yunwu_api import ImageGeneratorNanobananaYunwuAPI
from tools.image_generator_openrouter_api import ImageGeneratorOpenRouterAPI
from tools.image_generator_orcarouter_api import ImageGeneratorOrcaRouterAPI
from tools.reranker_bge_silicon_api import RerankerBgeSiliconapi
from tools.video_generator_openrouter_api import VideoGeneratorOpenRouterAPI
from tools.video_generator_veo_yunwu_api import VideoGeneratorVeoYunwuAPI

from .config import api_provider_from_base_url, embedding_api_key, embedding_base_url, embedding_model, embedding_model_provider, image_api_key, image_base_url, image_model, llm_api_key, llm_base_url, llm_model, llm_model_provider, reranker_api_key, reranker_base_url, reranker_model, video_api_key, video_base_url, video_model, video_provider
from .models import ToolResult
from .tools import ToolArgumentSchema, ToolRuntimeContext, ToolSpec


class _UnavailableGenerator:
    async def generate_single_image(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Image generator is not available in narrative planning mode")

    async def generate_single_video(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Video generator is not available in narrative planning mode")


def build_vimax_adapter_specs(workspace_root: str | Path, session_index: Any) -> list[ToolSpec]:
    adapter = ViMaxAdapters(Path(workspace_root), session_index)
    return [
        ToolSpec(
            name="vimax_narrative_planning",
            description=(
                "Create or revise ViMax structured text artifacts for the active session. "
                "Idea mode writes story, characters, script, and scene-level storyboard/shot_decomposition/camera_tree under idea2video/scene_<idx>/. "
                "Script mode writes characters, storyboard, shot_decomposition, and camera_tree under script2video/. "
                "Pass the active session_id from prompt context when the user is working in the selected project. An empty active session is initialized in place; a different source on a non-empty session creates a new session instead of overwriting existing artifacts. If idea/script/revision_target are omitted and the active session has an idea, continue that session and fill missing structured text artifacts. "
                "It does not generate keyframes, video clips, or final video. Call this before revising storyboard/shots when those artifacts do not exist."
            ),
            handler=adapter.vimax_narrative_planning,
            schema={
                "session_id": ToolArgumentSchema(str, required=False, default=""),
                "idea": ToolArgumentSchema(str, required=False, default=""),
                "script": ToolArgumentSchema(str, required=False, default=""),
                "user_requirement": ToolArgumentSchema(str, required=False, default=""),
                "style": ToolArgumentSchema(str, required=False, default=""),
                "revision_target": ToolArgumentSchema(str, required=False, default=""),
                "revision_instruction": ToolArgumentSchema(str, required=False, default=""),
            },
        ),
        ToolSpec(
            name="vimax_novel_planning",
            description=(
                "Create ViMax structured text artifacts from a novel or novel excerpt. "
                "This writes novel2video/novel, events, relevant_chunks, scenes, and global_information text artifacts. "
                "Use this when the user provides long prose, a novel excerpt, or asks for novel-to-video planning. Pass the active session_id when the user is working in a selected empty project. "
                "It does not generate character portraits, scene videos, or final video."
            ),
            handler=adapter.vimax_novel_planning,
            schema={
                "session_id": ToolArgumentSchema(str, required=False, default=""),
                "novel_text": ToolArgumentSchema(str, required=True),
                "user_requirement": ToolArgumentSchema(str, required=False, default=""),
                "style": ToolArgumentSchema(str, required=False, default=""),
            },
        ),
        ToolSpec(
            name="vimax_render_video",
            description=(
                "Render keyframes, video clips, and final video for the active ViMax session. "
                "This checks that structured text artifacts exist before rendering and reports missing dependencies instead of pretending render started."
            ),
            handler=adapter.vimax_render_video,
            schema={
                "session_id": ToolArgumentSchema(str, required=False, default=""),
                "mode": ToolArgumentSchema(str, required=False, default="foreground"),
                "force": ToolArgumentSchema(bool, required=False, default=False),
            },
        ),
    ]


class ViMaxAdapters:
    def __init__(self, workspace_root: Path, session_index: Any) -> None:
        self.workspace_root = workspace_root.resolve()
        self.session_index = session_index

    async def vimax_narrative_planning(self, args: dict[str, Any], runtime: ToolRuntimeContext | None = None) -> ToolResult:
        idea = str(args.get("idea", "") or "").strip()
        script = str(args.get("script", "") or "").strip()
        user_requirement = str(args.get("user_requirement", "") or "").strip()
        requested_style = str(args.get("style", "") or "").strip()
        style = requested_style
        session = self._resolve_session(str(args.get("session_id", "") or ""), idea=idea, script=script, user_requirement=user_requirement, style=requested_style)
        session_id = session["session_id"]
        working_dir = self.session_index.working_dir(session_id)
        idea_dir = working_dir / "idea2video"
        script_dir = working_dir / "script2video"
        idea_dir.mkdir(parents=True, exist_ok=True)
        script_dir.mkdir(parents=True, exist_ok=True)

        if not idea and not script:
            revision_target = str(args.get("revision_target") or "").strip()
            if revision_target:
                return await self._revise_narrative_artifact(session_id, working_dir, revision_target, str(args.get("revision_instruction") or "").strip(), runtime)
            session_idea = str(session.get("idea") or "").strip()
            if session_idea:
                idea = session_idea
                user_requirement = user_requirement or str(session.get("user_requirement") or "").strip()
                style = requested_style or str(session.get("style") or "").strip() or "Cinematic, coherent, 16:9"
            else:
                return ToolResult("vimax_narrative_planning", False, "Provide `idea`, `script`, a revision target, or an active session with an existing idea for narrative planning.", {"error_type": "missing_input", "session_id": session_id})

        style = style or str(session.get("style") or "").strip() or "Cinematic, coherent, 16:9"
        self._update_session_metadata(session_id, idea="", user_requirement="", style=style)

        try:
            self.session_index.update_stage(session_id, "narrative_planning", "Generating structured text artifacts")
            if runtime:
                runtime.emit_progress("Starting narrative planning", stage="starting", metadata={"session_id": session_id})
                await asyncio.sleep(0)
            generated_before = self.session_index.artifact_checklist(session_id)
            if runtime:
                runtime.emit_progress("Initializing bounded chat model", stage="initializing_llm", metadata={"session_id": session_id, "timeout_seconds": _llm_request_timeout_seconds(), "max_tokens": _narrative_max_tokens()})
                await asyncio.sleep(0)
            chat_model = _build_chat_model()
            if runtime:
                runtime.emit_progress("Bounded chat model initialized", stage="chat_model_ready", metadata={"session_id": session_id})
                await asyncio.sleep(0)
            dummy = _UnavailableGenerator()
            # Do not globally redirect stdout/stderr while the JSONL CLI is streaming events.
            # The adapter exposes pipeline progress through explicit tool_progress events instead.
            if idea:
                idea_pipeline = Idea2VideoPipeline(chat_model=chat_model, image_generator=dummy, video_generator=dummy, working_dir=str(idea_dir))
                if runtime:
                    runtime.emit_progress("Idea pipeline initialized", stage="idea_pipeline_ready", metadata={"session_id": session_id})
                    await asyncio.sleep(0)
                story = await _run_planning_step(
                    "Developing story from user idea",
                    "develop_story",
                    idea_pipeline.develop_story(idea=idea, user_requirement=user_requirement, quiet=True),
                    runtime,
                    {"session_id": session_id},
                )
                characters = await _run_planning_step(
                    "Extracting characters from story",
                    "extract_characters",
                    idea_pipeline.extract_characters(story=story, quiet=True),
                    runtime,
                    {"session_id": session_id},
                )
                scene_scripts = await _run_planning_step(
                    "Writing scene scripts from story",
                    "write_script",
                    idea_pipeline.write_script_based_on_story(story=story, user_requirement=user_requirement, quiet=True),
                    runtime,
                    {"session_id": session_id},
                )
                for idx, scene_script in enumerate(scene_scripts if isinstance(scene_scripts, list) else [scene_scripts]):
                    scene_dir = idea_dir / f"scene_{idx}"
                    scene_text = scene_script if isinstance(scene_script, str) else json.dumps(scene_script, ensure_ascii=False, indent=2)
                    script_pipeline = Script2VideoPipeline(chat_model=chat_model, image_generator=dummy, video_generator=dummy, working_dir=str(scene_dir))
                    await _run_planning_step(
                        f"Planning scene {idx} storyboard and shots",
                        "plan_scene",
                        script_pipeline.plan_text_artifacts(script=scene_text, user_requirement=user_requirement, style=style, characters=characters, progress=_pipeline_progress(runtime, session_id, scene_index=idx), quiet=True),
                        runtime,
                        {"session_id": session_id, "scene_index": idx},
                    )
            else:
                (script_dir / "script.txt").write_text(script, encoding="utf-8")
                script_pipeline = Script2VideoPipeline(chat_model=chat_model, image_generator=dummy, video_generator=dummy, working_dir=str(script_dir))
                if runtime:
                    runtime.emit_progress("Script pipeline initialized", stage="script_pipeline_ready", metadata={"session_id": session_id})
                    await asyncio.sleep(0)
                await _run_planning_step(
                    "Planning storyboard and shots from provided script",
                    "plan_script",
                    script_pipeline.plan_text_artifacts(script=script, user_requirement=user_requirement, style=style, progress=_pipeline_progress(runtime, session_id), quiet=True),
                    runtime,
                    {"session_id": session_id},
                )
        except Exception as exc:
            self.session_index.update_stage(session_id, "error", f"Narrative planning failed: {exc}")
            checklist = self.session_index.artifact_checklist(session_id)
            payload = {
                "session_id": session_id,
                "working_dir": str(working_dir.relative_to(self.workspace_root)),
                "error_type": "recoverable_planning_step_failed",
                "retryable": True,
                "error": str(exc),
                "present": [path for path, present in checklist.items() if present],
                "missing": [path for path, present in checklist.items() if not present],
            }
            if runtime:
                runtime.emit_progress("Narrative planning failed; partial artifacts were kept", stage="planning_failed", metadata=payload)
            return ToolResult("vimax_narrative_planning", False, f"Narrative planning failed: {exc}", payload)

        checklist = self.session_index.artifact_checklist(session_id)
        generated = [path for path, present in checklist.items() if present and not generated_before.get(path)]
        reused = [path for path, present in checklist.items() if present and generated_before.get(path)]
        ready_for_render = _ready_for_render(checklist)
        self.session_index.update_stage(session_id, "narrative_planned", "Structured text planning complete" if ready_for_render else "Structured text planning partially complete")
        if runtime:
            runtime.emit_progress("Narrative planning complete", stage="completed", metadata={"ready_for_render": ready_for_render})
        payload = {
            "session_id": session_id,
            "working_dir": str(working_dir.relative_to(self.workspace_root)),
            "generated": generated,
            "reused": reused,
            "missing": [path for path, present in checklist.items() if not present],
            "ready_for_render": ready_for_render,
        }
        return ToolResult("vimax_narrative_planning", True, json.dumps(payload, ensure_ascii=False, indent=2), payload)

    async def _revise_narrative_artifact(self, session_id: str, working_dir: Path, revision_target: str, revision_instruction: str, runtime: ToolRuntimeContext | None = None) -> ToolResult:
        if not revision_instruction:
            self.session_index.update_stage(session_id, "error", "Revision failed: missing revision_instruction")
            return ToolResult("vimax_narrative_planning", False, "revision_instruction is required when revision_target is provided.", {"error_type": "missing_revision_instruction", "session_id": session_id, "revision_target": revision_target})
        try:
            target_path = _resolve_artifact_path(working_dir, revision_target)
        except ValueError as exc:
            self.session_index.update_stage(session_id, "error", f"Revision failed: {exc}")
            return ToolResult("vimax_narrative_planning", False, str(exc), {"error_type": "invalid_revision_target", "session_id": session_id, "revision_target": revision_target})
        if not target_path.exists():
            self.session_index.update_stage(session_id, "error", f"Revision failed: target does not exist: {revision_target}")
            return ToolResult("vimax_narrative_planning", False, f"Revision target does not exist: {revision_target}", {"error_type": "dependency_missing", "session_id": session_id, "revision_target": revision_target})
        try:
            self.session_index.update_stage(session_id, "narrative_planning", "Revising structured text artifact")
            if runtime:
                runtime.emit_progress("Revising structured text artifact", stage="revising", metadata={"session_id": session_id, "revision_target": revision_target})
            chat_model = _build_chat_model()
            before = target_path.read_text(encoding="utf-8")
            revised = await _revise_artifact_with_llm(chat_model, target_path.relative_to(working_dir).as_posix(), before, revision_instruction)
            if target_path.suffix == ".json":
                try:
                    revised_payload = json.loads(revised)
                except json.JSONDecodeError as exc:
                    self.session_index.update_stage(session_id, "error", f"Revision failed: invalid JSON output: {exc}")
                    return ToolResult("vimax_narrative_planning", False, f"Revision output was not valid JSON: {exc}", {"error_type": "invalid_revision_json", "session_id": session_id, "revision_target": revision_target})
                revised = json.dumps(revised_payload, ensure_ascii=False, indent=2)
            target_path.write_text(revised, encoding="utf-8")
        except Exception as exc:
            self.session_index.update_stage(session_id, "error", f"Revision failed: {exc}")
            raise

        stale = _stale_keys_for_revision(target_path.relative_to(working_dir).as_posix())
        if stale:
            self.session_index.mark_stale(session_id, stale)
        self.session_index.append_log("revisions", {"session_id": session_id, "target": target_path.relative_to(working_dir).as_posix(), "instruction": revision_instruction, "stale": stale, "before_preview": before[:500], "after_preview": revised[:500]})
        checklist = self.session_index.artifact_checklist(session_id)
        ready_for_render = _ready_for_render(checklist)
        self.session_index.update_stage(session_id, "narrative_planned" if ready_for_render else "narrative_planning", "Revised structured text artifact")
        payload = {
            "session_id": session_id,
            "working_dir": str(working_dir.relative_to(self.workspace_root)),
            "generated": [],
            "reused": [path for path, present in checklist.items() if present],
            "revised": [target_path.relative_to(working_dir).as_posix()],
            "missing": [path for path, present in checklist.items() if not present],
            "stale": stale,
            "ready_for_render": ready_for_render,
            "revision_target": target_path.relative_to(working_dir).as_posix(),
        }
        return ToolResult("vimax_narrative_planning", True, json.dumps(payload, ensure_ascii=False, indent=2), payload)

    async def vimax_novel_planning(self, args: dict[str, Any], runtime: ToolRuntimeContext | None = None) -> ToolResult:
        novel_text = str(args.get("novel_text", "") or "").strip()
        user_requirement = str(args.get("user_requirement", "") or "").strip()
        style = str(args.get("style", "") or "").strip() or "Cinematic, coherent, 16:9"
        if not novel_text:
            return ToolResult("vimax_novel_planning", False, "novel_text is required for novel planning.", {"error_type": "missing_input"})

        session_id_arg = str(args.get("session_id", "") or "").strip()
        session = self._resolve_session(session_id_arg, idea=novel_text, script="", user_requirement=user_requirement, style=style)
        session_id = session["session_id"]
        working_dir = self.session_index.working_dir(session_id)
        novel_dir = working_dir / "novel2video"
        novel_dir.mkdir(parents=True, exist_ok=True)
        generated_before = self.session_index.artifact_checklist(session_id)

        try:
            self.session_index.update_stage(session_id, "novel_planning", "Generating novel structured text artifacts")
            if runtime:
                runtime.emit_progress("Starting novel planning", stage="starting", metadata={"session_id": session_id})
                await asyncio.sleep(0)
            pipeline = _build_novel_pipeline(novel_dir)
            await _run_planning_step(
                "Planning novel structured text artifacts",
                "novel_plan_text_artifacts",
                pipeline.plan_text_artifacts(
                    novel_text=novel_text,
                    user_requirement=user_requirement,
                    style=style,
                    progress=_pipeline_progress(runtime, session_id),
                    quiet=True,
                ),
                runtime,
                {"session_id": session_id},
            )
        except Exception as exc:
            self.session_index.update_stage(session_id, "error", f"Novel planning failed: {exc}")
            return ToolResult("vimax_novel_planning", False, str(exc), {"error_type": "exception", "session_id": session_id})

        checklist = self.session_index.artifact_checklist(session_id)
        generated = [path for path, present in checklist.items() if path.startswith("novel2video/") and present and not generated_before.get(path)]
        reused = [path for path, present in checklist.items() if path.startswith("novel2video/") and present and generated_before.get(path)]
        missing = [path for path, present in checklist.items() if path.startswith("novel2video/") and not present]
        ready = _novel_text_ready(checklist)
        self.session_index.update_stage(session_id, "novel_planned" if ready else "novel_planning", "Novel structured text planning complete" if ready else "Novel structured text planning partially complete")
        if runtime:
            runtime.emit_progress("Novel planning complete", stage="completed", metadata={"session_id": session_id, "ready_for_scene_render": False})
        payload = {
            "session_id": session_id,
            "working_dir": str(working_dir.relative_to(self.workspace_root)),
            "generated": generated,
            "reused": reused,
            "missing": missing,
            "ready_for_scene_render": False,
        }
        return ToolResult("vimax_novel_planning", True, json.dumps(payload, ensure_ascii=False, indent=2), payload)

    async def vimax_render_video(self, args: dict[str, Any], runtime: ToolRuntimeContext | None = None) -> ToolResult:
        session_id = str(args.get("session_id", "") or "").strip()
        session = self.session_index.get(session_id) if session_id else self.session_index.active()
        if session is None:
            return ToolResult("vimax_render_video", False, "No active session to render.", {"error_type": "missing_session"})
        session_id = session["session_id"]
        checklist = self.session_index.artifact_checklist(session_id)
        missing = _missing_render_dependencies(checklist)
        working_dir = self.session_index.working_dir(session_id)
        if missing:
            payload = {"error_type": "dependency_missing", "missing": missing, "session_id": session_id}
            _write_render_status(working_dir, status="dependency_missing", payload=payload)
            return ToolResult("vimax_render_video", False, f"Dependency missing: {', '.join(missing)}", payload)

        self.session_index.update_stage(session_id, "rendering", "Rendering video artifacts")
        _write_render_status(working_dir, status="rendering", payload={"session_id": session_id, "render_started": True, "render_completed": False})
        try:
            chat_model = _build_chat_model()
            image_generator = _build_image_generator()
            video_generator = _build_video_generator()
            if runtime:
                runtime.emit_progress("Starting video render", stage="rendering", metadata={"session_id": session_id})
            if _idea_mode_ready(checklist):
                idea_pipeline = Idea2VideoPipeline(chat_model=chat_model, image_generator=image_generator, video_generator=video_generator, working_dir=str(working_dir / "idea2video"))
                with _suppress_pipeline_output():
                    final_video = await idea_pipeline(idea=str(session.get("idea", "")), user_requirement=str(session.get("user_requirement", "")), style=str(session.get("style", "")), quiet=True)
                self.session_index.update_stage(session_id, "rendered", "Final video rendered")
                payload = {"session_id": session_id, "render_mode": "idea2video", "render_started": True, "render_completed": True, "final_video_path": str(Path(final_video).relative_to(self.workspace_root)), "missing": []}
                _write_render_status(working_dir, status="rendered", payload=payload)
                return ToolResult("vimax_render_video", True, json.dumps(payload, ensure_ascii=False, indent=2), payload)
            if _script_mode_ready(checklist):
                script_dir = working_dir / "script2video"
                script_text = _load_script_text(working_dir)
                characters = _load_characters(script_dir / "characters.json")
                pipeline = Script2VideoPipeline(chat_model=chat_model, image_generator=image_generator, video_generator=video_generator, working_dir=str(script_dir))
                with _suppress_pipeline_output():
                    final_video = await pipeline(script=script_text, user_requirement=str(session.get("user_requirement", "")), style=str(session.get("style", "")), characters=characters, quiet=True, progress=_pipeline_progress(runtime, session_id))
                self.session_index.update_stage(session_id, "rendered", "Final video rendered")
                payload = {"session_id": session_id, "render_mode": "script2video", "render_started": True, "render_completed": True, "final_video_path": str(Path(final_video).relative_to(self.workspace_root)), "missing": []}
                _write_render_status(working_dir, status="rendered", payload=payload)
                return ToolResult("vimax_render_video", True, json.dumps(payload, ensure_ascii=False, indent=2), payload)
            if _novel_mode_ready(checklist):
                novel_dir = working_dir / "novel2video"
                pipeline = _build_novel_render_pipeline(novel_dir, chat_model, image_generator, video_generator)
                with _suppress_pipeline_output():
                    render_result = await pipeline.render_video_artifacts(style=str(session.get("style", "")), user_requirement=str(session.get("user_requirement", "")), quiet=True, progress=_pipeline_progress(runtime, session_id))
                scene_videos_dir = Path(render_result["scene_videos_dir"])
                self.session_index.update_stage(session_id, "novel_scene_rendered", "Novel scene videos rendered")
                payload = {
                    "session_id": session_id,
                    "render_mode": "novel2video",
                    "render_started": True,
                    "render_completed": True,
                    "scene_render_completed": True,
                    "final_video_path": None,
                    "scene_videos_dir": str(scene_videos_dir.relative_to(self.workspace_root)),
                    "scene_video_dirs": [str(Path(path).relative_to(self.workspace_root)) for path in render_result.get("scene_video_dirs", [])],
                    "scene_count": render_result.get("scene_count", 0),
                    "missing": [],
                }
                _write_render_status(working_dir, status="rendered", payload=payload)
                return ToolResult("vimax_render_video", True, json.dumps(payload, ensure_ascii=False, indent=2), payload)
        except Exception as exc:
            unwrapped = _unwrap_retry_error(exc)
            error_text = _sanitize_error_text(str(unwrapped))
            wrapped_error_text = _sanitize_error_text(str(exc))
            self.session_index.update_stage(session_id, "error", f"Render failed: {error_text}")
            checklist = self.session_index.artifact_checklist(session_id)
            payload = {
                "error_type": "render_failed",
                "retryable": _is_retryable_render_error(unwrapped),
                "session_id": session_id,
                "error": error_text,
                "wrapped_error": wrapped_error_text,
                "present": [path for path, present in checklist.items() if present],
                "missing": [path for path, present in checklist.items() if not present],
            }
            _write_render_status(working_dir, status="error", payload=payload)
            if runtime:
                runtime.emit_progress("Render failed; partial artifacts were kept", stage="render_failed", metadata=payload)
            return ToolResult("vimax_render_video", False, f"Render failed: {error_text}", payload)
        payload = {"error_type": "dependency_missing", "session_id": session_id}
        _write_render_status(working_dir, status="dependency_missing", payload=payload)
        return ToolResult("vimax_render_video", False, "No render mode matched current session.", payload)

    def _resolve_session(self, session_id: str, *, idea: str, script: str, user_requirement: str, style: str) -> dict[str, Any]:
        requested_source = idea or script
        if session_id:
            session = self.session_index.get(session_id)
            if session is None:
                session = self.session_index.create(idea=requested_source, user_requirement=user_requirement, style=style, session_id=session_id)
            elif requested_source and _is_new_source_for_session(session, requested_source):
                session = self.session_index.create(idea=requested_source, user_requirement=user_requirement, style=style)
            else:
                self.session_index.set_active(session_id)
        else:
            if requested_source:
                active = self.session_index.active()
                if active is not None and self._session_is_empty(active):
                    session = self.session_index.set_active(active["session_id"])
                else:
                    session = self.session_index.create(idea=requested_source, user_requirement=user_requirement, style=style)
            else:
                session = self.session_index.active() or self.session_index.create(idea=requested_source, user_requirement=user_requirement, style=style)
        self._update_session_metadata(session["session_id"], idea=requested_source, user_requirement=user_requirement, style=style)
        return self.session_index.get(session["session_id"]) or session

    def _session_is_empty(self, session: dict[str, Any]) -> bool:
        if str(session.get("idea") or "").strip():
            return False
        session_id = str(session.get("session_id") or "").strip()
        if not session_id:
            return False
        return not any(self.session_index.artifact_checklist(session_id).values())

    def _update_session_metadata(self, session_id: str, *, idea: str, user_requirement: str, style: str) -> None:
        data = self.session_index.load()
        record = data.get("sessions", {}).get(session_id)
        if not isinstance(record, dict):
            return
        if idea and not record.get("idea"):
            record["idea"] = idea
        if user_requirement:
            record["user_requirement"] = user_requirement
        if style:
            record["style"] = style
        self.session_index.save(data)


class _DiscardStream:
    def write(self, text: str) -> int:
        return len(text)

    def flush(self) -> None:
        pass


_PIPELINE_OUTPUT_SINK = _DiscardStream()


@contextmanager
def _suppress_pipeline_output():
    previous_disable_level = logging.root.manager.disable
    logging.disable(logging.WARNING)
    try:
        with redirect_stdout(_PIPELINE_OUTPUT_SINK), redirect_stderr(_PIPELINE_OUTPUT_SINK):
            yield
    finally:
        logging.disable(previous_disable_level)


def _narrative_step_timeout_seconds() -> float:
    raw = os.environ.get("VIMAX_NARRATIVE_STEP_TIMEOUT_SECONDS", "900")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 900.0


async def _run_planning_step(
    message: str,
    stage: str,
    awaitable: Any,
    runtime: ToolRuntimeContext | None,
    metadata: dict[str, Any] | None = None,
) -> Any:
    timeout_seconds = _narrative_step_timeout_seconds()
    event_metadata = dict(metadata or {})
    event_metadata["timeout_seconds"] = timeout_seconds
    if runtime:
        runtime.emit_progress(message, stage=stage, metadata=event_metadata)
        await asyncio.sleep(0)
    try:
        with _suppress_pipeline_output():
            if timeout_seconds <= 0:
                return await awaitable
            return await asyncio.wait_for(awaitable, timeout=timeout_seconds)
    except asyncio.TimeoutError as exc:
        raise RuntimeError(f"{message} timed out after {timeout_seconds:g}s") from exc
    except Exception as exc:
        raise RuntimeError(f"{message} failed: {exc}") from exc


def _is_new_source_for_session(session: dict[str, Any], requested_source: str) -> bool:
    current = str(session.get("idea") or "").strip()
    requested = requested_source.strip()
    if not current or not requested:
        return False
    return current != requested


def _llm_request_timeout_seconds() -> float:
    raw = os.environ.get("VIMAX_LLM_REQUEST_TIMEOUT_SECONDS", "300")
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 300.0


def _narrative_max_tokens() -> int:
    raw = os.environ.get("VIMAX_NARRATIVE_MAX_TOKENS", "4096")
    try:
        return max(256, int(raw))
    except ValueError:
        return 4096


def _pipeline_progress(runtime: ToolRuntimeContext | None, session_id: str, *, scene_index: int | None = None):
    if runtime is None:
        return None

    def emit(stage: str, message: str, metadata: dict[str, Any] | None = None) -> None:
        payload = dict(metadata or {})
        payload["session_id"] = session_id
        if scene_index is not None:
            payload["scene_index"] = scene_index
        runtime.emit_progress(message, stage=stage, metadata=payload)

    return emit


def _build_chat_model() -> Any:
    api_key = llm_api_key()
    if not api_key:
        raise RuntimeError("VIMAX_LLM_API_KEY or configs/agent.local.yaml llm.api_key is required for narrative planning")
    return init_chat_model(
        model=llm_model(),
        model_provider=llm_model_provider(),
        api_key=api_key,
        base_url=llm_base_url(),
        timeout=_llm_request_timeout_seconds(),
        max_retries=0,
        max_completion_tokens=_narrative_max_tokens(),
    )


def _build_image_generator() -> ImageGeneratorNanobananaYunwuAPI | ImageGeneratorOpenRouterAPI | ImageGeneratorOrcaRouterAPI:
    api_key = image_api_key()
    if not api_key:
        raise RuntimeError("VIMAX_IMAGE_API_KEY, VIMAX_LLM_API_KEY, or configs/agent.local.yaml image/llm api_key is required for image generation")
    model = image_model()
    base_url = image_base_url()
    provider = api_provider_from_base_url(base_url)
    if provider == "openrouter":
        return ImageGeneratorOpenRouterAPI(api_key=api_key, model=model, base_url=base_url)
    if provider == "orcarouter":
        return ImageGeneratorOrcaRouterAPI(api_key=api_key, model=model, base_url=base_url)
    return ImageGeneratorNanobananaYunwuAPI(api_key=api_key, model=model, base_url=base_url)


def _build_video_generator() -> VideoGeneratorVeoYunwuAPI | VideoGeneratorOpenRouterAPI:
    api_key = video_api_key()
    if not api_key:
        raise RuntimeError("VIMAX_VIDEO_API_KEY, VIMAX_LLM_API_KEY, or configs/agent.local.yaml video/llm api_key is required for video generation")
    model = video_model()
    base_url = video_base_url()
    provider = video_provider().strip().lower()
    if provider == "openrouter":
        return VideoGeneratorOpenRouterAPI(api_key=api_key, model=model, base_url=base_url)
    if provider == "yunwu":
        return VideoGeneratorVeoYunwuAPI(api_key=api_key, t2v_model=model, ff2v_model=model, base_url=base_url)
    raise RuntimeError(f"Unsupported video base_url for automatic provider matching: {base_url}")


class _IdentityRewriter:
    async def __call__(self, prompt: str) -> str:
        return prompt


def _build_embedding_model() -> Any:
    api_key = embedding_api_key()
    base_url = embedding_base_url()
    provider = embedding_model_provider().strip().lower()
    if not api_key or not base_url:
        raise RuntimeError("VIMAX_EMBEDDING_API_KEY or configs/agent.local.yaml embedding api_key/base_url is required for novel planning")
    if provider != "openai":
        raise RuntimeError(f"Unsupported embedding model_provider: {provider}")
    return OpenAIEmbeddings(model=embedding_model(), api_key=api_key, base_url=base_url)


def _build_reranker() -> RerankerBgeSiliconapi:
    api_key = reranker_api_key()
    base_url = reranker_base_url()
    if not api_key or not base_url:
        raise RuntimeError("VIMAX_RERANKER_API_KEY or configs/agent.local.yaml reranker api_key/base_url is required for novel planning")
    return RerankerBgeSiliconapi(api_key=api_key, base_url=base_url, model=reranker_model())


def _build_novel_pipeline(working_dir: Path) -> Novel2MoviePipeline:
    api_key = llm_api_key()
    if not api_key:
        raise RuntimeError("VIMAX_LLM_API_KEY or configs/agent.local.yaml llm.api_key is required for novel planning")
    base_url = llm_base_url()
    model = llm_model()
    dummy = _UnavailableGenerator()
    return Novel2MoviePipeline(
        novel_compressor=NovelCompressor(api_key=api_key, base_url=base_url, chat_model=model),
        event_extractor=EventExtractor(api_key=api_key, base_url=base_url, chat_model=model),
        embeddings=_build_embedding_model(),
        rerank_model=_build_reranker(),
        scene_extractor=SceneExtractor(api_key=api_key, base_url=base_url, chat_model=model),
        global_information_planner=GlobalInformationPlanner(api_key=api_key, base_url=base_url, chat_model=model),
        image_generator=dummy,
        rewriter=_IdentityRewriter(),
        script2video_pipeline=dummy,
        working_dir=str(working_dir),
    )


def _build_novel_render_pipeline(working_dir: Path, chat_model: Any, image_generator: Any, video_generator: Any) -> Novel2MoviePipeline:
    api_key = llm_api_key()
    if not api_key:
        raise RuntimeError("VIMAX_LLM_API_KEY or configs/agent.local.yaml llm.api_key is required for novel rendering")
    base_url = llm_base_url()
    model = llm_model()
    script_pipeline = Script2VideoPipeline(chat_model=chat_model, image_generator=image_generator, video_generator=video_generator, working_dir=str(working_dir / "videos"))
    return Novel2MoviePipeline(
        novel_compressor=NovelCompressor(api_key=api_key, base_url=base_url, chat_model=model),
        event_extractor=EventExtractor(api_key=api_key, base_url=base_url, chat_model=model),
        embeddings=_build_embedding_model(),
        rerank_model=_build_reranker(),
        scene_extractor=SceneExtractor(api_key=api_key, base_url=base_url, chat_model=model),
        global_information_planner=GlobalInformationPlanner(api_key=api_key, base_url=base_url, chat_model=model),
        image_generator=image_generator,
        rewriter=_IdentityRewriter(),
        script2video_pipeline=script_pipeline,
        working_dir=str(working_dir),
    )


def _unwrap_retry_error(exc: Exception) -> Exception:
    if isinstance(exc, RetryError):
        try:
            return exc.last_attempt.exception() or exc
        except Exception:
            return exc
    return exc


def _is_retryable_render_error(exc: Exception) -> bool:
    text = str(exc).lower()
    if isinstance(exc, AttributeError):
        return False
    if "http 403" in text or "key limit exceeded" in text or "quota" in text:
        return False
    return True


def _sanitize_error_text(text: str) -> str:
    sanitized = text
    for marker in ("workspaces/default/keys/",):
        if marker in sanitized:
            prefix, rest = sanitized.split(marker, 1)
            key_id = []
            for char in rest:
                if char.isalnum() or char in "-_":
                    key_id.append(char)
                    continue
                break
            sanitized = prefix + marker + "<redacted>" + rest[len(key_id):]
    if "sk-" in sanitized:
        prefix, rest = sanitized.split("sk-", 1)
        token = []
        for char in rest:
            if char.isalnum() or char in "-_":
                token.append(char)
                continue
            break
        sanitized = prefix + "sk-<redacted>" + rest[len(token):]
    return sanitized


def _write_render_status(working_dir: Path, *, status: str, payload: dict[str, Any]) -> None:
    working_dir.mkdir(parents=True, exist_ok=True)
    event = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        **payload,
    }
    (working_dir / "render_status.json").write_text(json.dumps(event, ensure_ascii=False, indent=2), encoding="utf-8")
    with (working_dir / "render_events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def _write_characters_if_missing(path: Path, characters: list[CharacterInScene]) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([character.model_dump() for character in characters], ensure_ascii=False, indent=2), encoding="utf-8")


def _load_characters(path: Path) -> list[CharacterInScene]:
    return [CharacterInScene.model_validate(item) for item in json.loads(path.read_text(encoding="utf-8"))]


def _load_script_text(working_dir: Path) -> str:
    script_text = working_dir / "script2video" / "script.txt"
    if script_text.exists():
        return script_text.read_text(encoding="utf-8")
    idea_script = working_dir / "idea2video" / "script.json"
    if idea_script.exists():
        payload = json.loads(idea_script.read_text(encoding="utf-8"))
        return json.dumps(payload, ensure_ascii=False, indent=2) if not isinstance(payload, str) else payload
    story = working_dir / "idea2video" / "story.txt"
    if story.exists():
        return story.read_text(encoding="utf-8")
    return ""


def _resolve_artifact_path(working_dir: Path, revision_target: str) -> Path:
    rel = Path(revision_target)
    if rel.is_absolute():
        raise ValueError(f"revision_target must be relative to session working_dir: {revision_target}")
    path = (working_dir / rel).resolve()
    if path != working_dir and working_dir not in path.parents:
        raise ValueError(f"revision_target escapes session working_dir: {revision_target}")
    return path


async def _revise_artifact_with_llm(chat_model: Any, target: str, current_text: str, instruction: str) -> str:
    prompt = (
        "Revise this ViMax structured artifact exactly as requested. "
        "Return only the complete replacement file content, with no Markdown fences or explanation. "
        "If the file is JSON, preserve valid JSON and the existing schema shape.\n\n"
        f"Target: {target}\n"
        f"Revision instruction: {instruction}\n\n"
        "Current file content:\n"
        f"{current_text}"
    )
    if hasattr(chat_model, "ainvoke"):
        response = await chat_model.ainvoke(prompt)
    elif hasattr(chat_model, "invoke"):
        response = chat_model.invoke(prompt)
    else:
        raise RuntimeError("chat_model does not support invoke/ainvoke for revision mode")
    content = getattr(response, "content", response)
    if isinstance(content, list):
        content = "".join(str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in content)
    return _strip_markdown_fences(str(content).strip())


def _strip_markdown_fences(text: str) -> str:
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _stale_keys_for_revision(target: str) -> list[str]:
    if "storyboard.json" in target:
        return ["shot_descriptions", "camera_tree", "frames", "clips", "final_video"]
    if "shot_description.json" in target:
        return ["frames", "clips", "final_video"]
    if "camera_tree.json" in target:
        return ["frames", "clips", "final_video"]
    if target.endswith("script.json") or target.endswith("story.txt"):
        return ["storyboard", "shot_descriptions", "camera_tree", "frames", "clips", "final_video"]
    if target.endswith("characters.json"):
        return ["storyboard", "shot_descriptions", "frames", "clips", "final_video"]
    return ["frames", "clips", "final_video"]


def _ready_for_render(checklist: dict[str, bool]) -> bool:
    return _idea_mode_ready(checklist) or _script_mode_ready(checklist) or _novel_mode_ready(checklist)


def _missing_render_dependencies(checklist: dict[str, bool]) -> list[str]:
    if _ready_for_render(checklist):
        return []
    idea_required = ["idea2video/story.txt", "idea2video/characters.json", "idea2video/script.json", "idea2video/scene_*/storyboard.json", "idea2video/scene_*/shots/*/shot_description.json", "idea2video/scene_*/camera_tree.json"]
    script_required = ["script2video/script.txt", "script2video/characters.json", "script2video/storyboard.json", "script2video/shots/*/shot_description.json", "script2video/camera_tree.json"]
    novel_required = ["novel2video/novel/novel_compressed.txt", "novel2video/events/event_*.json", "novel2video/relevant_chunks/event_*", "novel2video/scenes/event_*/scene_*.json", "novel2video/global_information/characters/event_level/*.json", "novel2video/global_information/characters/novel_level/*.json"]
    return [f"idea mode: {path}" for path in idea_required if not checklist.get(path)] + [f"script mode: {path}" for path in script_required if not checklist.get(path)] + [f"novel mode: {path}" for path in novel_required if not checklist.get(path)]


def _idea_mode_ready(checklist: dict[str, bool]) -> bool:
    return bool(checklist.get("idea2video/story.txt") and checklist.get("idea2video/characters.json") and checklist.get("idea2video/script.json") and checklist.get("idea2video/scene_*/storyboard.json") and checklist.get("idea2video/scene_*/shots/*/shot_description.json") and checklist.get("idea2video/scene_*/camera_tree.json"))


def _novel_text_ready(checklist: dict[str, bool]) -> bool:
    return _novel_mode_ready(checklist)


def _novel_mode_ready(checklist: dict[str, bool]) -> bool:
    return bool(checklist.get("novel2video/novel/novel_compressed.txt") and checklist.get("novel2video/events/event_*.json") and checklist.get("novel2video/relevant_chunks/event_*") and checklist.get("novel2video/scenes/event_*/scene_*.json") and checklist.get("novel2video/global_information/characters/event_level/*.json") and checklist.get("novel2video/global_information/characters/novel_level/*.json"))


def _script_mode_ready(checklist: dict[str, bool]) -> bool:
    return bool(checklist.get("script2video/script.txt") and checklist.get("script2video/characters.json") and checklist.get("script2video/storyboard.json") and checklist.get("script2video/shots/*/shot_description.json") and checklist.get("script2video/camera_tree.json"))
