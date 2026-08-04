# TODO: NOT IMPLEMENTED YET

import os
import shutil
import yaml
import json
import importlib
import asyncio
from typing import Any, Callable, List, Dict
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from PIL import Image

from interfaces import (
    Event,
    Scene,
    CharacterInScene,
    CharacterInNovel,
    CharacterInEvent,
)
from tenacity import retry

from utils.text import safe_path_component



def _pipeline_print(quiet: bool, message: str) -> None:
    if not quiet:
        print(message)


def _emit_text_plan_progress(progress, stage: str, message: str, metadata: dict | None = None) -> None:
    if progress is not None:
        progress(stage, message, metadata or {})


def _event_file_index(path: str) -> int:
    return int(os.path.basename(path).split("_")[1].split(".")[0])


def _scene_file_index(path: str) -> int:
    return int(os.path.basename(path).split("_")[1].split(".")[0])

class Novel2MoviePipeline:
    def __init__(
        self,
        novel_compressor: Any,
        event_extractor: Any,
        embeddings: Any,
        rerank_model: Any,
        scene_extractor: Any,
        global_information_planner: Any,
        image_generator: Any,
        rewriter: Any,
        script2video_pipeline: Any,
        working_dir: str,
    ):
        self.novel_compressor = novel_compressor
        self.event_extractor = event_extractor
        self.embeddings = embeddings
        self.rerank_model = rerank_model
        self.scene_extractor = scene_extractor
        self.global_information_planner = global_information_planner
        self.image_generator = image_generator
        self.rewriter = rewriter
        self.script2video_pipeline = script2video_pipeline
        self.working_dir = working_dir
        os.makedirs(self.working_dir, exist_ok=True)


    async def plan_text_artifacts(
        self,
        novel_text: str,
        user_requirement: str = "",
        style: str = "",
        progress: Callable[[str, str, Dict[str, Any] | None], None] | None = None,
        quiet: bool = False,
    ) -> dict[str, Any]:
        """Generate structured text artifacts for novel adaptation only.

        This helper intentionally stops before character portrait generation,
        scene video generation, and final concatenation so the agent loop can
        pause after the novel planning stage.
        """
        del user_requirement, style

        _emit_text_plan_progress(progress, "save_novel", "Saving and splitting novel text")
        working_dir_novel = os.path.join(self.working_dir, "novel")
        os.makedirs(working_dir_novel, exist_ok=True)
        with open(os.path.join(working_dir_novel, "novel.txt"), "w", encoding="utf-8") as f:
            f.write(novel_text)

        novel_chunks = self.novel_compressor.split(novel_text)
        for idx, novel_chunk in enumerate(novel_chunks):
            with open(os.path.join(working_dir_novel, f"novel_chunk_{idx}.txt"), "w", encoding="utf-8") as f:
                f.write(novel_chunk)
        _pipeline_print(quiet, f"Split novel into {len(novel_chunks)} chunks.")

        _emit_text_plan_progress(progress, "compress_novel", "Compressing novel chunks", {"chunk_count": len(novel_chunks)})
        compressed_novel_chunks: list[str | None] = [None] * len(novel_chunks)
        unfinished_pairs = []
        for index, novel_chunk in enumerate(novel_chunks):
            path = os.path.join(working_dir_novel, f"novel_chunk_{index}_compressed.txt")
            if os.path.exists(path):
                compressed_novel_chunks[index] = open(path, "r", encoding="utf-8").read()
            else:
                unfinished_pairs.append((index, novel_chunk))
        if unfinished_pairs:
            sem = asyncio.Semaphore(5)
            outputs = await asyncio.gather(*[
                self.novel_compressor.compress_single_novel_chunk(sem, index, novel_chunk)
                for index, novel_chunk in unfinished_pairs
            ])
            for index, compressed in outputs:
                path = os.path.join(working_dir_novel, f"novel_chunk_{index}_compressed.txt")
                with open(path, "w", encoding="utf-8") as f:
                    f.write(compressed)
                compressed_novel_chunks[index] = compressed

        compressed_path = os.path.join(working_dir_novel, "novel_compressed.txt")
        if os.path.exists(compressed_path):
            compressed_novel = open(compressed_path, "r", encoding="utf-8").read()
        else:
            compressed_novel = self.novel_compressor.aggregate([chunk or "" for chunk in compressed_novel_chunks])
            with open(compressed_path, "w", encoding="utf-8") as f:
                f.write(compressed_novel)

        _emit_text_plan_progress(progress, "extract_events", "Extracting events from compressed novel")
        working_dir_events = os.path.join(self.working_dir, "events")
        os.makedirs(working_dir_events, exist_ok=True)
        extracted_events: list[Event] = []
        event_files = [
            os.path.join(working_dir_events, fname)
            for fname in os.listdir(working_dir_events)
            if fname.startswith("event_") and fname.endswith(".json")
        ]
        for event_path in sorted(event_files, key=_event_file_index):
            with open(event_path, "r", encoding="utf-8") as f:
                extracted_events.append(Event.model_validate(json.load(f)))
        while len(extracted_events) == 0 or not extracted_events[-1].is_last:
            _ensure_extraction_cap(len(extracted_events), MAX_EXTRACTED_EVENTS, "events")
            next_event = self.event_extractor.extract_next_event(
                novel_text=compressed_novel,
                extracted_events=extracted_events,
            )
            event_path = os.path.join(working_dir_events, f"event_{len(extracted_events)}.json")
            with open(event_path, "w", encoding="utf-8") as f:
                json.dump(next_event.model_dump(), f, ensure_ascii=False, indent=4)
            extracted_events.append(next_event)

        _emit_text_plan_progress(progress, "retrieve_chunks", "Retrieving relevant chunks for events", {"event_count": len(extracted_events)})
        working_dir_knowledge_base = os.path.join(self.working_dir, "knowledge_base")
        working_dir_retrieve = os.path.join(self.working_dir, "relevant_chunks")
        os.makedirs(working_dir_knowledge_base, exist_ok=True)
        os.makedirs(working_dir_retrieve, exist_ok=True)
        embeddings = CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings=self.embeddings,
            document_embedding_cache=LocalFileStore(root_path=working_dir_knowledge_base),
            namespace=getattr(self.embeddings, "model", "default"),
            key_encoder="sha256",
        )
        novel_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
        knowledge_chunks = novel_splitter.split_text(novel_text)
        knowledge_base = FAISS.from_texts(texts=knowledge_chunks, embedding=embeddings)
        event_idx_to_relevant_chunk_score_dict: dict[int, dict[str, float]] = {}

        async def retrieve_relevant_chunks(sem, event: Event):
            async with sem:
                relevant: dict[str, float] = {}
                for process in event.process_chain:
                    chunks = knowledge_base.similarity_search(process, k=10)
                    chunk_texts = [chunk.page_content for chunk in chunks if chunk.page_content not in relevant]
                    if not chunk_texts:
                        continue
                    chunk_score_pairs = await self.rerank_model(documents=chunk_texts, query=process, top_n=10)
                    for chunk, score in chunk_score_pairs:
                        if score >= 0.7:
                            relevant[chunk] = relevant.get(chunk, 0.0) + score
                return event.index, relevant

        retrieve_tasks = []
        retrieve_sem = asyncio.Semaphore(10)
        for event in extracted_events:
            chunks_dir = os.path.join(working_dir_retrieve, f"event_{event.index}")
            if os.path.exists(chunks_dir) and os.listdir(chunks_dir):
                relevant = {}
                for chunk_fname in os.listdir(chunks_dir):
                    chunk_path = os.path.join(chunks_dir, chunk_fname)
                    score = float(chunk_fname.split("-score_")[1].split(".txt")[0])
                    with open(chunk_path, "r", encoding="utf-8") as f:
                        relevant[f.read()] = score
                event_idx_to_relevant_chunk_score_dict[event.index] = relevant
            else:
                retrieve_tasks.append(retrieve_relevant_chunks(retrieve_sem, event))
        if retrieve_tasks:
            for event_index, relevant in await asyncio.gather(*retrieve_tasks):
                chunks_dir = os.path.join(working_dir_retrieve, f"event_{event_index}")
                os.makedirs(chunks_dir, exist_ok=True)
                for idx, (chunk, score) in enumerate(relevant.items()):
                    with open(os.path.join(chunks_dir, f"chunk_{idx}-score_{score:.2f}.txt"), "w", encoding="utf-8") as f:
                        f.write(chunk)
                event_idx_to_relevant_chunk_score_dict[event_index] = relevant

        _emit_text_plan_progress(progress, "extract_scenes", "Extracting screenplay scenes", {"event_count": len(extracted_events)})
        working_dir_scenes = os.path.join(self.working_dir, "scenes")
        os.makedirs(working_dir_scenes, exist_ok=True)
        event_idx_to_scenes: dict[int, list[Scene]] = {event.index: [] for event in extracted_events}
        unfinished_events: list[Event] = []
        for event in extracted_events:
            scenes_dir = os.path.join(working_dir_scenes, f"event_{event.index}")
            if os.path.exists(scenes_dir):
                scene_files = [
                    os.path.join(scenes_dir, fname)
                    for fname in os.listdir(scenes_dir)
                    if fname.startswith("scene_") and fname.endswith(".json")
                ]
                for scene_path in sorted(scene_files, key=_scene_file_index):
                    with open(scene_path, "r", encoding="utf-8") as f:
                        event_idx_to_scenes[event.index].append(Scene.model_validate(json.load(f)))
            if not event_idx_to_scenes[event.index] or not event_idx_to_scenes[event.index][-1].is_last:
                unfinished_events.append(event)

        async def extract_scenes_for_event(sem, event: Event, previous_scenes: list[Scene]):
            async with sem:
                scenes_dir = os.path.join(working_dir_scenes, f"event_{event.index}")
                os.makedirs(scenes_dir, exist_ok=True)
                while len(previous_scenes) == 0 or not previous_scenes[-1].is_last:
                    _ensure_extraction_cap(len(previous_scenes), MAX_SCENES_PER_EVENT, "scenes")
                    next_scene = await self.scene_extractor.get_next_scene(
                        relevant_chunks=list(event_idx_to_relevant_chunk_score_dict.get(event.index, {}).keys()),
                        event=event,
                        previous_scenes=previous_scenes,
                    )
                    scene_path = os.path.join(scenes_dir, f"scene_{len(previous_scenes)}.json")
                    with open(scene_path, "w", encoding="utf-8") as f:
                        json.dump(next_scene.model_dump(), f, ensure_ascii=False, indent=4)
                    previous_scenes.append(next_scene)
                return event.index, previous_scenes

        if unfinished_events:
            sem = asyncio.Semaphore(8)
            scene_outputs = await asyncio.gather(*[
                extract_scenes_for_event(sem, event, event_idx_to_scenes[event.index])
                for event in unfinished_events
            ])
            for event_index, scenes in scene_outputs:
                event_idx_to_scenes[event_index] = scenes

        _emit_text_plan_progress(progress, "merge_characters", "Merging scene characters into novel-level characters", {"event_count": len(extracted_events)})
        working_dir_global = os.path.join(self.working_dir, "global_information")
        working_dir_characters = os.path.join(working_dir_global, "characters")
        os.makedirs(working_dir_characters, exist_ok=True)
        event_idx_to_characters_in_event: dict[int, list[CharacterInEvent]] = {}

        async def merge_event_characters(sem, event: Event):
            async with sem:
                characters = await self.global_information_planner.merge_characters_across_scenes_in_event(
                    event_idx=event.index,
                    scenes=event_idx_to_scenes[event.index],
                )
                path = os.path.join(working_dir_characters, "event_level", f"event_{event.index}_characters.json")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    json.dump([char.model_dump() for char in characters], f, ensure_ascii=False, indent=4)
                return event.index, characters

        merge_tasks = []
        merge_sem = asyncio.Semaphore(8)
        for event in extracted_events:
            path = os.path.join(working_dir_characters, "event_level", f"event_{event.index}_characters.json")
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    event_idx_to_characters_in_event[event.index] = [CharacterInEvent.model_validate(item) for item in json.load(f)]
            else:
                merge_tasks.append(merge_event_characters(merge_sem, event))
        if merge_tasks:
            for event_index, characters in await asyncio.gather(*merge_tasks):
                event_idx_to_characters_in_event[event_index] = characters

        working_dir_novel_chars = os.path.join(working_dir_characters, "novel_level")
        os.makedirs(working_dir_novel_chars, exist_ok=True)
        existing_files = [fname for fname in os.listdir(working_dir_novel_chars) if fname.startswith("novel_characters_after_event_") and fname.endswith(".json")]
        if existing_files:
            latest = max(existing_files, key=lambda fname: int(fname.split("_")[-1].split(".json")[0]))
            start_event_idx = int(latest.split("_")[-1].split(".json")[0]) + 1
            with open(os.path.join(working_dir_novel_chars, latest), "r", encoding="utf-8") as f:
                characters_in_novel = [CharacterInNovel.model_validate(item) for item in json.load(f)]
        else:
            start_event_idx = 0
            characters_in_novel = []
        for event in extracted_events[start_event_idx:]:
            characters_in_novel = self.global_information_planner.merge_characters_to_existing_characters_in_novel(
                event_idx=event.index,
                existing_characters_in_novel=characters_in_novel,
                characters_in_event=event_idx_to_characters_in_event[event.index],
            )
            path = os.path.join(working_dir_novel_chars, f"novel_characters_after_event_{event.index}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump([char.model_dump() for char in characters_in_novel], f, ensure_ascii=False, indent=4)

        _emit_text_plan_progress(progress, "completed", "Novel structured text planning complete", {"event_count": len(extracted_events)})
        return {
            "compressed_novel": compressed_novel,
            "events": extracted_events,
            "scenes": event_idx_to_scenes,
            "characters_in_novel": characters_in_novel,
        }


    async def render_video_artifacts(
        self,
        style: str,
        user_requirement: str = "",
        progress: Callable[[str, str, Dict[str, Any] | None], None] | None = None,
        quiet: bool = False,
    ) -> dict[str, Any]:
        """Render portraits and per-scene videos from existing novel planning artifacts.

        This helper assumes plan_text_artifacts has already completed. It does not
        re-run compression, event extraction, RAG retrieval, scene extraction, or
        character merging.
        """
        del user_requirement

        _emit_text_plan_progress(progress, "novel_render_load", "Loading novel structured text artifacts")
        working_dir_events = os.path.join(self.working_dir, "events")
        working_dir_scenes = os.path.join(self.working_dir, "scenes")
        working_dir_characters = os.path.join(self.working_dir, "global_information", "characters")
        event_level_dir = os.path.join(working_dir_characters, "event_level")
        novel_level_dir = os.path.join(working_dir_characters, "novel_level")

        if not os.path.isdir(working_dir_events):
            raise RuntimeError("novel2video/events is missing; run vimax_novel_planning first")
        if not os.path.isdir(working_dir_scenes):
            raise RuntimeError("novel2video/scenes is missing; run vimax_novel_planning first")
        if not os.path.isdir(event_level_dir) or not os.path.isdir(novel_level_dir):
            raise RuntimeError("novel2video/global_information/characters is missing; run vimax_novel_planning first")

        event_files = [
            os.path.join(working_dir_events, fname)
            for fname in os.listdir(working_dir_events)
            if fname.startswith("event_") and fname.endswith(".json")
        ]
        extracted_events = []
        for event_path in sorted(event_files, key=_event_file_index):
            with open(event_path, "r", encoding="utf-8") as f:
                extracted_events.append(Event.model_validate(json.load(f)))
        if not extracted_events:
            raise RuntimeError("novel2video/events has no event_*.json files")

        event_idx_to_scenes: dict[int, list[Scene]] = {}
        for event in extracted_events:
            scenes_dir = os.path.join(working_dir_scenes, f"event_{event.index}")
            if not os.path.isdir(scenes_dir):
                raise RuntimeError(f"novel2video/scenes/event_{event.index} is missing")
            scene_files = [
                os.path.join(scenes_dir, fname)
                for fname in os.listdir(scenes_dir)
                if fname.startswith("scene_") and fname.endswith(".json")
            ]
            scenes = []
            for scene_path in sorted(scene_files, key=_scene_file_index):
                with open(scene_path, "r", encoding="utf-8") as f:
                    scenes.append(Scene.model_validate(json.load(f)))
            if not scenes:
                raise RuntimeError(f"novel2video/scenes/event_{event.index} has no scene_*.json files")
            event_idx_to_scenes[event.index] = scenes

        event_idx_to_characters_in_event: dict[int, list[CharacterInEvent]] = {}
        for event in extracted_events:
            path = os.path.join(event_level_dir, f"event_{event.index}_characters.json")
            if not os.path.exists(path):
                raise RuntimeError(f"novel2video/global_information/characters/event_level/event_{event.index}_characters.json is missing")
            with open(path, "r", encoding="utf-8") as f:
                event_idx_to_characters_in_event[event.index] = [CharacterInEvent.model_validate(item) for item in json.load(f)]

        novel_files = [fname for fname in os.listdir(novel_level_dir) if fname.startswith("novel_characters_after_event_") and fname.endswith(".json")]
        if not novel_files:
            raise RuntimeError("novel2video/global_information/characters/novel_level has no novel characters file")
        latest_novel_file = max(novel_files, key=lambda fname: int(fname.split("_")[-1].split(".json")[0]))
        with open(os.path.join(novel_level_dir, latest_novel_file), "r", encoding="utf-8") as f:
            characters_in_novel = [CharacterInNovel.model_validate(item) for item in json.load(f)]

        _emit_text_plan_progress(progress, "novel_portraits_start", "Generating novel character portraits", {"character_count": len(characters_in_novel)})
        working_dir_character_portrait = os.path.join(self.working_dir, "character_portraits")
        base_character_portrait_dir = os.path.join(working_dir_character_portrait, "base")
        os.makedirs(base_character_portrait_dir, exist_ok=True)

        async def generate_base_portrait(sem, character: CharacterInNovel):
            async with sem:
                image_path = os.path.join(base_character_portrait_dir, f"character_{character.index}_{safe_path_component(character.identifier_in_novel)}.png")
                if os.path.exists(image_path):
                    return image_path
                prompt = f"Generate a full-body, front-view portrait based on the following description, in the style of {style}:"
                prompt += f"\nCharacter Identifier: {character.identifier_in_novel}"
                prompt += f"\nFeatures: {character.static_features}"
                prompt += "\nThe character should be centered in the image, occupying most of the frame. Gazing straight ahead. Standing with arms relaxed at sides. Natural expression. The background should be plain white."
                image = await self.image_generator.generate_single_image(prompt=prompt, size="512x512")
                image.save(image_path)
                return image_path

        sem = asyncio.Semaphore(5)
        await asyncio.gather(*[generate_base_portrait(sem, character) for character in characters_in_novel])
        _emit_text_plan_progress(progress, "novel_portraits_base_done", "Base character portraits ready", {"character_count": len(characters_in_novel)})

        async def generate_scene_portrait(sem, base_character_image_path: str, character: CharacterInScene, event_idx: int, scene_idx: int):
            async with sem:
                image_path = os.path.join(working_dir_character_portrait, f"event_{event_idx}", f"scene_{scene_idx}", f"character_{character.idx}_{safe_path_component(character.identifier_in_scene)}.png")
                os.makedirs(os.path.dirname(image_path), exist_ok=True)
                if os.path.exists(image_path):
                    return image_path
                if not character.is_visible or character.dynamic_features is None:
                    shutil.copy(base_character_image_path, image_path)
                    return image_path
                prompt = f"Generate a full-body, front-view portrait based on the provided base image. Modify the base image according to the following dynamic features, in the style of {style}. Keep the character's identity consistent with the base image:"
                prompt += f"\nCharacter Identifier: {character.identifier_in_scene}"
                prompt += f"\nDynamic Features: {character.dynamic_features}"
                prompt += "\nThe character should be centered in the image, occupying most of the frame. Gazing straight ahead. Standing with arms relaxed at sides. Natural expression. The background should be plain white."
                prompt = await self.rewriter(prompt)
                image = await self.image_generator.generate_single_image(prompt=prompt, reference_image_paths=[base_character_image_path], size="512x512")
                image.save(image_path)
                return image_path

        _emit_text_plan_progress(progress, "novel_portraits_scene_start", "Generating scene character portraits")
        scene_portrait_tasks = []
        sem = asyncio.Semaphore(3)
        for character in characters_in_novel:
            base_path = os.path.join(base_character_portrait_dir, f"character_{character.index}_{safe_path_component(character.identifier_in_novel)}.png")
            for event_idx, identifier_in_event in character.active_events.items():
                event_characters = event_idx_to_characters_in_event[int(event_idx)]
                character_in_event = [char for char in event_characters if char.identifier_in_event == identifier_in_event][0]
                for scene_idx, identifier_in_scene in character_in_event.active_scenes.items():
                    scene = event_idx_to_scenes[int(event_idx)][int(scene_idx)]
                    character_in_scene = [char for char in scene.characters if char.identifier_in_scene == identifier_in_scene][0]
                    scene_portrait_tasks.append(generate_scene_portrait(sem, base_path, character_in_scene, int(event_idx), int(scene_idx)))
        if scene_portrait_tasks:
            await asyncio.gather(*scene_portrait_tasks)
        _emit_text_plan_progress(progress, "novel_portraits_done", "Scene character portraits ready")

        working_dir_scene_videos = os.path.join(self.working_dir, "videos")
        os.makedirs(working_dir_scene_videos, exist_ok=True)
        scene_video_dirs: list[str] = []
        for event in extracted_events:
            for scene in event_idx_to_scenes[event.index]:
                scene_video_dir = os.path.join(working_dir_scene_videos, f"event_{event.index}", f"scene_{scene.idx}")
                os.makedirs(scene_video_dir, exist_ok=True)
                self.script2video_pipeline.working_dir = scene_video_dir
                character_portraits_registry = {}
                for character in scene.characters:
                    character_portraits_registry[character.identifier_in_scene] = {
                        "portrait": {
                            "path": os.path.join(working_dir_character_portrait, f"event_{event.index}", f"scene_{scene.idx}", f"character_{character.idx}_{safe_path_component(character.identifier_in_scene)}.png"),
                            "description": f"A portrait of {character.identifier_in_scene}",
                        }
                    }
                _emit_text_plan_progress(progress, "novel_scene_render_start", "Rendering novel scene video", {"event_idx": event.index, "scene_idx": scene.idx})
                await self.script2video_pipeline(
                    script=scene.script,
                    user_requirement="",
                    style=style or "realistic movie style",
                    characters=scene.characters,
                    character_portraits_registry=character_portraits_registry,
                    quiet=quiet,
                    progress=progress,
                )
                scene_video_dirs.append(scene_video_dir)
                _emit_text_plan_progress(progress, "novel_scene_render_done", "Rendered novel scene video", {"event_idx": event.index, "scene_idx": scene.idx, "path": scene_video_dir})

        _emit_text_plan_progress(progress, "novel_render_completed", "Novel scene render complete", {"scene_count": len(scene_video_dirs)})
        return {
            "character_portraits_dir": working_dir_character_portrait,
            "scene_videos_dir": working_dir_scene_videos,
            "scene_video_dirs": scene_video_dirs,
            "scene_count": len(scene_video_dirs),
        }

    async def __call__(
        self,
        novel_text: str,
        style: str,
    ):
        print("🎬 Novel to Movie Pipeline Started".center(80, "="))
        await self.plan_text_artifacts(novel_text)
        return await self.render_video_artifacts(style=style)


# is_last flags are asserted by the LLM only; cap the extraction loops so a
# model that never sets one cannot spend tokens forever.
MAX_EXTRACTED_EVENTS = 50
MAX_SCENES_PER_EVENT = 30


def _ensure_extraction_cap(count: int, cap: int, what: str) -> None:
    if count >= cap:
        raise RuntimeError(
            f"Extraction reached {count} {what} without an is_last marker (cap: {cap}); "
            "aborting to avoid unbounded LLM calls."
        )
