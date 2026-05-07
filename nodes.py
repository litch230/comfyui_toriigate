import random
import sys
import threading
import time
from pathlib import Path
from queue import Empty

import numpy as np
import torch
from PIL import Image

from .prompts import make_user_query, prompts_b, prompts_names_only, system_prompt


CAPTION_TYPES = [
    "long_thoughts_v2",
    "long_thoughts",
    "json",
    "long",
    "min_structured_md",
    "json_comic",
    "md_comic",
    "min_structured_json",
    "chroma-style",
    "short",
]


_BANNER_PRINTED = set()


TORIIGATE_BANNER = r"""
======================================================================

  TTTTT  OOO  RRRR   III  III   GGG    A    TTTTT EEEEE
    T   O   O R   R   I    I   G      A A     T   E
    T   O   O RRRR    I    I   G  GG AAAAA    T   EEEE
    T   O   O R  R    I    I   G   G A   A    T   E
    T    OOO  R   R  III  III   GGG  A   A    T   EEEEE

                         ToriiGate-0.5
======================================================================
"""


MODEL_NOTES = [
    "Digital art/anime image captioning model.",
    "Base: Qwen3.5-4B-Base fine-tuned for image-to-text captioning.",
    "Recommended image budget: up to about 1 MP.",
    "Designed for single-image captioning, not general chat or multi-turn use.",
    "Uses special ToriiGate prompts from the official Gradio Space.",
]


def _empty_grounding():
    return {
        "tags": [],
        "characters": [],
        "char_p_tags": {"chars": {}, "skins": {}},
        "char_descr": {"chars": {}, "skins": {}},
    }


def _split_csv(value):
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _resolve_device(device):
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print("[ToriiGate] CUDA requested but unavailable; falling back to CPU.")
        return "cpu"
    return device


def _resolve_dtype(dtype):
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return dtype_map.get(dtype, torch.bfloat16)


def _print_model_intro(model_path, device, dtype, max_pixels_mp):
    if model_path in _BANNER_PRINTED:
        return
    _BANNER_PRINTED.add(model_path)
    print(TORIIGATE_BANNER)
    print(f"[ToriiGate] Model path : {model_path}")
    print(f"[ToriiGate] Device     : {device}")
    print(f"[ToriiGate] DType      : {dtype}")
    print(f"[ToriiGate] Max pixels : {max_pixels_mp:.2f} MP")
    for note in MODEL_NOTES:
        print(f"[ToriiGate] - {note}")
    print("[ToriiGate] First run may download several GB from Hugging Face.")
    print("[ToriiGate] Download progress is shown by the Hugging Face Hub.\n")


def _is_local_model_path(model_path):
    return Path(model_path).expanduser().exists()


def _pre_download_model(model_path):
    if _is_local_model_path(model_path):
        print("[ToriiGate] Local model path detected; skipping Hub download check.")
        return model_path

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[ToriiGate] huggingface_hub not available; transformers will download the model.")
        return model_path

    print(f"[ToriiGate] Checking/downloading model snapshot: {model_path}")
    print("[ToriiGate] If files are missing, a progress bar should appear below.")
    snapshot_path = snapshot_download(
        repo_id=model_path,
        local_files_only=False,
    )
    print(f"[ToriiGate] Snapshot ready: {snapshot_path}")
    return snapshot_path


def _cuda_memory_status(prefix="[ToriiGate]"):
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    print(f"{prefix} CUDA memory: allocated {allocated:.2f} GB | reserved {reserved:.2f} GB")


def _model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _ensure_model_on_device(model, device):
    current_device = _model_device(model)
    target_device = torch.device(device)
    if current_device != target_device:
        print(f"[ToriiGate] Moving model from {current_device} to {target_device}.")
        start_time = time.perf_counter()
        model.to(target_device)
        model.eval()
        elapsed = time.perf_counter() - start_time
        print(f"[ToriiGate] Model ready on {target_device} in {elapsed:.1f}s.")
        _cuda_memory_status()


def _unload_model_from_vram(model, cache_key=None):
    print("[ToriiGate] keep_model_alive is disabled; unloading model from VRAM.")
    if cache_key is not None:
        ToriiGateModelLoader._cache.pop(cache_key, None)
    try:
        model.to("cpu")
    except Exception as exc:
        print(f"[ToriiGate] Warning: could not move model to CPU before unload: {exc}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
        _cuda_memory_status()


class ToriiGateModelLoader:
    _cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "model_path": (
                    "STRING",
                    {
                        "default": "Minthy/ToriiGate-0.5",
                        "tooltip": "Model path. Use Minthy/ToriiGate-0.5 to download from Hugging Face, or a local path for an already downloaded copy.",
                    },
                ),
                "device": (
                    ["cuda", "cpu", "auto"],
                    {
                        "default": "cuda",
                        "tooltip": "Where to run the model. auto uses CUDA when available; cuda is recommended for speed; cpu works, but will be very slow.",
                    },
                ),
                "dtype": (
                    ["bfloat16", "float16", "float32"],
                    {
                        "default": "bfloat16",
                        "tooltip": "Weight precision. bfloat16 is the native/recommended format; float16 may save VRAM on some GPUs; float32 uses more memory and is usually slower.",
                    },
                ),
                "max_pixels_mp": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 8.0,
                        "step": 0.1,
                        "tooltip": "Resolution limit sent to the model, in megapixels. 1.0 MP is recommended; higher values may improve detail, but increase VRAM use and runtime.",
                    },
                ),
                "keep_model_alive": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Keeps the model loaded in VRAM/cache after generation. Enable for repeated captions; disable to free VRAM after each run.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("TORIIGATE_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "ToriiGate/Loaders"

    def load_model(
        self,
        model_path="Minthy/ToriiGate-0.5",
        device="cuda",
        dtype="bfloat16",
        max_pixels_mp=1.0,
        keep_model_alive=False,
    ):
        resolved_device = _resolve_device(device)
        torch_dtype = _resolve_dtype(dtype)
        cache_key = (model_path, resolved_device, dtype)

        _print_model_intro(model_path, resolved_device, dtype, float(max_pixels_mp))

        if cache_key not in self._cache:
            try:
                from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
            except ImportError as exc:
                raise ImportError(
                    "ToriiGate requires a recent transformers build with "
                    "Qwen3_5ForConditionalGeneration. Install requirements.txt "
                    "or upgrade transformers."
                ) from exc

            load_path = _pre_download_model(model_path)
            print(
                f"[ToriiGate] Loading model '{load_path}' on {resolved_device} "
                f"with dtype {dtype}."
            )
            start_time = time.perf_counter()
            model = Qwen3_5ForConditionalGeneration.from_pretrained(
                load_path,
                torch_dtype=torch_dtype,
                attn_implementation="sdpa",
            )
            model.to(resolved_device)
            model.eval()

            processor = AutoProcessor.from_pretrained(
                load_path,
                min_pixels=256 * 32 * 32,
                padding_side="right",
            )
            elapsed = time.perf_counter() - start_time
            print(f"[ToriiGate] Model loaded in {elapsed:.1f}s.")
            _cuda_memory_status()
            self._cache[cache_key] = (model, processor, resolved_device)
        else:
            print(f"[ToriiGate] Reusing cached model '{model_path}'.")

        model, processor, resolved_device = self._cache[cache_key]
        return (
            (
                model,
                processor,
                resolved_device,
                float(max_pixels_mp),
                bool(keep_model_alive),
                cache_key,
            ),
        )


class ToriiGateGroundingBuilder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tags": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "General booru tags for the image, separated by commas. Example: 1girl, blue_hair, school_uniform. These are only added to the prompt when add_tags is enabled in the Captioner.",
                    },
                ),
                "character_names": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "General character list, separated by commas. If char1_name through char5_name are empty, their tag/description slots are matched to this list by position.",
                    },
                ),
                "char1_name": ("STRING", {"default": "", "tooltip": "Name/tag for character 1. Example: hoshimi_miyabi."}),
                "char1_tags": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Booru tags specific to character 1, separated by commas. Uses char1_name, or the first name from character_names if char1_name is empty.",
                    },
                ),
                "char2_name": ("STRING", {"default": "", "tooltip": "Name/tag for character 2."}),
                "char2_tags": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Booru tags specific to character 2, separated by commas. Uses char2_name, or the second name from character_names if char2_name is empty.",
                    },
                ),
                "char3_name": ("STRING", {"default": "", "tooltip": "Name/tag for character 3."}),
                "char3_tags": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Booru tags specific to character 3, separated by commas. Uses char3_name, or the third name from character_names if char3_name is empty.",
                    },
                ),
                "char4_name": ("STRING", {"default": "", "tooltip": "Name/tag for character 4."}),
                "char4_tags": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Booru tags specific to character 4, separated by commas. Uses char4_name, or the fourth name from character_names if char4_name is empty.",
                    },
                ),
                "char5_name": ("STRING", {"default": "", "tooltip": "Name/tag for character 5."}),
                "char5_tags": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Booru tags specific to character 5, separated by commas. Uses char5_name, or the fifth name from character_names if char5_name is empty.",
                    },
                ),
                "char1_description": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Free-form description for character 1. Useful when tags are not enough, such as visual personality, a specific uniform, or an alternate version.",
                    },
                ),
                "char2_description": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Free-form description for character 2.",
                    },
                ),
                "char3_description": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Free-form description for character 3.",
                    },
                ),
                "char4_description": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Free-form description for character 4.",
                    },
                ),
                "char5_description": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Free-form description for character 5.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("TORIIGATE_GROUNDING",)
    FUNCTION = "build_grounding"
    CATEGORY = "ToriiGate/Grounding"

    def build_grounding(
        self,
        tags="",
        character_names="",
        char1_name="",
        char1_tags="",
        char2_name="",
        char2_tags="",
        char3_name="",
        char3_tags="",
        char4_name="",
        char4_tags="",
        char5_name="",
        char5_tags="",
        char1_description="",
        char2_description="",
        char3_description="",
        char4_description="",
        char5_description="",
    ):
        item = _empty_grounding()
        item["tags"] = _split_csv(tags)
        item["characters"] = _split_csv(character_names)

        char_entries = [
            (char1_name, char1_tags, char1_description),
            (char2_name, char2_tags, char2_description),
            (char3_name, char3_tags, char3_description),
            (char4_name, char4_tags, char4_description),
            (char5_name, char5_tags, char5_description),
        ]

        auto_chars = []
        for index, (raw_name, raw_tags, raw_description) in enumerate(char_entries):
            name = raw_name.strip() if raw_name else ""
            if not name and index < len(item["characters"]):
                name = item["characters"][index]
            if not name:
                continue

            auto_chars.append(name)

            parsed_tags = _split_csv(raw_tags)
            if parsed_tags:
                item["char_p_tags"]["chars"][name] = parsed_tags

            description = raw_description.strip() if raw_description else ""
            if description:
                item["char_descr"]["chars"][name] = description

        if auto_chars and not item["characters"]:
            item["characters"] = auto_chars

        return (item,)


class ToriiGateCaptioner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "toriigate_model": (
                    "TORIIGATE_MODEL",
                    {
                        "tooltip": "Output from ToriiGate Model Loader. Contains the model, processor, device, resolution limit, and VRAM keep/unload policy.",
                    },
                ),
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "ComfyUI image tensor. The node uses only the first image in the batch and converts it to RGB before captioning.",
                    },
                ),
            },
            "optional": {
                "grounding": (
                    "TORIIGATE_GROUNDING",
                    {
                        "tooltip": "Optional output from ToriiGate Grounding Builder with tags, names, and character descriptions to guide the caption.",
                    },
                ),
                "caption_type": (
                    CAPTION_TYPES,
                    {
                        "default": "short",
                        "tooltip": "Caption format. short is fastest; long is detailed natural text; json/min_structured produce structured output; long_thoughts_v2 is the most detailed, but also slower.",
                    },
                ),
                "use_names": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Allows the model to use or try to recognize character names. Recommended for long_thoughts_v2, long_thoughts, md_comic, and min_structured_md.",
                    },
                ),
                "add_tags": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Adds general booru tags from the Grounding Builder to the prompt. Improves precision, but can bias the caption if the tags are wrong.",
                    },
                ),
                "add_character_list": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Explicitly adds the character name list to the prompt. Enable when you already know who appears in the image.",
                    },
                ),
                "add_character_tags": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Adds per-character tags. Useful for distinguishing similar characters, outfits, hair, and accessories.",
                    },
                ),
                "add_character_descriptions": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Adds free-form per-character descriptions. Use this when you want to guide the model with richer text than booru tags.",
                    },
                ),
                "max_new_tokens": (
                    "INT",
                    {
                        "default": 512,
                        "min": 64,
                        "max": 4096,
                        "tooltip": "Maximum generated tokens. Lower this for speed: 256-512 for quick captions; 1024+ for fuller long_thoughts formats.",
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.01,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": "Generation randomness in sample mode. Lower values are more consistent; higher values are more creative, but may invent details.",
                    },
                ),
                "decoding": (
                    ["sample", "greedy_fast"],
                    {
                        "default": "greedy_fast",
                        "tooltip": "sample uses temperature-based sampling and can vary more; greedy_fast chooses deterministic tokens and is usually faster and more stable.",
                    },
                ),
                "show_generation_progress": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Shows a progress bar and tokens/s in the console during generation. Disable to reduce overhead and gain a little speed.",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "tooltip": "Seed for reproducibility. Use 0 for a random seed; use a fixed value to repeat results in sample mode.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("caption",)
    FUNCTION = "caption"
    CATEGORY = "ToriiGate"

    def caption(
        self,
        toriigate_model,
        image,
        grounding=None,
        caption_type="short",
        use_names=True,
        add_tags=True,
        add_character_list=False,
        add_character_tags=False,
        add_character_descriptions=False,
        max_new_tokens=512,
        temperature=0.5,
        decoding="greedy_fast",
        show_generation_progress=False,
        seed=0,
    ):
        if len(toriigate_model) >= 6:
            model, processor, device, max_pixels_mp, keep_model_alive, cache_key = toriigate_model
        else:
            model, processor, device, max_pixels_mp = toriigate_model
            keep_model_alive = True
            cache_key = None

        _ensure_model_on_device(model, device)

        if caption_type not in prompts_b:
            raise ValueError(f"Unknown ToriiGate caption_type: {caption_type}")

        if prompts_names_only.get(caption_type, False) and not use_names:
            print(
                f"[ToriiGate] Warning: caption format '{caption_type}' is "
                "designed for character names, but use_names is disabled."
            )

        img_np = (image[0].detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)

        if img_pil.mode != "RGB":
            img_pil = img_pil.convert("RGB")

        current_pixels = img_pil.width * img_pil.height
        max_pixels_count = float(max_pixels_mp) * 1_000_000
        if current_pixels > max_pixels_count:
            scale = (max_pixels_count / current_pixels) ** 0.5
            new_w = max(1, int(img_pil.width * scale))
            new_h = max(1, int(img_pil.height * scale))
            img_pil = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)

        item = grounding if grounding else _empty_grounding()

        user_query = make_user_query(
            item,
            c_type=caption_type,
            use_names=use_names,
            add_tags=add_tags,
            add_characters=add_character_list,
            add_char_tags=add_character_tags,
            add_description=add_character_descriptions,
            underscores_replace=False,
        )

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_query},
                ],
            },
        ]

        texts = [
            processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        ]
        inputs = processor(text=texts, images=[img_pil], return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        if seed != 0:
            torch.manual_seed(int(seed))
        else:
            torch.manual_seed(random.SystemRandom().randint(1, 2**63 - 1))

        caption = ""
        try:
            if show_generation_progress:
                caption = self._generate_with_progress(
                    model=model,
                    processor=processor,
                    inputs=inputs,
                    max_new_tokens=int(max_new_tokens),
                    temperature=float(temperature),
                    decoding=decoding,
                )
            else:
                caption = self._generate_without_streamer(
                    model=model,
                    processor=processor,
                    inputs=inputs,
                    max_new_tokens=int(max_new_tokens),
                    temperature=float(temperature),
                    decoding=decoding,
                )
            return (caption.strip(),)
        finally:
            if not keep_model_alive:
                _unload_model_from_vram(model, cache_key)

    def _generation_kwargs(self, processor, inputs, max_new_tokens, temperature, decoding):
        tokenizer = getattr(processor, "tokenizer", processor)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
        if eos_token_id is not None:
            kwargs["pad_token_id"] = eos_token_id

        if decoding == "greedy_fast":
            kwargs["do_sample"] = False
        else:
            kwargs["do_sample"] = True
            kwargs["temperature"] = temperature

        return kwargs

    def _generate_with_progress(self, model, processor, inputs, max_new_tokens, temperature, decoding):
        try:
            from transformers import TextIteratorStreamer
        except ImportError:
            return self._generate_without_streamer(
                model,
                processor,
                inputs,
                max_new_tokens,
                temperature,
                decoding,
            )

        tokenizer = getattr(processor, "tokenizer", processor)
        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=0.5,
        )
        generation_error = []

        generation_kwargs = self._generation_kwargs(
            processor=processor,
            inputs=inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            decoding=decoding,
        )
        generation_kwargs["streamer"] = streamer

        def _worker():
            try:
                with torch.inference_mode():
                    model.generate(**generation_kwargs)
            except Exception as exc:
                generation_error.append(exc)

        print(
            f"[ToriiGate] Caption generation started "
            f"(max_new_tokens={max_new_tokens}, decoding={decoding}, "
            f"temperature={temperature:.2f})."
        )
        start_time = time.perf_counter()
        last_print = start_time
        chunks = []
        generated_chars = 0
        approx_tokens = 0
        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

        while thread.is_alive():
            try:
                text = next(streamer)
            except Empty:
                if generation_error:
                    break
                continue
            except StopIteration:
                break
            chunks.append(text)
            generated_chars += len(text)
            try:
                approx_tokens += len(tokenizer.encode(text, add_special_tokens=False))
            except Exception:
                approx_tokens = max(approx_tokens + 1, generated_chars // 4)
            now = time.perf_counter()
            if now - last_print >= 0.5:
                elapsed = max(now - start_time, 0.001)
                toks_per_sec = approx_tokens / elapsed
                progress = min(1.0, approx_tokens / max(max_new_tokens, 1))
                self._print_generation_progress(progress, approx_tokens, toks_per_sec)
                last_print = now

        thread.join()
        if generation_error:
            raise generation_error[0]

        while True:
            try:
                text = next(streamer)
            except (Empty, StopIteration):
                break
            chunks.append(text)
            generated_chars += len(text)
            try:
                approx_tokens += len(tokenizer.encode(text, add_special_tokens=False))
            except Exception:
                approx_tokens = max(approx_tokens + 1, generated_chars // 4)

        elapsed = max(time.perf_counter() - start_time, 0.001)
        toks_per_sec = approx_tokens / elapsed
        self._print_generation_progress(1.0, approx_tokens, toks_per_sec, done=True)
        print(f"\n[ToriiGate] Caption generation finished in {elapsed:.1f}s.")
        return "".join(chunks)

    def _generate_without_streamer(self, model, processor, inputs, max_new_tokens, temperature, decoding):
        print(
            "[ToriiGate] Caption generation started "
            f"(no streaming, decoding={decoding})."
        )
        start_time = time.perf_counter()
        generation_kwargs = self._generation_kwargs(
            processor=processor,
            inputs=inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            decoding=decoding,
        )
        with torch.inference_mode():
            generate_ids = model.generate(
                **generation_kwargs,
            )

        generated_texts = processor.batch_decode(
            generate_ids[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        elapsed = time.perf_counter() - start_time
        new_tokens = max(0, generate_ids.shape[1] - inputs["input_ids"].shape[1])
        speed = new_tokens / max(elapsed, 0.001)
        print(f"[ToriiGate] Caption generation finished in {elapsed:.1f}s ({speed:.2f} tok/s).")
        return generated_texts[0] if generated_texts else ""

    def _print_generation_progress(self, progress, approx_tokens, toks_per_sec, done=False):
        width = 28
        filled = int(width * progress)
        bar = "#" * filled + "-" * (width - filled)
        end = "\n" if done else "\r"
        sys.stdout.write(
            f"[ToriiGate] Generating [{bar}] "
            f"~{approx_tokens} tok | {toks_per_sec:.2f} tok/s"
            f"{' ' * 8}"
        )
        sys.stdout.write(end)
        sys.stdout.flush()


NODE_CLASS_MAPPINGS = {
    "ToriiGate_ModelLoader": ToriiGateModelLoader,
    "ToriiGate_GroundingBuilder": ToriiGateGroundingBuilder,
    "ToriiGate_Captioner": ToriiGateCaptioner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ToriiGate_ModelLoader": "ToriiGate Model Loader",
    "ToriiGate_GroundingBuilder": "ToriiGate Grounding Builder",
    "ToriiGate_Captioner": "ToriiGate Captioner",
}
