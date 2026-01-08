import base64
import logging
import os
from pathlib import Path
import tempfile
import threading
import typing as tp

import torch

import runpod
from audiocraft.data.audio import audio_write
from audiocraft.models import AudioGen, MusicGen


logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("runpod_audiocraft")

DEFAULT_TASK = os.getenv("DEFAULT_TASK", "music")
DEFAULT_MUSIC_MODEL = os.getenv("DEFAULT_MUSIC_MODEL", "facebook/musicgen-medium")
DEFAULT_SFX_MODEL = os.getenv("DEFAULT_SFX_MODEL", "facebook/audiogen-medium")
DEFAULT_OUTPUT_FORMAT = os.getenv("DEFAULT_OUTPUT_FORMAT", "wav")
DEFAULT_DURATION = float(os.getenv("DEFAULT_DURATION", "8"))
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "250"))
DEFAULT_TOP_P = float(os.getenv("DEFAULT_TOP_P", "0"))
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "1.0"))
DEFAULT_CFG_COEF = float(os.getenv("DEFAULT_CFG_COEF", "3.0"))
DEFAULT_LOUDNESS_DB = float(os.getenv("DEFAULT_LOUDNESS_DB", "16"))
DEFAULT_LOUDNESS_COMPRESSOR = os.getenv("DEFAULT_LOUDNESS_COMPRESSOR", "true").lower() in (
    "1",
    "true",
    "yes",
    "on",
)

MAX_BATCH = int(os.getenv("MAX_BATCH", "8"))
MAX_DURATION = float(os.getenv("MAX_DURATION", "120"))

DEVICE = os.getenv("MODEL_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")

MODEL_CACHE: tp.Dict[tp.Tuple[str, str], tp.Any] = {}
MODEL_LOCKS: tp.Dict[tp.Tuple[str, str], threading.Lock] = {}

ALLOWED_FORMATS = {"wav", "mp3", "ogg", "flac"}


def _parse_bool(value: tp.Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


def _normalize_task(task: tp.Optional[str]) -> tp.Optional[str]:
    if not task:
        return None
    value = task.strip().lower()
    if value in {"music", "bgm", "background"}:
        return "music"
    if value in {"sfx", "sound", "audio", "fx", "soundfx"}:
        return "sfx"
    return None


def _resolve_task(task: tp.Optional[str], model_name: tp.Optional[str]) -> str:
    normalized = _normalize_task(task)
    if normalized:
        return normalized
    if model_name:
        lower_name = model_name.lower()
        if "audiogen" in lower_name:
            return "sfx"
        if "musicgen" in lower_name:
            return "music"
    default_task = _normalize_task(DEFAULT_TASK)
    return default_task or "music"


def _resolve_model_name(task: str, model_name: tp.Optional[str]) -> str:
    if model_name:
        return model_name
    if task == "music":
        return DEFAULT_MUSIC_MODEL
    return DEFAULT_SFX_MODEL


def _get_model(task: str, model_name: str):
    key = (task, model_name)
    model = MODEL_CACHE.get(key)
    if model is None:
        logger.info("Loading model %s for task %s on %s", model_name, task, DEVICE)
        if task == "music":
            model = MusicGen.get_pretrained(model_name, device=DEVICE)
        else:
            model = AudioGen.get_pretrained(model_name, device=DEVICE)
        model.eval()
        MODEL_CACHE[key] = model
        MODEL_LOCKS[key] = threading.Lock()
    return model, MODEL_LOCKS[key]


def _ensure_prompts(job_input: tp.Dict[str, tp.Any]) -> tp.List[str]:
    prompts = job_input.get("prompts") or job_input.get("descriptions")
    if prompts is None:
        prompt = job_input.get("prompt") or job_input.get("text")
        if prompt is None:
            raise ValueError("Missing 'prompt' or 'prompts' in input.")
        prompts = [prompt]
    if isinstance(prompts, str):
        prompts = [prompts]
    if not isinstance(prompts, list):
        raise ValueError("'prompts' must be a list of strings.")
    prompts = [str(prompt) for prompt in prompts]
    if not prompts:
        raise ValueError("'prompts' must not be empty.")
    num_outputs = job_input.get("num_outputs")
    if num_outputs is not None:
        num_outputs = int(num_outputs)
        if num_outputs < 1 or num_outputs > MAX_BATCH:
            raise ValueError(f"'num_outputs' must be between 1 and {MAX_BATCH}.")
        if len(prompts) == 1:
            prompts = prompts * num_outputs
        elif len(prompts) != num_outputs:
            raise ValueError("'num_outputs' must match 'prompts' length when multiple prompts are provided.")
    if len(prompts) > MAX_BATCH:
        raise ValueError(f"Too many prompts. Max batch size is {MAX_BATCH}.")
    return prompts


def _write_audio_bytes(
    wav: torch.Tensor,
    sample_rate: int,
    output_format: str,
    normalize: bool,
    strategy: str,
    loudness_headroom_db: float,
    loudness_compressor: bool,
) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=f".{output_format}", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        audio_write(
            tmp_path,
            wav,
            sample_rate,
            format=output_format,
            normalize=normalize,
            strategy=strategy,
            loudness_headroom_db=loudness_headroom_db,
            loudness_compressor=loudness_compressor,
            add_suffix=False,
        )
        return tmp_path.read_bytes()
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass


def handler(job: tp.Dict[str, tp.Any]) -> tp.Dict[str, tp.Any]:
    job_input = job.get("input", {}) or {}
    try:
        model_name = job_input.get("model") or job_input.get("model_name")
        task = _resolve_task(job_input.get("task"), model_name)
        model_name = _resolve_model_name(task, model_name)

        prompts = _ensure_prompts(job_input)

        duration = float(job_input.get("duration", DEFAULT_DURATION))
        if duration <= 0 or duration > MAX_DURATION:
            raise ValueError(f"'duration' must be between 0 and {MAX_DURATION} seconds.")

        output_format = str(job_input.get("format", DEFAULT_OUTPUT_FORMAT)).lower()
        if output_format not in ALLOWED_FORMATS:
            raise ValueError(f"Unsupported output format '{output_format}'.")

        temperature = float(job_input.get("temperature", DEFAULT_TEMPERATURE))
        top_k = int(job_input.get("top_k", DEFAULT_TOP_K))
        top_p = float(job_input.get("top_p", DEFAULT_TOP_P))
        cfg_coef = float(job_input.get("cfg_coef", DEFAULT_CFG_COEF))
        use_sampling = _parse_bool(job_input.get("use_sampling"), True)
        two_step_cfg = _parse_bool(job_input.get("two_step_cfg"), False)
        extend_stride = job_input.get("extend_stride")

        normalize = _parse_bool(job_input.get("normalize"), True)
        normalize_strategy = str(job_input.get("normalize_strategy", "loudness"))
        loudness_headroom_db = float(job_input.get("loudness_headroom_db", DEFAULT_LOUDNESS_DB))
        loudness_compressor = _parse_bool(
            job_input.get("loudness_compressor"), DEFAULT_LOUDNESS_COMPRESSOR
        )

        seed = job_input.get("seed")
        if seed is not None:
            seed = int(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        model, lock = _get_model(task, model_name)

        gen_params = {
            "use_sampling": use_sampling,
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "duration": duration,
            "cfg_coef": cfg_coef,
            "two_step_cfg": two_step_cfg,
        }
        if extend_stride is not None:
            gen_params["extend_stride"] = float(extend_stride)
        if task == "music":
            cfg_coef_beta = job_input.get("cfg_coef_beta")
            if cfg_coef_beta is not None:
                gen_params["cfg_coef_beta"] = float(cfg_coef_beta)

        with lock:
            model.set_generation_params(**gen_params)
            with torch.inference_mode():
                wavs = model.generate(prompts, progress=False)

        outputs = []
        for wav in wavs:
            audio_bytes = _write_audio_bytes(
                wav,
                model.sample_rate,
                output_format,
                normalize,
                normalize_strategy,
                loudness_headroom_db,
                loudness_compressor,
            )
            outputs.append(
                {
                    "audio": base64.b64encode(audio_bytes).decode("utf-8"),
                    "format": output_format,
                    "sample_rate": model.sample_rate,
                    "channels": int(wav.shape[0]) if wav.dim() > 1 else 1,
                }
            )

        return {"task": task, "model": model_name, "outputs": outputs}
    except Exception as exc:
        logger.exception("Handler failed")
        return {"error": str(exc)}


runpod.serverless.start({"handler": handler})
