# RunPod serverless deployment

This repo includes a minimal RunPod serverless handler for text-to-music (MusicGen) and text-to-sound (AudioGen).

## Quick start

1. Fork the repo on GitHub.
2. Build and push the image using `runpod/Dockerfile`.
3. Create a RunPod Serverless endpoint pointing at that image.
4. Send requests to the endpoint with the JSON input below.

## Docker image

The provided Dockerfile installs system deps (ffmpeg, libsndfile1), the RunPod SDK, and AudioCraft.

```bash
# Example local build
docker build -f runpod/Dockerfile -t audiocraft-runpod .
```

## Handler entrypoint

The container starts `runpod/handler.py`. It loads models on first request and caches them in memory.

## Input schema

Required:
- `prompt` (string) or `prompts` (list of strings)

Optional:
- `task`: `music` or `sfx` (defaults to `DEFAULT_TASK`)
- `model`: HF model id (defaults to `DEFAULT_MUSIC_MODEL` or `DEFAULT_SFX_MODEL`)
- `duration`: seconds, max controlled by `MAX_DURATION`
- `num_outputs`: int, duplicates a single prompt into a batch
- `temperature`, `top_k`, `top_p`, `cfg_coef`
- `cfg_coef_beta`: MusicGen only
- `two_step_cfg`: bool
- `extend_stride`: float for long generations
- `format`: `wav`, `mp3`, `ogg`, `flac`
- `seed`: int for reproducibility
- `normalize`: bool
- `normalize_strategy`: `loudness`, `peak`, `rms`, `clip`
- `loudness_headroom_db`: float
- `loudness_compressor`: bool

## Output schema

```json
{
  "task": "music",
  "model": "facebook/musicgen-medium",
  "outputs": [
    {
      "audio": "<base64>",
      "format": "wav",
      "sample_rate": 32000,
      "channels": 1
    }
  ]
}
```

## Example request payloads

Background music:

```json
{
  "input": {
    "task": "music",
    "prompt": "dreamy ambient pad with soft piano",
    "duration": 15,
    "format": "wav"
  }
}
```

SFX:

```json
{
  "input": {
    "task": "sfx",
    "prompt": "short metallic impact with reverb tail",
    "duration": 3,
    "format": "wav"
  }
}
```

## Environment variables

- `DEFAULT_TASK`: `music` or `sfx`
- `DEFAULT_MUSIC_MODEL`: default MusicGen id
- `DEFAULT_SFX_MODEL`: default AudioGen id
- `DEFAULT_OUTPUT_FORMAT`: `wav`/`mp3`/`ogg`/`flac`
- `DEFAULT_DURATION`, `DEFAULT_TOP_K`, `DEFAULT_TOP_P`, `DEFAULT_TEMPERATURE`, `DEFAULT_CFG_COEF`
- `DEFAULT_LOUDNESS_DB`, `DEFAULT_LOUDNESS_COMPRESSOR`
- `MAX_BATCH`, `MAX_DURATION`
- `MODEL_DEVICE`: `cuda` or `cpu`
- `AUDIOCRAFT_CACHE_DIR`: cache location for model weights

## License note

AudioCraft weights are under CC-BY-NC 4.0. Ensure your usage complies with the non-commercial terms.
