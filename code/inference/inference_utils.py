"""Dependency-free validation helpers for the preview inference script."""

import re


def validate_prompt_args(args):
    if args.use_audio_prompt and not args.audio_prompt_path:
        raise ValueError("--use_audio_prompt requires --audio_prompt_path")
    if args.use_dual_tracks_prompt and not (
        args.vocal_track_prompt_path and args.instrumental_track_prompt_path
    ):
        raise ValueError(
            "--use_dual_tracks_prompt requires both --vocal_track_prompt_path "
            "and --instrumental_track_prompt_path"
        )
    if args.run_n_segments <= 0 or args.stage2_batch_size <= 0:
        raise ValueError("--run_n_segments and --stage2_batch_size must be positive")
    if not 100 <= args.max_new_tokens < 16383:
        raise ValueError("--max_new_tokens must be between 100 and 16382")
    if args.prompt_start_time < 0 or args.prompt_end_time <= args.prompt_start_time:
        raise ValueError("Audio prompt times must satisfy 0 <= start < end")


def split_lyrics(lyrics):
    segments = re.findall(r"\[(\w+)\](.*?)(?=\[|\Z)", lyrics, re.DOTALL)
    return [f"[{name}]\n{text.strip()}\n\n" for name, text in segments if text.strip()]


def prompt_count(lyrics, requested_segments):
    """Include the instruction prefix as well as every requested section."""
    if requested_segments <= 0:
        raise ValueError("The requested number of sections must be positive")
    if not lyrics:
        raise ValueError("Lyrics must contain a nonempty section such as [verse]")
    return min(requested_segments, len(lyrics)) + 1


def filename_tag(genres):
    """Keep a user-supplied tag within a single, bounded filename component."""
    return re.sub(r"[^\w-]+", "-", genres).strip("-")[:80] or "music"


def create_codec_model(config, generators):
    """Resolve only explicitly supported model names."""
    try:
        generator = generators[config.name]
    except KeyError:
        raise ValueError(f"Unsupported codec generator: {config.name!r}") from None
    return generator(**config.config)
