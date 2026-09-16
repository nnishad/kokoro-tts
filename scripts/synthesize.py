#!/usr/bin/env python3
"""
Kokoro-ONNX Hinglish TTS CLI (NVIDIA CUDA Optimized)
Usage:
    python synthesize.py "नमस्ते! आपका server 100% normal चल रहा है।" -o output.wav
    python synthesize.py --voice "hm_omega:70,am_adam:30" --speed 1.0 "Text here"
    python synthesize.py --stream "लंबा पैराग्राफ यहाँ..."
"""

import os
import sys
import argparse
import time
import asyncio
import soundfile as sf
import onnxruntime as ort
from kokoro_onnx import Kokoro

# Force CUDA Execution Provider and suppress C++ warnings
os.environ["ONNX_PROVIDER"] = "CUDAExecutionProvider"
ort.set_default_logger_severity(3)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(SCRIPT_DIR, "kokoro-v1.0.onnx")
DEFAULT_VOICES = os.path.join(SCRIPT_DIR, "voices-v1.0.bin")


def get_optimized_kokoro(model_path: str = DEFAULT_MODEL, voices_path: str = DEFAULT_VOICES) -> Kokoro:
    """Builds an InferenceSession tuned for maximum RTX 3060 throughput & bounded VRAM."""
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.intra_op_num_threads = 4

    cuda_provider_options = {
        "device_id": "0",
        "arena_extend_strategy": "kNextPowerOfTwo",
        "gpu_mem_limit": str(1024 * 1024 * 1024),  # Cap at 1GB VRAM
        "cudnn_conv_algo_search": "DEFAULT",
        "do_copy_in_default_stream": "1",
    }

    providers = [
        ("CUDAExecutionProvider", cuda_provider_options),
        "CPUExecutionProvider"
    ]

    session = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
    return Kokoro.from_session(session, voices_path=voices_path)


def parse_voice_spec(kokoro: Kokoro, voice_spec: str):
    """Parses single voice or blended voice (e.g., 'hf_alpha:70,af_heart:30')."""
    if "," not in voice_spec and ":" not in voice_spec:
        return kokoro.get_voice_style(voice_spec)

    parts = voice_spec.split(",")
    blended = None

    for part in parts:
        if ":" in part:
            name, weight_str = part.strip().split(":")
            weight = float(weight_str)
        else:
            name = part.strip()
            weight = 100.0 / len(parts)

        style = kokoro.get_voice_style(name)
        if blended is None:
            blended = (weight / 100.0) * style
        else:
            blended += (weight / 100.0) * style

    return blended


async def stream_audio(kokoro: Kokoro, text: str, voice, speed: float, lang: str, output_path: str):
    """Streams audio chunks with sub-350ms Time-To-First-Audio."""
    t0 = time.perf_counter()
    stream = kokoro.create_stream(text, voice=voice, speed=speed, lang=lang, sentence_pause=0.3)

    all_chunks = []
    sr = 24000
    ttfa = None
    chunk_idx = 0

    async for chunk, sr in stream:
        if ttfa is None:
            ttfa = (time.perf_counter() - t0) * 1000
            print(f"Time-To-First-Audio (TTFA): {ttfa:.1f} ms")
        all_chunks.append(chunk)
        chunk_idx += 1

    total_time = (time.perf_counter() - t0) * 1000
    import numpy as np
    full_audio = np.concatenate(all_chunks) if all_chunks else np.array([], dtype=np.float32)
    duration = len(full_audio) / sr

    sf.write(output_path, full_audio, sr)
    print(f"Stream finished in {total_time:.1f} ms ({duration:.2f}s audio, {chunk_idx} chunks) -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Kokoro-82M ONNX Hinglish TTS (Optimized)")
    parser.add_argument("text", nargs="?", help="Text to speak (Hinglish / Hindi in Devanagari)")
    parser.add_argument("-o", "--output", default="output.wav", help="Output WAV path (default: output.wav)")
    parser.add_argument(
        "-v", "--voice",
        default="hf_alpha:70,af_heart:30",
        help="Voice name or blend ratio (default: 'hf_alpha:70,af_heart:30')"
    )
    parser.add_argument("-s", "--speed", type=float, default=1.0, help="Speech speed (0.5 to 2.0, default 1.0)")
    parser.add_argument("-l", "--lang", default="hi", help="Language code (default: 'hi')")
    parser.add_argument("--sentence-pause", type=float, default=0.3, help="Pause between sentences in seconds (default: 0.3)")
    parser.add_argument("--stream", action="store_true", help="Enable streaming synthesis for low TTFA")
    parser.add_argument("--list-voices", action="store_true", help="List all available voices and exit")

    args = parser.parse_args()

    # Load optimized model
    kokoro = get_optimized_kokoro()

    if args.list_voices:
        voices = kokoro.get_voices()
        hindi = [v for v in voices if v.startswith("h")]
        english = [v for v in voices if v.startswith("a")]
        print(f"Hindi Voices  : {', '.join(hindi)}")
        print(f"English Voices: {', '.join(english[:10])} ... ({len(english)} total)")
        sys.exit(0)

    # Read from stdin if no text argument is given
    text = args.text
    if not text:
        if not sys.stdin.isatty():
            text = sys.stdin.read().strip()
        else:
            parser.print_help()
            sys.exit(1)

    if not text:
        print("Error: Empty text provided.")
        sys.exit(1)

    voice_tensor = parse_voice_spec(kokoro, args.voice)

    if args.stream:
        asyncio.run(stream_audio(kokoro, text, voice_tensor, args.speed, args.lang, args.output))
        return

    # Standard synthesis
    t0 = time.perf_counter()
    samples, sample_rate = kokoro.create(
        text,
        voice=voice_tensor,
        speed=args.speed,
        lang=args.lang,
        sentence_pause=args.sentence_pause
    )
    elapsed = (time.perf_counter() - t0) * 1000
    duration = len(samples) / sample_rate

    sf.write(args.output, samples, sample_rate)

    print(f"Synthesized: \"{text[:50]}{'...' if len(text) > 50 else ''}\"")
    print(f"- Voice    : {args.voice}")
    print(f"- Duration : {duration:.2f}s (Speed: {duration / (elapsed / 1000):.1f}x real-time | Latency: {elapsed:.1f}ms)")
    print(f"- Saved to : {args.output}")


if __name__ == "__main__":
    main()
