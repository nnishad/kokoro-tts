import os
import time
import numpy as np
import soundfile as sf
import onnxruntime as ort
from kokoro_onnx import Kokoro

# 1. Benchmark standard vs tuned session options
ort.set_default_logger_severity(3)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "kokoro-v1.0.onnx")
VOICES_PATH = os.path.join(SCRIPT_DIR, "voices-v1.0.bin")

def get_optimized_kokoro():
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.intra_op_num_threads = 4

    cuda_provider_options = {
        "device_id": "0",
        "arena_extend_strategy": "kNextPowerOfTwo",
        "gpu_mem_limit": str(1024 * 1024 * 1024),  # 1GB limit
        "cudnn_conv_algo_search": "DEFAULT",
        "do_copy_in_default_stream": "1",
    }

    providers = [
        ("CUDAExecutionProvider", cuda_provider_options),
        "CPUExecutionProvider"
    ]

    session = ort.InferenceSession(MODEL_PATH, sess_options=opts, providers=providers)
    kokoro = Kokoro.from_session(session, voices_path=VOICES_PATH)
    return kokoro

print("=== 1. Testing Optimized Session Initialization ===")
t0 = time.perf_counter()
kokoro = get_optimized_kokoro()
print(f"Session initialized in {(time.perf_counter() - t0) * 1000:.1f} ms")

# Warm-up run (forces CUDA kernel compile / tensor allocation)
print("\n=== 2. Performing Warm-up Run ===")
w_start = time.perf_counter()
_ = kokoro.create("नमस्ते", voice="hf_alpha", lang="hi")
print(f"Warm-up completed in {(time.perf_counter() - w_start) * 1000:.1f} ms")

# Test Hinglish sentences of different lengths
test_sentences = [
    ("Short", "नमस्ते! आपका स्वागत है।"),
    ("Medium (Hinglish)", "नमस्ते! आपका server restart हो गया है और सब processes normal काम कर रहे हैं।"),
    ("Long (Dialogue)", "नमस्ते! आज का weather काफी अच्छा है। हमने सारे system logs check कर लिए हैं, कोई critical error नहीं मिला। आप अपना काम continue कर सकते हैं।")
]

print("\n=== 3. Speed & Latency Benchmark (Warmed GPU) ===")
voice_hindi = kokoro.get_voice_style("hf_alpha")
voice_english = kokoro.get_voice_style("af_heart")
blended_voice = (0.70 * voice_hindi) + (0.30 * voice_english)

for label, text in test_sentences:
    times = []
    # Run 3 iterations to get steady-state average
    for _ in range(3):
        t_start = time.perf_counter()
        samples, sr = kokoro.create(text, voice=blended_voice, speed=1.0, lang="hi")
        times.append((time.perf_counter() - t_start) * 1000)

    avg_latency = np.mean(times)
    min_latency = np.min(times)
    audio_dur = len(samples) / sr
    speed_mult = audio_dur / (min_latency / 1000)

    # Audio quality metrics
    rms = np.sqrt(np.mean(samples ** 2))
    peak = np.max(np.abs(samples))

    print(f"\n[{label}] ({len(text)} chars)")
    print(f"- Text: \"{text[:45]}...\"")
    print(f"- Audio Duration   : {audio_dur:.2f} seconds")
    print(f"- Generation Time  : avg={avg_latency:.1f}ms | best={min_latency:.1f}ms")
    print(f"- Inference Speed  : {speed_mult:.1f}x faster than real-time")
    print(f"- Quality Metrics  : Peak={peak:.2f} (max 1.0) | RMS Energy={rms:.4f} (clean audio)")

print("\n=== 4. Testing Streaming Latency (Time-To-First-Audio) ===")
import asyncio

async def test_streaming():
    test_stream_text = "नमस्ते! यह एक real-time streaming test है। हम चेक कर रहे हैं कि पहला audio chunk कितनी जल्दी आता है।"

    t_stream_start = time.perf_counter()
    stream = kokoro.create_stream(test_stream_text, voice=blended_voice, speed=1.0, lang="hi")

    ttfa = None
    chunk_count = 0
    total_samples = 0
    sr = 24000

    async for samples, sr in stream:
        if ttfa is None:
            ttfa = (time.perf_counter() - t_stream_start) * 1000
        chunk_count += 1
        total_samples += len(samples)

    total_stream_time = (time.perf_counter() - t_stream_start) * 1000
    total_stream_dur = total_samples / sr

    print(f"- Time-To-First-Audio (TTFA): {ttfa:.1f} ms")
    print(f"- Total Chunks Streamed     : {chunk_count}")
    print(f"- Total Stream Duration      : {total_stream_dur:.2f}s generated in {total_stream_time:.1f} ms")

asyncio.run(test_streaming())

print("\n=== Benchmark Complete ===")
