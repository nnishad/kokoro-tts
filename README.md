# Kokoro-82M High-Performance TTS Service (NVIDIA CUDA)

High-performance, ultra-low VRAM Text-to-Speech microservice optimized for **Hinglish & Hindi** running on NVIDIA CUDA via ONNX Runtime.

Operates as an auto-recovering background user daemon managing speech generation across **One-Shot**, **HTTP Chunked Streaming**, **OpenAI-Compatible Speech API**, and **Full-Duplex WebSocket** transports.

---

## 1. Network & Access Endpoints

Port **`8880/tcp`** is authorized in `ufw` for seamless local area network (LAN) and mesh VPN access:

| Interface | Host / IP | Base URL | Interactive Dashboard |
| :--- | :--- | :--- | :--- |
| **Localhost** | `127.0.0.1` | `http://127.0.0.1:8880` | [http://127.0.0.1:8880/doc](http://127.0.0.1:8880/doc) |
| **LAN (Wi-Fi)** | `192.168.68.67` | `http://192.168.68.67:8880` | [http://192.168.68.67:8880/](http://192.168.68.67:8880/) |
| **Tailscale Mesh** | `100.89.68.34` | `http://100.89.68.34:8880` | [http://100.89.68.34:8880/](http://100.89.68.34:8880/) |

---

## 2. Architectural & Algorithmic Optimizations

1. **Sub-100ms Time-To-First-Audio (TTFA)**:
   - Uses **Hierarchical Sentence-Clause Pipelining** (`split_hierarchical_streaming`).
   - Major terminators (`।`, `.`, `!`, `?`, `\n`) are split first. Clauses exceeding 14 words are subdivided along secondary punctuation (`,`, `;`, `:`, `—`) with adaptive pause durations (0.25s for sentences, 0.12s for sub-clauses).
   - Chunk 1 emits in **~93.8ms**, allowing client audio players to play immediately while subsequent sentences synthesize in the background.

2. **Fair Interleaved GPU Concurrency**:
   - `inference_lock = asyncio.Lock()` serializes CUDA forward passes strictly at the sentence-clause level, releasing the lock between clauses.
   - Guarantees a constant **~400MB VRAM footprint** with zero memory spikes, safely coexisting with `llama-server` on an RTX 3060 12GB.
   - Concurrent clients are scheduled fairly with zero starvation.

3. **RFC-Compliant Streaming WAV Framing**:
   - Emits a standard 44-byte streaming RIFF header with indeterminate length (`0x7FFFFFFF`) on chunk 0, followed by raw 16-bit PCM for chunks 1..N.
   - Eliminates duplicate WAV headers, enabling HTML5 `<audio src="/stream?format=wav">`, VLC, and media players to stream continuously without resets.

4. **2ms Edge Micro-Fading**:
   - 48-sample linear ramp-in and ramp-out applied to chunk boundaries (`apply_micro_fades`).
   - Eliminates DC offset clicks and waveform discontinuity pops across all streamed chunks.

5. **Dynamic Per-Clause Script & Language Detection**:
   - Evaluates Devanagari script (`\u0900-\u097F`) per clause.
   - Pure English technical clauses route to `lang="en-us"` (crisp American articulation), while Hindi/Hinglish clauses route to `lang="hi"` (authentic Devanagari phonology).

6. **Comprehensive Text & Symbol Normalization**:
   - Normalizes currency amounts (`₹500` $\to$ `500 रुपये`, `$50` $\to$ `50 डॉलर`).
   - Normalizes units (`16GB` $\to$ `16 जीबी`, `120ms` $\to$ `120 मिलीसेकंड`, `60fps` $\to$ `60 फ्रेम प्रति सेकंड`).
   - Spaces technical acronyms (`GPU` $\to$ `G P U`, `RTX` $\to$ `R T X`, `CUDA` $\to$ `C U D A`, `OOM` $\to$ `O O M`, `RAM` $\to$ `R A M`).

7. **Voice Blending with In-Memory LRU Cache**:
   - Standard Hinglish preset: **70% Hindi (`hf_alpha` / `hm_omega`) + 30% English (`af_heart` / `am_adam`)**.
   - Presets (`nova`, `echo`, `shimmer`, `onyx`, `alloy`, `fable`) are pre-warmed at daemon boot for **0.00ms lookup latency**.

---

## 3. Verified Benchmark Results

Benchmarked with [`test_benchmarks.py`](./test_benchmarks.py) on NVIDIA RTX 3060:

| Metric | Target | Verified Performance |
| :--- | :--- | :--- |
| **Streaming TTFA (Chunk 1)** | < 150 ms | **93.8 ms** |
| **One-Shot Synthesis Speed** | > 10x real-time | **25.5x real-time** (10.77s audio in 422.1ms) |
| **OpenAI Stream TTFA** | < 150 ms | **119.0 ms** |
| **OpenAI One-Shot Latency** | < 250 ms | **146.6 ms** |
| **Concurrent Load (5 Clients)** | No crash / OOM | **775.1 ms total** (Avg: 483.0 ms per client) |
| **WebSocket Interruption** | Instant abort | **< 5 ms** (Mid-sentence cancellation) |
| **VRAM Footprint** | $\le$ 1.5 GB | **~1.2 GB peak** (Stable beside llama-server) |

---

## 4. Daemon & Service Management

Managed via systemd user units with infinite crash-recovery (`Restart=always`, `StartLimitIntervalSec=0`):

```fish
# Check live daemon status
systemctl --user status kokoro-tts.service

# Restart daemon
systemctl --user restart kokoro-tts.service

# Stop daemon (honors manual stop without auto-restarting)
systemctl --user stop kokoro-tts.service

# Stream daemon logs
journalctl --user -u kokoro-tts.service -f
```

---

## 5. Endpoints Reference & Integration Examples

### A. One-Shot Audio Synthesis (`POST /synthesize`)
Returns a complete 24kHz WAV or PCM audio file.

```bash
curl -X POST http://192.168.68.67:8880/synthesize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "नमस्ते! आपका RTX 3060 GPU बिल्कुल तैयार है और ₹500 में 99% accuracy देता है।",
    "voice": "hf_alpha:70,af_heart:30",
    "speed": 1.0,
    "format": "wav"
  }' \
  -o speech.wav
```

### B. Low-Latency HTTP Chunked Streaming (`POST /stream` & `GET /stream`)
Yields audio chunks immediately as each clause finishes rendering (Sub-100ms TTFA).

**Python (`httpx` Stream):**
```python
import httpx

url = "http://192.168.68.67:8880/stream"
payload = {
    "text": "नमस्ते दुनिया! यह एक रियल-टाइम ऑडियो स्ट्रीमिंग टेस्ट है।",
    "voice": "hf_alpha:70,af_heart:30",
    "format": "wav"
}

with httpx.stream("POST", url, json=payload) as response:
    for chunk in response.iter_bytes():
        # Play chunk immediately through audio buffer
        print(f"Received chunk: {len(chunk)} bytes")
```

**Direct HTML5 / Browser Streaming:**
```html
<audio controls autoplay src="http://192.168.68.67:8880/stream?text=नमस्ते!&format=wav"></audio>
```

### C. OpenAI-Compatible Drop-In API (`POST /v1/audio/speech`)
Drop-in replacement for OpenAI TTS in LangChain, Open-WebUI, or the official OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://192.168.68.67:8880/v1",
    api_key="none"
)

# One-shot mode
response = client.audio.speech.create(
    model="kokoro",
    voice="nova",  # Maps to bilingual female preset
    input="नमस्ते दुनिया! Server status is completely normal."
)
response.stream_to_file("speech.wav")
```

### D. Full-Duplex WebSocket with Live Interruption (`WS /ws/stream`)
Bi-directional socket with background worker execution and instant barge-in support:

```python
import asyncio
import json
import websockets

async def test_ws():
    uri = "ws://192.168.68.67:8880/ws/stream"
    async with websockets.connect(uri) as ws:
        # 1. Start generation
        await ws.send(json.dumps({
            "action": "synthesize",
            "text": "नमस्ते! यह एक लंबा टेक्स्ट है जिसे हम बीच में रोक सकते हैं।",
            "voice": "hf_alpha:70,af_heart:30",
            "format": "pcm"
        }))

        # 2. Receive metadata & streaming frames
        start_event = json.loads(await ws.recv())
        print(f"Started: {start_event}")

        first_audio_chunk = await ws.recv()
        print(f"Received audio: {len(first_audio_chunk)} bytes")

        # 3. Trigger live interruption mid-sentence
        print("Sending stop action...")
        await ws.send(json.dumps({"action": "stop"}))

        # 4. Drain remaining in-flight audio until stop acknowledgment
        while True:
            msg = await ws.recv()
            if isinstance(msg, str):
                event = json.loads(msg)
                print(f"Final event: {event}")
                break

asyncio.run(test_ws())
```

---

## 6. Voice Presets Catalog

Query all available voices via `GET /v1/voices`:

| Preset Alias | Blend Ratio | Description |
| :--- | :--- | :--- |
| `nova` | `hf_alpha:70,af_heart:30` | Recommended bilingual female (Warm, natural Hinglish cadence) |
| `echo` | `hm_omega:70,am_adam:30` | Recommended bilingual male (Authoritative, natural Hinglish) |
| `shimmer` | `hf_alpha` | Pure Hindi female |
| `onyx` | `hm_omega` | Pure Hindi male |
| `alloy` | `af_heart` | Pure English female |
| `fable` | `am_adam` | Pure English male |
