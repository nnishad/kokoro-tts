#!/usr/bin/env python3
"""
Benchmark and validation suite for Kokoro-82M TTS Microservice.
Tests:
  1. One-Shot latency and audio validity
  2. Streaming Time-To-First-Audio (TTFA)
  3. Concurrent multi-client load test (safe GPU concurrency & fairness)
  4. OpenAI compatibility endpoint (/v1/audio/speech)
  5. WebSocket duplex streaming with interruption
"""

import time
import asyncio
import httpx
import websockets
import json

BASE_URL = "http://127.0.0.1:8880"
WS_URL = "ws://127.0.0.1:8880/ws/stream"


async def test_one_shot():
    print("\n--- [1/5] Testing One-Shot Endpoint (/synthesize) ---")
    async with httpx.AsyncClient() as client:
        t0 = time.perf_counter()
        resp = await client.post(
            f"{BASE_URL}/synthesize",
            json={
                "text": "नमस्ते! आपका RTX 3060 GPU बिल्कुल तैयार है और ₹500 में 99% accuracy देता है।",
                "voice": "hf_alpha:70,af_heart:30",
                "speed": 1.0,
                "format": "wav"
            },
            timeout=10.0
        )
        total_time = (time.perf_counter() - t0) * 1000
        assert resp.status_code == 200, f"Error: {resp.text}"
        data = resp.content
        latency_hdr = resp.headers.get("x-latency-ms", "N/A")
        duration_hdr = resp.headers.get("x-audio-duration-s", "N/A")
        print(f"Status: {resp.status_code} OK")
        print(f"Total Request Time: {total_time:.1f}ms")
        print(f"Reported Latency: {latency_hdr}ms | Audio Duration: {duration_hdr}s")
        print(f"WAV File Size: {len(data)} bytes (RIFF header: {data[:4]})")
        assert data[:4] == b"RIFF", "Output must be valid RIFF WAV"


async def test_streaming_ttfa():
    print("\n--- [2/5] Testing Streaming Endpoint (/stream) for TTFA ---")
    text = (
        "नमस्ते दुनिया! यह एक रियल-टाइम ऑडियो स्ट्रीमिंग टेस्ट है। "
        "पहला वाक्य तुरंत प्ले होना शुरू हो जाता है, "
        "जबकि बाकी का टेक्स्ट बैकग्राउंड में लगातार प्रोसेस होता रहता है।"
    )
    async with httpx.AsyncClient() as client:
        t0 = time.perf_counter()
        first_chunk_time = None
        total_bytes = 0
        chunk_count = 0

        async with client.stream(
            "POST",
            f"{BASE_URL}/stream",
            json={"text": text, "voice": "hf_alpha:70,af_heart:30", "format": "wav"},
            timeout=15.0
        ) as response:
            assert response.status_code == 200, f"Error: {response.status_code}"
            async for chunk in response.aiter_bytes():
                chunk_count += 1
                total_bytes += len(chunk)
                if first_chunk_time is None:
                    first_chunk_time = (time.perf_counter() - t0) * 1000
                    print(f"--> Time-To-First-Audio (TTFA): {first_chunk_time:.1f}ms <-- (Chunk 1 size: {len(chunk)} bytes)")

        total_elapsed = (time.perf_counter() - t0) * 1000
        print(f"Total Chunks Received: {chunk_count}")
        print(f"Total Stream Bytes: {total_bytes} bytes")
        print(f"Total Streaming Duration: {total_elapsed:.1f}ms")
        assert first_chunk_time is not None and first_chunk_time < 300, f"TTFA too high: {first_chunk_time}ms"


async def test_openai_compatibility():
    print("\n--- [3/5] Testing OpenAI Speech Endpoint (/v1/audio/speech) ---")
    async with httpx.AsyncClient() as client:
        # Test one-shot
        t0 = time.perf_counter()
        resp = await client.post(
            f"{BASE_URL}/v1/audio/speech",
            json={
                "model": "kokoro",
                "input": "OpenAI drop-in compatibility is fully operational.",
                "voice": "alloy",
                "response_format": "wav"
            },
            timeout=10.0
        )
        total_time = (time.perf_counter() - t0) * 1000
        assert resp.status_code == 200
        print(f"One-shot OpenAI Speech Status: {resp.status_code} in {total_time:.1f}ms ({len(resp.content)} bytes)")

        # Test streaming mode
        t0 = time.perf_counter()
        ttfa = None
        async with client.stream(
            "POST",
            f"{BASE_URL}/v1/audio/speech",
            json={
                "model": "kokoro",
                "input": "Streaming test for OpenAI API. First audio chunk should arrive immediately.",
                "voice": "nova",
                "response_format": "wav",
                "stream": True
            },
            timeout=10.0
        ) as stream_resp:
            assert stream_resp.status_code == 200
            async for chunk in stream_resp.aiter_bytes():
                if ttfa is None:
                    ttfa = (time.perf_counter() - t0) * 1000
                    print(f"OpenAI Stream TTFA: {ttfa:.1f}ms")
                    break


async def test_concurrency(num_concurrent: int = 5):
    print(f"\n--- [4/5] Testing Concurrency ({num_concurrent} Concurrent Clients) ---")
    texts = [
        "पहला क्लाइंट: सर्वर स्टेटस नॉर्मल है।",
        "दूसरा क्लाइंट: डेटाबेस कनेक्शन सुरक्षित है।",
        "तीसरा क्लाइंट: मेमोरी उपयोग सीमा के अंदर है।",
        "चौथा क्लाइंट: GPU तापमान सामान्य है।",
        "पाँचवाँ क्लाइंट: सभी प्रक्रियाएँ ठीक से चल रही हैं।"
    ]

    async def single_client(client_id: int, text: str):
        t0 = time.perf_counter()
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{BASE_URL}/synthesize",
                json={"text": text, "voice": "hf_alpha:70,af_heart:30"},
                timeout=15.0
            )
            elapsed = (time.perf_counter() - t0) * 1000
            assert resp.status_code == 200
            print(f"Client {client_id} finished in {elapsed:.1f}ms (WAV: {len(resp.content)} bytes)")
            return elapsed

    t_all_start = time.perf_counter()
    tasks = [single_client(i + 1, texts[i % len(texts)]) for i in range(num_concurrent)]
    latencies = await asyncio.gather(*tasks)
    total_concurrency_time = (time.perf_counter() - t_all_start) * 1000

    print(f"All {num_concurrent} requests completed successfully in {total_concurrency_time:.1f}ms")
    print(f"Average client turnaround time: {sum(latencies)/len(latencies):.1f}ms")


async def test_websocket_interruption():
    print("\n--- [5/5] Testing WebSocket Streaming & Interruption (/ws/stream) ---")
    async with websockets.connect(WS_URL) as ws:
        # Start synthesis
        await ws.send(json.dumps({
            "action": "synthesize",
            "text": "नमस्ते! यह एक बहुत लंबा टेक्स्ट है जिसे हम बीच में रोक देंगे ताकि इंटरप्शन का परीक्षण हो सके।",
            "voice": "hf_alpha:70,af_heart:30",
            "format": "pcm"
        }))

        start_event = json.loads(await ws.recv())
        print(f"WebSocket start event received: {start_event}")

        # Receive 1st audio chunk
        first_audio_chunk = await ws.recv()
        print(f"Received audio chunk: {len(first_audio_chunk)} bytes")

        # Now send interrupt (stop)
        print("Sending interruption signal: {'action': 'stop'}...")
        await ws.send(json.dumps({"action": "stop"}))

        # Drain until we get the JSON 'stopped' event
        done_event = None
        while True:
            msg = await ws.recv()
            if isinstance(msg, str):
                done_event = json.loads(msg)
                break

        print(f"Received post-stop event: {done_event}")
        assert done_event.get("event") in ["stopped", "interrupted"]
        print("WebSocket live interruption test passed successfully!")


async def main():
    print("=" * 60)
    print(" Kokoro-82M Microservice Algorithmic Verification Suite")
    print("=" * 60)
    await test_one_shot()
    await test_streaming_ttfa()
    await test_openai_compatibility()
    await test_concurrency(5)
    await test_websocket_interruption()
    print("\n" + "=" * 60)
    print(" ALL 5 BENCHMARK & ALGORITHM TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
