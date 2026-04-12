"""
transcriber.py — NoteFlow Hackathon
Based on the official Qualcomm AI Hub simple-whisper-transcription reference app.
Runs Whisper-base-en ONNX on QNN NPU (Snapdragon X Elite) with CPU fallback.
"""

import numpy as np
import os
import scipy.signal
import soundfile as sf
import torch
import onnxruntime as ort

# ─────────────────────────────────────────────────────────────
# Paths  (model files live in ../models/ relative to backend/)
# ─────────────────────────────────────────────────────────────
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR       = os.path.join(BASE_DIR, "..", "models")
ENCODER_PATH     = os.path.join(MODELS_DIR, "WhisperEncoder.onnx")
DECODER_PATH     = os.path.join(MODELS_DIR, "WhisperDecoder.onnx")
MEL_FILTERS_PATH = os.path.join(MODELS_DIR, "mel_filters.npz")

# ─────────────────────────────────────────────────────────────
# Whisper-base-en constants  (must not change — baked into ONNX)
# ─────────────────────────────────────────────────────────────
SAMPLE_RATE      = 16000
CHUNK_LENGTH     = 30            # seconds
N_FFT            = 400
HOP_LENGTH       = 160
N_MELS           = 80
N_SAMPLES        = CHUNK_LENGTH * SAMPLE_RATE   # 480 000
MEAN_DECODE_LEN  = 224           # KV-cache depth in the ONNX decoder

# Token IDs
TOKEN_SOT             = 50257   # <|startoftranscript|>
TOKEN_EOT             = 50256   # <|endoftext|>
TOKEN_BLANK           = 220     # " " (space)
TOKEN_NO_TIMESTAMP    = 50362
TOKEN_TIMESTAMP_BEGIN = 50363
TOKEN_NO_SPEECH       = 50361
NO_SPEECH_THRESHOLD   = 0.6

# Whisper-base non-speech tokens — suppress these during decoding.
# This is the exact list from the reference repo; do not edit it.
NON_SPEECH_TOKENS = [
    1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62, 63,
    90, 91, 92, 93, 357, 366, 438, 532, 685, 705, 796, 930, 1058, 1220, 1267,
    1279, 1303, 1343, 1377, 1391, 1635, 1782, 1875, 2162, 2361, 2488, 3467,
    4008, 4211, 4600, 4808, 5299, 5855, 6329, 7203, 9609, 9959, 10563, 10786,
    11420, 11709, 11907, 13163, 13697, 13700, 14808, 15306, 16410, 16791,
    17992, 19203, 19510, 20724, 22305, 22935, 27007, 30109, 30420, 33409,
    34949, 40283, 40493, 40549, 47282, 49146, 50257, 50357, 50358, 50359,
    50360, 50361,
]

# Whisper-base-en model dimensions
NUM_DECODER_BLOCKS = 6
NUM_DECODER_HEADS  = 8
ATTENTION_DIM      = 512        # head_dim = ATTENTION_DIM // NUM_DECODER_HEADS = 64


# ─────────────────────────────────────────────────────────────
# ONNX session  (QNN NPU → CPU fallback)
# ─────────────────────────────────────────────────────────────
def _load_session(path: str) -> ort.InferenceSession:
    try:
        session = ort.InferenceSession(
            path,
            providers=["QNNExecutionProvider"],
            provider_options=[{
                "backend_path":                       "QnnHtp.dll",
                "htp_performance_mode":               "burst",
                "high_power_saver":                   "sustained_high_performance",
                "enable_htp_fp16_precision":          "1",
                "htp_graph_finalization_optimization_mode": "3",
            }],
        )
        print(f"[transcriber] {os.path.basename(path)} → NPU ✅")
        return session
    except Exception:
        session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        print(f"[transcriber] {os.path.basename(path)} → CPU (fallback)")
        return session


# ─────────────────────────────────────────────────────────────
# Mel filter  (load from .npz or fall back to scipy)
# ─────────────────────────────────────────────────────────────
def _load_mel_filter() -> np.ndarray:
    """
    Load the exact Whisper mel filterbank from mel_filters.npz.
    Falls back to scipy.signal.windows if the file is missing.
    The .npz file ships with the Qualcomm sample repo — keep it next to your
    model files or in the project root.
    """
    # Try project root first, then models dir
    candidates = [
        os.path.join(BASE_DIR, "..", "mel_filters.npz"),
        os.path.join(BASE_DIR, "mel_filters.npz"),
        os.path.join(MODELS_DIR, "mel_filters.npz"),
        "mel_filters.npz",
    ]
    for path in candidates:
        if os.path.exists(path):
            data = np.load(path)
            key  = "mel_filters" if "mel_filters" in data else data.files[0]
            filt = data[key].astype(np.float32)
            data.close()
            print(f"[transcriber] Mel filter loaded from {path}  shape={filt.shape}")
            return filt

    # Fallback — build a basic triangular mel filterbank with librosa-style math
    print("[transcriber] ⚠ mel_filters.npz not found — building fallback filterbank")
    n_mels = N_MELS
    fmax   = SAMPLE_RATE // 2
    mel_f  = 2595.0 * np.log10(1.0 + np.linspace(0, fmax, n_mels + 2) / 700.0)
    hz     = 700.0 * (10.0 ** (mel_f / 2595.0) - 1.0)
    bins   = np.floor((N_FFT + 1) * hz / SAMPLE_RATE).astype(int)
    fbank  = np.zeros((n_mels, N_FFT // 2 + 1), dtype=np.float32)
    for j in range(n_mels):
        for i in range(bins[j],   bins[j + 1]):
            fbank[j, i] = (i - bins[j]) / max(bins[j + 1] - bins[j], 1)
        for i in range(bins[j + 1], bins[j + 2]):
            fbank[j, i] = (bins[j + 2] - i) / max(bins[j + 2] - bins[j + 1], 1)
    return fbank


# ─────────────────────────────────────────────────────────────
# Log-mel spectrogram
# ─────────────────────────────────────────────────────────────
def _log_mel_spectrogram(audio_np: np.ndarray, mel_filter: np.ndarray) -> np.ndarray:
    """
    Exact replica of the reference repo's _log_mel_spectrogram().
    Uses PyTorch STFT (identical numerical behaviour to openai/whisper).

    Returns numpy array of shape (1, 80, 3000).
    """
    audio = torch.from_numpy(audio_np.astype(np.float32))

    # Zero-pad or trim to exactly 30 seconds
    if len(audio) < N_SAMPLES:
        audio = torch.nn.functional.pad(audio, (0, N_SAMPLES - len(audio)))
    else:
        audio = audio[:N_SAMPLES]

    # Hann-windowed STFT
    window     = torch.hann_window(N_FFT)
    stft       = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2      # (n_fft/2+1, n_frames)

    # Apply mel filterbank
    mel_spec = torch.from_numpy(mel_filter) @ magnitudes   # (80, n_frames)

    # ── THE CORRECT 3-STEP NORMALISATION (from reference repo) ──
    # Step 1: clamp + log10
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    # Step 2: clip values more than 8 below the peak  ← was missing before
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    # Step 3: shift and scale to roughly [-1, 1]
    log_spec = (log_spec + 4.0) / 4.0

    return log_spec.unsqueeze(0).detach().float().numpy()   # (1, 80, 3000)


# ─────────────────────────────────────────────────────────────
# Timestamp suppression + logprobs  (from reference repo)
# ─────────────────────────────────────────────────────────────
def _apply_timestamp_rules(
    logits: np.ndarray,
    decoded_tokens: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    from scipy.special import log_softmax

    # Always suppress <|notimestamps|>
    logits[TOKEN_NO_TIMESTAMP] = -np.inf

    # At step 0 (only SOT in decoded_tokens), suppress all timestamp tokens
    if len(decoded_tokens) == 1:
        logits[TOKEN_TIMESTAMP_BEGIN:] = -np.inf

    logprobs = log_softmax(logits.astype(np.float64)).astype(np.float32)
    return logits, logprobs


# ─────────────────────────────────────────────────────────────
# Chunk + resample  (handles recordings longer than 30 seconds)
# ─────────────────────────────────────────────────────────────
def _chunk_and_resample(audio: np.ndarray, sr: int) -> list[np.ndarray]:
    """
    Resample to 16 kHz, then split into ≤30-second chunks.
    Short recordings (< 30 s) are returned as a single-element list.
    """
    # Stereo → mono
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # Resample if needed
    if sr != SAMPLE_RATE:
        audio = scipy.signal.resample(audio, int(len(audio) * SAMPLE_RATE / sr))

    audio = audio.astype(np.float32)

    n_full = audio.shape[0] // N_SAMPLES
    end    = n_full * N_SAMPLES

    if n_full == 0:
        return [audio]

    chunks = list(np.array_split(audio[:end], n_full))
    tail   = audio[end:]
    if len(tail) > 0:
        chunks.append(tail)
    return chunks


# ─────────────────────────────────────────────────────────────
# Single-chunk transcription  (core decoding loop)
# ─────────────────────────────────────────────────────────────
def _transcribe_chunk(
    audio_chunk: np.ndarray,
    encoder_session: ort.InferenceSession,
    decoder_session: ort.InferenceSession,
    mel_filter: np.ndarray,
) -> str:
    """
    Transcribe one ≤30-second audio chunk.
    Mirrors _transcribe_single_chunk() from the reference repo exactly.
    """

    # 1. Mel spectrogram
    mel_input = _log_mel_spectrogram(audio_chunk, mel_filter)

    # 2. Encode
    encoder_out  = encoder_session.run(None, {"audio": mel_input})
    k_cache_cross = encoder_out[0]
    v_cache_cross = encoder_out[1]

    # 3. Initialise self-attention KV caches
    head_dim   = ATTENTION_DIM // NUM_DECODER_HEADS   # 64
    sample_len = MEAN_DECODE_LEN                       # 224

    k_cache_self = np.zeros(
        (NUM_DECODER_BLOCKS, NUM_DECODER_HEADS, head_dim, sample_len),
        dtype=np.float32,
    )
    v_cache_self = np.zeros(
        (NUM_DECODER_BLOCKS, NUM_DECODER_HEADS, sample_len, head_dim),
        dtype=np.float32,
    )

    # 4. Autoregressive decoding loop
    # Start with just SOT — the model handles language/task internally
    x              = np.array([[TOKEN_SOT]], dtype=np.int32)
    decoded_tokens = [TOKEN_SOT]

    for i in range(sample_len):
        index = np.array([[i]], dtype=np.int32)

        decoder_out  = decoder_session.run(
            None,
            {
                "x":             x,
                "index":         index,
                "k_cache_cross": k_cache_cross,
                "v_cache_cross": v_cache_cross,
                "k_cache_self":  k_cache_self,
                "v_cache_self":  v_cache_self,
            },
        )
        logits       = decoder_out[0]
        k_cache_self = decoder_out[1]
        v_cache_self = decoder_out[2]

        logits = logits[0, -1].copy()   # (vocab_size,)  ← must copy!

        # ── Suppress unwanted tokens ──────────────────────────────
        # At step 0: block EOT and BLANK so the model must produce content
        if i == 0:
            logits[TOKEN_EOT]   = -np.inf
            logits[TOKEN_BLANK] = -np.inf

        # Always block the full non-speech token list
        logits[NON_SPEECH_TOKENS] = -np.inf

        # Timestamp rules + compute logprobs for no-speech detection
        logits, logprobs = _apply_timestamp_rules(logits, decoded_tokens)

        # ── No-speech detection at step 0 ─────────────────────────
        if i == 0:
            no_speech_prob = np.exp(logprobs[TOKEN_NO_SPEECH])
            if no_speech_prob > NO_SPEECH_THRESHOLD:
                print(f"[transcriber] Silence detected (p={no_speech_prob:.2f}) — skipping chunk")
                return ""

        # ── Greedy decode ─────────────────────────────────────────
        next_token = int(np.argmax(logits))

        if next_token == TOKEN_EOT:
            break

        x = np.array([[next_token]], dtype=np.int32)
        decoded_tokens.append(next_token)

    # 5. Token IDs → text
    try:
        import whisper as _whisper
        tokenizer = _whisper.decoding.get_tokenizer(
            multilingual=False, language="en", task="transcribe"
        )
        text = tokenizer.decode(decoded_tokens[1:])   # strip SOT
        return text.strip()

    except Exception as e:
        print(f"[transcriber] ⚠ Tokenizer error: {e}")
        # Fallback: try tiktoken (also installed with openai-whisper)
        try:
            import tiktoken
            enc  = tiktoken.get_encoding("gpt2")
            text = enc.decode(decoded_tokens[1:])
            return text.strip()
        except Exception:
            return f"[{len(decoded_tokens)-1} tokens decoded but tokenizer unavailable]"


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────
def transcribe_audio(wav_path: str) -> str:
    """
    Transcribe a WAV file to text.

    Args:
        wav_path: Path to any WAV file (any sample rate, mono or stereo).

    Returns:
        Transcribed text string.
        Returns a descriptive error string (never raises) so Flask stays alive.
    """

    if not os.path.exists(wav_path):
        return "Error: recording not found — please record audio first."

    print(f"\n[transcriber] ── Starting transcription ──")
    print(f"[transcriber] File : {wav_path}")

    # Load audio
    try:
        audio, sr = sf.read(wav_path)
    except Exception as e:
        return f"Error: could not read audio file — {e}"

    duration = len(audio) / sr
    print(f"[transcriber] Audio: {duration:.1f}s  sr={sr}  shape={audio.shape}")

    # Load models + mel filter
    try:
        encoder_session = _load_session(ENCODER_PATH)
        decoder_session = _load_session(DECODER_PATH)
        mel_filter      = _load_mel_filter()
    except Exception as e:
        return f"Error: could not load model — {e}"

    # Resample + chunk (handles recordings > 30 seconds automatically)
    chunks = _chunk_and_resample(audio, sr)
    print(f"[transcriber] Chunks: {len(chunks)}")

    # Transcribe each chunk and join
    results = []
    for idx, chunk in enumerate(chunks):
        print(f"[transcriber] Chunk {idx+1}/{len(chunks)}  samples={len(chunk)}")
        text = _transcribe_chunk(chunk, encoder_session, decoder_session, mel_filter)
        if text:
            results.append(text)

    transcript = " ".join(results).strip()

    if not transcript:
        return "No speech detected in recording."

    print(f"[transcriber] ✅ Done: {repr(transcript[:80])}{'...' if len(transcript)>80 else ''}")
    return transcript


# ─────────────────────────────────────────────────────────────
# CLI:  python transcriber.py [recording.wav]
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    wav = sys.argv[1] if len(sys.argv) > 1 else "recording.wav"
    print(transcribe_audio(wav))
