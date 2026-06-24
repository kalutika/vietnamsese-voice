#!/usr/bin/env python3
"""
Gwen-TTS: Vietnamese Voice Cloning Inference

Usage:
    # Using a built-in reference speaker
    python inference.py --text "Xin chào Việt Nam" --speaker yen_nhi

    # Using a custom reference audio
    python inference.py --text "Xin chào" --ref_audio my_voice.wav --ref_text "transcript of audio"

    # Using HuggingFace model (auto-download)
    python inference.py --text "Xin chào" --speaker yen_nhi --model_path g-group-ai-lab/gwen-tts-0.6B

    # List available speakers
    python inference.py --list_speakers
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime
import glob
import numpy as np

import torch
import soundfile as sf

# Recommended generation config for Gwen-TTS
# These parameters are optimized for natural Vietnamese voice cloning
GENERATION_CONFIG = dict(
    temperature=0.3,
    top_k=20,
    top_p=0.9,
    max_new_tokens=4096,
    repetition_penalty=2.0,
    subtalker_do_sample=True,
    subtalker_temperature=0.1,
    subtalker_top_k=20,
    subtalker_top_p=1.0,
)


def resolve_runtime(device: str):
    """Resolve runtime device and dtype with graceful CUDA fallback."""
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda:0", torch.bfloat16
        return "cpu", torch.float32

    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but no NVIDIA GPU/driver found. Falling back to CPU.")
        return "cpu", torch.float32

    if device.startswith("cuda"):
        return device, torch.bfloat16

    return device, torch.float32


def load_model(model_path, device, dtype):
    """Load the Gwen-TTS model."""
    from qwen_tts import Qwen3TTSModel

    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except Exception:
        attn_impl = "sdpa"
        print("Note: flash-attn not available, using sdpa attention (slightly slower)")

    model = Qwen3TTSModel.from_pretrained(
        model_path,
        device_map=device,
        dtype=dtype,
        attn_implementation=attn_impl,
    )
    return model


def load_speaker_info(ref_info_path):
    """Load reference speaker metadata."""
    with open(ref_info_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_text_input(text: Optional[str], text_file: Optional[str]) -> str:
    """Resolve synthesis text from --text or --text_file."""
    if text and text_file:
        raise ValueError("Use either --text or --text_file, not both")

    if text_file:
        text_path = Path(text_file)
        if not text_path.exists():
            raise FileNotFoundError(f"Text file not found: {text_file}")
        file_text = text_path.read_text(encoding="utf-8").strip()
        if not file_text:
            raise ValueError("Text file is empty")
        return file_text

    if text:
        cleaned_text = text.strip()
        if cleaned_text:
            return cleaned_text

    raise ValueError("Either --text or --text_file must be provided")


def generate_voice_clone(model, text, language, ref_audio, ref_text):
    """Generate speech using voice cloning with optimized parameters."""
    wavs, sr = model.generate_voice_clone(
        text=text,
        # language=language,
        ref_audio=ref_audio,
        ref_text=ref_text,
        **GENERATION_CONFIG,
    )
    return wavs[0], sr


def generate_with_speaker(model, text, language, speaker_key, ref_info, base_dir):
    """Generate speech using a built-in reference speaker."""
    if speaker_key not in ref_info:
        available = ", ".join(ref_info.keys())
        raise ValueError(f"Speaker '{speaker_key}' not found. Available: {available}")

    speaker = ref_info[speaker_key]
    ref_audio_path = os.path.join(base_dir, speaker["audio_path"])
    ref_text = speaker["text"]

    return generate_voice_clone(model, text, language, ref_audio_path, ref_text)


def main():
    parser = argparse.ArgumentParser(description="Gwen-TTS: Vietnamese Voice Cloning")
    parser.add_argument("--text", type=str, help="Text to synthesize")
    parser.add_argument(
        "--text_file", type=str, default=None,
        help="Path to a UTF-8 text file containing text to synthesize",
    )
    parser.add_argument(
        "--speaker", type=str, default=None,
        help="Built-in speaker key (e.g., yen_nhi, khanh_toan)",
    )
    parser.add_argument(
        "--ref_audio", type=str, default=None,
        help="Path to custom reference audio WAV file",
    )
    parser.add_argument(
        "--ref_text", type=str, default=None,
        help="Transcript of the reference audio",
    )
    parser.add_argument(
        "--language", type=str, default="vietnamese",
        help="Language (default: vietnamese)",
    )
    parser.add_argument(
        "--model_path", type=str, default="g-group-ai-lab/gwen-tts-0.6B",
        help="Path to model or HuggingFace model ID (default: g-group-ai-lab/gwen-tts-0.6B)",
    )
    parser.add_argument(
        "--output", type=str, default="output.wav",
        help="Output WAV file path (default: output.wav)",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device: auto, cpu, or cuda:0 (default: auto)",
    )
    parser.add_argument(
        "--list_speakers", action="store_true",
        help="List available built-in speakers and exit",
    )

    args = parser.parse_args()

    base_dir = Path(__file__).parent
    ref_info_path = base_dir / "data" / "ref_info.json"

    if args.list_speakers:
        if ref_info_path.exists():
            ref_info = load_speaker_info(ref_info_path)
            print("Available speakers:")
            for key, info in ref_info.items():
                print(f"  {key:20s} - {info['name']}")
        else:
            print("ref_info.json not found.")
        sys.exit(0)

    # try:
    #     text_input = load_text_input(args.text, args.text_file)
    # except (ValueError, FileNotFoundError) as exc:
    #     parser.error(str(exc))

    # if args.speaker is None and args.ref_audio is None:
    #     parser.error("Either --speaker or --ref_audio must be provided")
    # if args.ref_audio and not args.ref_text:
    #     parser.error("--ref_text is required when using --ref_audio")

    # runtime_device, runtime_dtype = resolve_runtime(args.device)
    # print(f"Loading model from {args.model_path} on {runtime_device}...")
    # model = load_model(args.model_path, device=runtime_device, dtype=runtime_dtype)
    # print("Model loaded successfully.")

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # if args.speaker:
    #     ref_info = load_speaker_info(ref_info_path)
    #     print(f"Generating with speaker: {ref_info[args.speaker]['name']}...")
    #     # wav, sr = generate_with_speaker(
    #     #     model, text_input, args.language, args.speaker, ref_info, base_dir,
    #     # )
    #     # sf.write(args.output, wav, sr)
    #     # print(f"Saved to {args.output} (sample rate: {sr}Hz)")
    #     texts = [line.strip() for line in text_input.splitlines() if line.strip()]
    #     for index, text in enumerate(texts, start=1):
    #         print(f"Generating audio for text {index}/{len(texts)}: {text}")
    #         wav, sr = generate_with_speaker(model, text, args.language, args.speaker, ref_info, base_dir)
    #         filename = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{index}.wav"
    #         full_path = os.path.join(output_dir, filename)
    #         sf.write(full_path, wav, sr)
    #         print(f"Saved to {full_path} (sample rate: {sr}Hz)")
       
    # else:
    #     print(f"Generating with custom reference audio: {args.ref_audio}...")
    #     texts = [line.strip() for line in text_input.splitlines() if line.strip()]
    #     for index, text in enumerate(texts, start=1):
    #         print(f"Generating audio for text {index}/{len(texts)}: {text}")
    #         wav, sr = generate_voice_clone(model, text, args.language, args.ref_audio, args.ref_text)
    #         # put file to output folder
    #         filename = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{index}.wav"
    #         full_path = os.path.join(output_dir, filename)
    #         sf.write(full_path, wav, sr)
    #         print(f"Saved to {full_path} (sample rate: {sr}Hz)")

    # Merging all generated .wav files into a single file
    print("Merging all generated .wav files into a single file...")    
    # 1. Lấy tất cả các file .wav trong thư mục và sắp xếp theo thứ tự tên (tăng dần theo index)
    wav_files = sorted(glob.glob(os.path.join(output_dir, "*.wav")))

    if not wav_files:
        print("Không tìm thấy file .wav nào trong thư mục output!")
    else:
        all_data = []
        sample_rate = None

        # 2. Đọc từng file và đưa dữ liệu vào mảng
        for file_path in wav_files:
            # Bỏ qua chính file tổng nếu bạn vô tình chạy lại lệnh này
            if "final_merged_output" in file_path:
                continue
                
            data, sr = sf.read(file_path)
            sample_rate = sr  # Lấy tần số lấy mẫu (thường là giống nhau giữa các file)
            all_data.append(data)

        # 3. Nối tất cả mảng dữ liệu âm thanh lại với nhau
        merged_data = np.concatenate(all_data, axis=0)

        # 4. Ghi file tổng ra thư mục output
        final_filename = os.path.join(output_dir, "final_merged_output.wav")
        sf.write(final_filename, merged_data, sample_rate)
        
        print(f"Đã nối thành công {len(all_data)} file thành: {final_filename}")



if __name__ == "__main__":
    main()
