import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

# Load model (auto-downloads from HuggingFace)
model = Qwen3TTSModel.from_pretrained(
    "g-group-ai-lab/gwen-tts-0.6B",
    device_map="cpu",
    dtype=torch.float32,
    attn_implementation="flash_attention_2",
)

# Recommended generation config for Gwen-TTS
generation_config = dict(
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

# Voice cloning
wavs, sr = model.generate_voice_clone(
    text="VnExpress tin tức mới nhất - Thông tin nhanh & chính xác được cập nhật hàng giờ. Đọc báo tin tức online Việt Nam & Thế giới nóng nhất trong ngày về thể thao ...",
    language="Vietnamese",
    ref_audio="./data/ref_audio/yen_nhi.wav",
    ref_text="",
    **generation_config,
)

sf.write("output.wav", wavs[0], sr)