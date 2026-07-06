import os
import shutil
import spaces
import gradio as gr
import librosa
import torch
from peft import PeftModel
from transformers import AutoProcessor, Gemma4ForConditionalGeneration
from fastapi import UploadFile, File

MODEL_ID = "google/gemma-4-E2B-it"
ADAPTER_DIR = os.environ.get(
    "ADAPTER_DIR", "rookie-systems/gemma-4-audio-asr-mn-adapter"
)
TARGET_SR = 16000
HF_TOKEN = os.environ.get("HF_TOKEN")        
INSTRUCTION = "Transcribe this audio."       

def pick_device_map():
    if os.environ.get("SPACES_ZERO_GPU"): 
        return "cuda"
    if not torch.cuda.is_available():
        print(">>> CUDA not available — loading on CPU (slow).")
        return "cpu"
    free_gb = torch.cuda.mem_get_info()[0] / 1024**3
    need_gb = float(os.environ.get("MIN_FREE_GB", "12"))  
    if free_gb < need_gb:
        print(f">>> Free GPU memory {free_gb:.1f}GB < required {need_gb}GB — loading on CPU (slow).")
        return "cpu"
    print(f">>> Sufficient GPU memory available ({free_gb:.1f}GB) — loading on cuda.")
    return "cuda"

DEVICE_MAP = pick_device_map()

print(">>> Loading base model:", MODEL_ID, "| device:", DEVICE_MAP)
base = Gemma4ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map=DEVICE_MAP,
    token=HF_TOKEN,
)
print(">>> Loading adapter:", ADAPTER_DIR)
model = PeftModel.from_pretrained(base, ADAPTER_DIR, token=HF_TOKEN).eval()
processor = AutoProcessor.from_pretrained(MODEL_ID, token=HF_TOKEN)
print(">>> Model and processor ready.")


@spaces.GPU(duration=60)
@torch.inference_mode()
def transcribe(audio_path):
    if audio_path is None:
        return "⚠️ Аудио оруулна уу."

    audio, _ = librosa.load(audio_path, sr=TARGET_SR, mono=True)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio},
                {"type": "text", "text": INSTRUCTION},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    inputs = processor(text=text, audio=[audio], return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    gen = out[0][inputs["input_ids"].shape[-1]:]
    return processor.tokenizer.decode(gen, skip_special_tokens=True).strip()


FIXED_AUDIO_CSS = """
#audio_in { min-height: 320px; }
#audio_in .audio-container,
#audio_in .source-wrap,
#audio_in > .container { min-height: 260px; }
"""

demo = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(
        sources=["microphone", "upload"],
        type="filepath",
        label="Аудио",
        elem_id="audio_in",
    ),
    outputs=gr.Textbox(label="Текст", lines=8),
    title="🎙️ Gemma-4 Audio ASR (Монгол)",
    description="Аудио (микрофон эсвэл файл) оруулаад илгээнэ үү — Монгол текст болгон хөрвүүлнэ.",
    flagging_mode="never",
    css=FIXED_AUDIO_CSS,
)

# ─── DIO-Д ЗОРИУЛСАН HTTP API ENDPOINT ─────────────────────────────────────
# Gradio-ийн өөрийнх нь дотоод FastAPI сервер дээр шууд route нэмж өгч байна
@demo.app.post("/api/transcribe")
async def api_transcribe(file: UploadFile = File(...)):
    try:
        temp_file_path = f"temp_{file.filename}"
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        result_text = transcribe(temp_file_path)
        
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            
        return {"text": result_text}
    except Exception as e:
        return {"error": str(e)}
# ───────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # demo.launch() нь серверийг унтраалгүй тасралтгүй ажиллуулах үүрэгтэй
    demo.launch()