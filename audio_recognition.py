import torch
import time
import json
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


def get_text(audio):
    time_start = time.perf_counter()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
    )

    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )

    result = pipe(audio, return_timestamps=True)

    time_end = time.perf_counter()
    recognition_time = round((time_end-time_start)/60,2)
    return result,recognition_time


def get_finished_json(audio):
    result, recognition_time = get_text(audio)
    raw = result["text"].strip()
    summary = "<b>Это конспект лекции</b>"
    glossary = ""
    for res in result["chunks"]:
        glossary += f'{res["timestamp"]} - {res["text"]}\n'

    data = {"raw": raw, "summary": summary, "glossary": glossary, "recognition_time":  f"{recognition_time} мин."}
    with open('resources/outputs/data4.json', 'w', encoding='utf-8') as f:
        json.dump(data, f)
    return data





