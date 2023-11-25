import csv
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, Body, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from audio_recognition import get_finished_json, get_text

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/audio")
def upload_audio(file: UploadFile = File(...)):
    audio = file.file.read()
    return get_finished_json(audio)


@app.post("/submission")
def create_submission(file: UploadFile = File(...)):
    audio = file.file.read()
    audio_file = file.filename
    audio_name = Path(audio_file).stem
    result, recognition_time = get_text(audio)

    glossary = result["chunks"]
    df = pd.read_csv('resources/outputs/sample_submission.csv', encoding='utf-8')
    for term in glossary:
        termin = term["text"].strip()
        df.loc[len(df.index)] = [audio_file, termin]
    df.to_csv(f'resources/outputs/submission_{audio_name}.csv', sep=',', encoding='utf-8', index=False)
    return "Check csv file in resources/outputs"
