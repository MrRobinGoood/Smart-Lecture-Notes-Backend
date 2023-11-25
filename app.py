import os
import pandas as pd

from uuid import uuid4
from huggingface_hub import snapshot_download
from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PDFMinerLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from chromadb.config import Settings
from llama_cpp import Llama

from pathlib import Path
from fastapi import FastAPI, Body, UploadFile, File

from fastapi.middleware.cors import CORSMiddleware
from audio_recognition import get_finished_json, get_text

SYSTEM_PROMPT = "Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им."
SYSTEM_TOKEN = 1788
USER_TOKEN = 1404
BOT_TOKEN = 9225
LINEBREAK_TOKEN = 13

ROLE_TOKENS = {
    "user": USER_TOKEN,
    "bot": BOT_TOKEN,
    "system": SYSTEM_TOKEN
}

LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PDFMinerLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
}

repo_name = "IlyaGusev/saiga_13b_lora_llamacpp"
model_name = "data/cache/model-q4_K.gguf"
embedder_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

snapshot_download(repo_id=repo_name, local_dir="data/saiga_13b_lora_llamacpp", allow_patterns=model_name,
                  cache_dir="data/cache")
snapshot_download(repo_id=embedder_name, local_dir="data/paraphrase-multilingual-mpnet-base-v2", cache_dir="data/cache")

model = Llama(
    model_path="data/cache/model-q4_K.gguf",
    n_ctx=2000,
    n_parts=1,
)

max_new_tokens = None
embeddings = HuggingFaceEmbeddings(model_name="data/paraphrase-multilingual-mpnet-base-v2")


def get_uuid():
    return str(uuid4())


def build_index(file_paths, chunk_size, chunk_overlap):
    if file_paths:
        print('класс')
        documents = [load_single_document(path) for path in file_paths]
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        documents = text_splitter.split_documents(documents)
        fixed_documents = []
        for doc in documents:
            doc.page_content = process_text(doc.page_content)
            if not doc.page_content:
                continue
            fixed_documents.append(doc)

        db = Chroma.from_documents(
            fixed_documents,
            embeddings,
            client_settings=Settings(
                anonymized_telemetry=False
            )
        )
        return db
    return None


def load_single_document(file_path: str) -> Document:
    ext = "." + file_path.rsplit(".", 1)[-1]
    assert ext in LOADER_MAPPING
    loader_class, loader_args = LOADER_MAPPING[ext]
    loader = loader_class(file_path, **loader_args)
    return loader.load()[0]


def get_message_tokens(model, role, content):
    message_tokens = model.tokenize(content.encode("utf-8"))
    message_tokens.insert(1, ROLE_TOKENS[role])
    message_tokens.insert(2, LINEBREAK_TOKEN)
    message_tokens.append(model.token_eos())
    return message_tokens


def get_system_tokens(model):
    system_message = {"role": "system", "content": SYSTEM_PROMPT}
    return get_message_tokens(model, **system_message)


def upload_files(files):
    file_paths = ["resources/" + f.filename for f in files]
    return file_paths


def process_text(text):
    lines = text.split("\n")
    lines = [line for line in lines if len(line.strip()) > 2]
    text = "\n".join(lines).strip()
    if len(text) < 10:
        return None
    return text


def user(message, history):
    new_history = history + [[message, None]]
    return "", new_history


def retrieve(history, db, k_documents):
    if db:
        last_user_message = history[-1][0]
        retriever = db.as_retriever(search_kwargs={"k": k_documents})
        docs = retriever.get_relevant_documents(last_user_message)
        retrieved_docs = "\n\n".join([doc.page_content for doc in docs])
    return retrieved_docs


def bot(
        history,
        retrieved_docs,
        top_p,
        top_k,
        temp
):
    if not history:
        return

    tokens = get_system_tokens(model)[:]
    tokens.append(LINEBREAK_TOKEN)

    for user_message, bot_message in history[:-1]:
        message_tokens = get_message_tokens(model=model, role="user", content=user_message)
        tokens.extend(message_tokens)
        if bot_message:
            message_tokens = get_message_tokens(model=model, role="bot", content=bot_message)
            tokens.extend(message_tokens)

    last_user_message = history[-1][0]
    if retrieved_docs:
        last_user_message = f"Контекст: {retrieved_docs}\n\nИспользуя контекст, ответь на вопрос: {last_user_message}"
    message_tokens = get_message_tokens(model=model, role="user", content=last_user_message)
    tokens.extend(message_tokens)

    role_tokens = [model.token_bos(), BOT_TOKEN, LINEBREAK_TOKEN]
    tokens.extend(role_tokens)
    generator = model.generate(
        tokens,
        top_k=top_k,
        top_p=top_p,
        temp=temp
    )
    partial_text = ""
    for i, token in enumerate(generator):
        print(token)
        if token == model.token_eos() or (max_new_tokens is not None and i >= max_new_tokens):
            break
        partial_text += model.detokenize([token]).decode("utf-8", "ignore")
        history[-1][1] = partial_text
    return partial_text


app = FastAPI()

origins = [""]
chunk_size = 250
chunk_overlap = 30
k_documents = 2
top_p = 0.9
top_k = 30
temp = 0.1
file_paths = ["resources/" + f for f in os.listdir("resources")]
db = build_index(file_paths, chunk_size, chunk_overlap)

chatbot = []

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/uploadfiles")
def create_upload_files(files: list[UploadFile]):
    global file_paths, db
    file_paths = file_paths + upload_files(files)
    for file in files:
        with open("resources/" + file.filename, "w", encoding='utf-8') as fin:
            fin.write(file.file.read().decode('utf-8'))
    db = build_index(file_paths, chunk_size, chunk_overlap)
    return 'Файлы добавлены'


@app.delete("/del_files")
def del_files(data=Body()):
    global file_paths, db
    del_paths = ["resources/" + f for f in data['msg']]
    for file in del_paths:
        if os.path.exists(file):
            os.remove(file)
    file_paths = [file for file in file_paths if file not in del_paths]
    db = build_index(file_paths, chunk_size, chunk_overlap)
    clear_chat()
    return 'Файлы удалены'


@app.post("/answer")
def question_answering(data=Body()):
    global chatbot
    print(chatbot)
    msg = data['msg']
    if file_paths:
        msg, chatbot = user(msg, chatbot)
        retrieved_docs = retrieve(chatbot, db, k_documents)
        result = bot(chatbot, retrieved_docs, top_p, top_k, temp)
        return {'msg': result}
    return {'msg': 'Модель пуста, добавьте файлы в модель'}


@app.post("/clear_chat")
def clear_chat():
    global chatbot
    chatbot = [["", ""]]


@app.get("/files_list")
def get_files_list():
    global file_paths
    return {'msg': file_paths}


@app.get("/dialog")
def get_dialog():
    global chatbot
    return chatbot


@app.post("/excel")
def create_excel(file: UploadFile):
    global chatbot, db
    with open(file.filename, "wb") as fin:
        fin.write(file.file.read())
    df = pd.read_excel(file.filename)
    answers = []
    writer = pd.ExcelWriter("output.xlsx", engine="xlsxwriter")
    for msg in df['Вопрос']:
        msg, chatbot = user(msg, chatbot)
        retrieved_docs = retrieve(chatbot, db, k_documents)
        result = bot(chatbot, retrieved_docs, top_p, top_k, temp)
        answers.append(result)
        # df['Ответ'] = answers
        # df.to_excel(writer)
        # writer.save()
        print('Записал!')
    df['Ответ'] = answers
    df.to_excel(writer, index=False)
    writer.close()


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
    df = pd.read_csv('resources/submission/sample_submission.csv', encoding='utf-8')
    for term in glossary:
        termin = term["text"].strip()
        df.loc[len(df.index)] = [audio_file, termin]
    df.to_csv(f'resources/outputs/submission_{audio_name}.csv', sep=',', encoding='utf-8', index=False)
    return "Check csv file in resources/outputs"
