from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from multiprocessing import Process, Event, Manager
from typing import Dict
from pathlib import Path
import os, shutil

from main_video import recognize_faces

# --- arquivos ---
IMAGES_DIR = Path("images")
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

def _save_fileobj(src_file, dst_path: Path):
    with open(dst_path, "wb") as out:
        shutil.copyfileobj(src_file, out)

def _extract_prefix(filename: str) -> str:
    base = os.path.basename(filename)
    prefix, sep, _ = base.partition("-")
    return prefix if sep and prefix.isdigit() else ""

# --- app/processos ---
app = FastAPI()
manager = Manager()

class StreamProc:
    def __init__(self, cam: str):
        self.stop_evt = Event()
        self.faces = manager.list()
        src = int(cam) if cam.isdigit() else cam
        self.proc = Process(
            target=recognize_faces,
            args=(src, self.stop_evt, self.faces),
            daemon=True,
        )
        self.proc.start()

    def stop(self):
        self.stop_evt.set()
        self.proc.join(timeout=5)

streams: Dict[str, StreamProc] = {}

@app.get("/")
async def root():
    return {"message": "Bem-vindo à API Face Recognition"}

@app.post("/start/{turma_id}")
async def start_(turma_id: int, cam: str = Query("/dev/video0")):
    if turma_id in streams:
        raise HTTPException(409, "turma já em execução")
    streams[turma_id] = StreamProc(cam)
    return {"started": True, "turma": turma_id, "camera": cam}

@app.post("/stop/{turma_id}")
async def stop_(turma_id: int):
    task = streams.pop(turma_id, None)
    if not task:
        raise HTTPException(404, "turma não encontrada")
    task.stop()
    return {"stopped": turma_id, "faces": list(task.faces)}

@app.post("/upload-face", status_code=201)
async def upload_face(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "filename ausente")
    if not (file.content_type and file.content_type.startswith("image/")):
        raise HTTPException(415, "somente image/* é aceito")

    safe_name = os.path.basename(file.filename)
    prefix = _extract_prefix(safe_name)
    if not prefix:
        raise HTTPException(400, "filename deve iniciar com numeros seguidos de '-'")

    # verifica colisão por prefixo único
    matches = list(IMAGES_DIR.glob(f"{prefix}-*"))
    if matches:
        await file.close()
        raise HTTPException(
            409,
            detail={"message": "já existe arquivo com este prefixo",
                    "prefix": prefix,
                    "existing": [m.name for m in matches]},
        )

    dst = IMAGES_DIR / safe_name
    if dst.exists():
        await file.close()
        raise HTTPException(409, "filename já existente")

    try:
        await run_in_threadpool(_save_fileobj, file.file, dst)
    finally:
        await file.close()

    return {"message": "ok", "file": safe_name, "path": str(dst)}