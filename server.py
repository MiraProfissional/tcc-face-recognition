from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from multiprocessing import Process, Event, Manager
from typing import Dict
from pathlib import Path
import os
import re
import shutil
from datetime import datetime
from main_video import recognize_faces

IMAGES_DIR = Path(__file__).parent / "images"
IMAGES_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Face Recognition API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

manager = Manager()
streams: Dict[int, dict] = {}

@app.get("/")
def read_root():
    return {
        "status": "online",
        "active_streams": len(streams),
        "images_loaded": len(list(IMAGES_DIR.glob("*.jpg")))
    }

@app.post("/start/{discipline_id}")
def start_stream(discipline_id: int, cam: str = Query("0")):
    start_time = datetime.now()
    print(f"[API] Start stream: disciplina={discipline_id}, cam={cam}, start_time={start_time.isoformat()}")
    
    if discipline_id in streams:
        raise HTTPException(status_code=409, detail="Stream já ativa")
    
    try:
        src = int(cam) if cam.isdigit() else cam
        stop_evt = Event()
        ready_evt = Event()
        faces = manager.list()
        
        proc = Process(
            target=recognize_faces,
            args=(src, stop_evt, faces, ready_evt),
            daemon=True,
        )
        proc.start()
        
        # Aguardar confirmação de que a câmera abriu (timeout de 5 segundos)
        camera_ready = ready_evt.wait(timeout=5)
        
        if not camera_ready:
            # Câmera não abriu, terminar processo
            stop_evt.set()
            proc.join(timeout=2)
            if proc.is_alive():
                proc.terminate()
            raise HTTPException(
                status_code=503,
                detail=f"Não foi possível abrir a câmera {cam}. Verifique se ela está conectada e disponível."
            )
        
        streams[discipline_id] = {
            "proc": proc,
            "stop_evt": stop_evt,
            "faces": faces,
            "camera": cam,
            "start_time": start_time
        }
        
        print(f"[API] Stream iniciada: PID={proc.pid}")
        
        return {
            "success": True,
            "discipline_id": discipline_id,
            "camera": cam,
            "start_time": start_time.isoformat(),
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[API] Erro ao iniciar stream: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stop/{discipline_id}")
def stop_stream(discipline_id: int):
    stop_time = datetime.now()
    print(f"[API] Stop stream: disciplina={discipline_id}, stop_time={stop_time.isoformat()}")
    
    stream = streams.pop(discipline_id, None)
    if not stream:
        raise HTTPException(status_code=404, detail="Stream não encontrada")
    
    try:
        stream["stop_evt"].set()
        stream["proc"].join(timeout=2)
        
        if stream["proc"].is_alive():
            print("[API] Forcando termino do processo...")
            stream["proc"].terminate()
            stream["proc"].join(timeout=1)
        
        start_time = stream["start_time"]
        duration = (stop_time - start_time).total_seconds()
        
        print(f"[API] Stream parada: {len(stream['faces'])} matriculas em {duration:.1f}s")
        
        return {
            "success": True,
            "discipline_id": discipline_id,
            "camera": stream["camera"],
            "start_time": start_time.isoformat(),
            "stop_time": stop_time.isoformat(),
            "duration_seconds": duration,
            "faces_recognized": list(stream["faces"])
        }
        
    except Exception as e:
        print(f"[API] Erro ao parar stream: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-face", status_code=201)
async def upload_face(file: UploadFile = File(...)):
    print(f"[API] Upload: {file.filename}")
    
    try:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=415, detail="Apenas imagens são aceitas")
        
        # Validar nome do arquivo (deve começar com matrícula)
        filename = os.path.basename(file.filename)
        if not re.match(r"^\d+-", filename):
            raise HTTPException(
                status_code=400,
                detail="Nome do arquivo deve começar com a matrícula (ex: 123456-nome.jpg)"
            )
        
        # Salvar
        save_path = IMAGES_DIR / filename
        temp_path = None
        
        try:
            temp_path = IMAGES_DIR / f"temp_{datetime.now().timestamp()}_{filename}"
            
            with open(temp_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # Validar se é uma imagem válida
            import cv2
            img = cv2.imread(str(temp_path))
            if img is None:
                raise ValueError("Arquivo não é uma imagem válida")
            
            shutil.move(str(temp_path), str(save_path))
            
            registration = filename.split("-")[0]
            
            print(f"[API] Face cadastrada: {registration}")
            
            return {
                "success": True,
                "registration": registration,
                "filename": filename,
                "saved_path": str(save_path)
            }
            
        except Exception as e:
            if temp_path and temp_path.exists():
                temp_path.unlink()
            raise e
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"[API] Erro no upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
