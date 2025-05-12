from fastapi import FastAPI, HTTPException, Query
from multiprocessing import Process, Event, Manager
from typing import Dict

from main_video import recognize_faces

app = FastAPI()
manager = Manager()

class StreamProc:
    def __init__(self, cam: str):
        self.stop_evt = Event()
        self.faces = manager.list()
        self.proc = Process(
            target=recognize_faces,
            args=(cam if not cam.isdigit() else int(cam), self.stop_evt, self.faces),
            daemon=True,
        )
        self.proc.start()

    def stop(self):
        self.stop_evt.set()
        self.proc.join(timeout=5)

streams: Dict[str, StreamProc] = {}

@app.post("/start/{turma}")
async def start_(turma: str, cam: str = Query("/dev/video0")):
    if turma in streams:
        raise HTTPException(409, "turma já em execução")
    streams[turma] = StreamProc(cam)
    return {"started": turma, "camera": cam}

@app.post("/stop/{turma}")
async def stop_(turma: str):
    task = streams.pop(turma, None)
    if not task:
        raise HTTPException(404, "turma não encontrada")
    task.stop()
    return {"stopped": turma, "faces": list(task.faces)}