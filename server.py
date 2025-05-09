from fastapi import FastAPI
from fastapi.responses import JSONResponse
import threading
from main_video import recognize_faces, stop_recognition  # Importando o script de reconhecimento

app = FastAPI()

stream_thread = None

@app.get("/start_stream/")
async def start_stream():
    global stream_thread
    if stream_thread is None or not stream_thread.is_alive():
        # Criar uma thread para iniciar o reconhecimento
        stream_thread = threading.Thread(target=recognize_faces)
        stream_thread.start()
        return {"message": "Reconhecimento iniciado!"}
    return {"message": "O reconhecimento já está em andamento."}

@app.get("/stop_stream/")
async def stop_stream():
    global stop_recognition
    if stream_thread and stream_thread.is_alive():
        # Alterando a flag para parar o reconhecimento
        stop_recognition = True
        # Espera a thread finalizar a execução
        stream_thread.join()
        return {"message": "Reconhecimento interrompido!", "recognized_faces": recognize_faces}
    return {"message": "Nenhum stream ativo para parar."}