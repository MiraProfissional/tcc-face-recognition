import cv2
import time
import re
from typing import List, Union
from multiprocessing.synchronize import Event as EventType
from simple_facerec import SimpleFacerec


def extract_registration(label: str) -> str:
    """Extrai matrícula do label"""
    match = re.match(r"^(\d+)", label)
    return match.group(1) if match else None


def recognize_faces(src: Union[int, str], stop_evt: EventType, faces: List[str], ready_evt: EventType = None) -> None:
    print(f"Starting recognition for camera: {src}")
    
    # Converter src para int se for string numérica
    if isinstance(src, str) and src.isdigit():
        src = int(src)
    
    # Abrir câmera
    cap = cv2.VideoCapture(src)
    
    # Configurar timeout e buffer reduzido para resposta rápida ao stop
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print(f"ERROR: Could not open camera {src}")
        if ready_evt:
            ready_evt.clear()  # Sinaliza falha
        return
    
    print("Camera opened successfully")
    
    # Sinalizar que a câmera abriu com sucesso
    if ready_evt:
        ready_evt.set()
    
    # Carregar faces conhecidas
    sfr = SimpleFacerec()
    sfr.load_encoding_images("images/")
    
    print("Starting frame processing...")
    
    frame_count = 0
    
    while not stop_evt.is_set():
        # Verificar stop antes de ler frame
        if stop_evt.is_set():
            break
            
        ret, frame = cap.read()
        
        if not ret:
            # Verificar stop também quando não conseguir ler
            if stop_evt.is_set():
                break
            time.sleep(0.1)
            continue
        
        frame_count += 1
        
        # Detectar faces
        locations, names = sfr.detect_known_faces(frame)
        
        # Processar reconhecimentos
        for label in names:
            if label == "Unknown":
                continue
            
            registration = extract_registration(label)
            if registration and registration not in faces:
                faces.append(registration)
                print(f"New face recognized: {registration} (total: {len(faces)})")
        
        # Log a cada 100 frames
        if frame_count % 100 == 0:
            print(f"Frames: {frame_count}, Faces: {len(faces)}")
        
        # Verificar stop a cada frame também
        if stop_evt.is_set():
            break
    
    print(f"Processing finished: {frame_count} frames, {len(faces)} faces")
    
    cap.release()
    cv2.destroyAllWindows()
