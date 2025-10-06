import os
import re
import cv2
from typing import List
from simple_facerec import SimpleFacerec

# Extrai os dígitos do começo do rótulo (que costuma ser o nome do arquivo sem extensão).
# Ex.: "987654323-Vitor-Mira-....jpg" -> "987654323"
def extract_registration(label: str) -> str | None:
    # Garante que estamos trabalhando só com o “nome”, caso venha com caminho.
    base = os.path.basename(label)
    # Remove extensão, se vier
    base = os.path.splitext(base)[0]
    # Pega só os dígitos do começo (mais seguro do que split por "-")
    m = re.match(r"^(\d+)", base)
    return m.group(1) if m else None

def recognize_faces(src, stop_evt, faces: List[str], linux_backend: bool = True):
    # No Linux, CAP_V4L2 ajuda; no Windows use apenas cv2.VideoCapture(src)
    cap = cv2.VideoCapture(src, cv2.CAP_V4L2) if linux_backend else cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[ERRO] não abriu {src}")
        return

    sfr = SimpleFacerec()
    sfr.load_encoding_images("images/")

    while not stop_evt.is_set():
        ok, frame = cap.read()
        if not ok:
            break

        locs, names = sfr.detect_known_faces(frame)

        for (_, _, _, _), label in zip(locs, names):
            if not label or label.strip().lower() == "unknown":
                continue

            reg = extract_registration(label)
            if not reg:
                # Se por algum motivo o arquivo não começar com dígitos, ignora
                continue

            if reg not in faces:
                faces.append(reg)

    cap.release()