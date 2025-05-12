import cv2
from typing import List
from simple_facerec import SimpleFacerec

def recognize_faces(src, stop_evt, faces: List[str]):
    cap = cv2.VideoCapture(src, cv2.CAP_V4L2)     # evita fallback GStreamer
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
        for (_, _, _, _), n in zip(locs, names):
            n = n.strip().lower()
            if n != "unknown" and n not in faces:
                faces.append(n)

    cap.release()