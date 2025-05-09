import cv2
from simple_facerec import SimpleFacerec

PATH_IMAGES = 'images/'


sfr = SimpleFacerec()
sfr.load_encoding_images(PATH_IMAGES)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro: Não foi possível acessar a câmera!")
    exit()

recognized_faces = []

while True:
    ret, frame = cap.read()

    face_locations, face_names = sfr.detect_known_faces(frame)
    for face_loc, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        if (name.strip().lower() not in recognized_faces and name.strip().lower() != 'unknown'):
            recognized_faces.append(name.strip().lower())

        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)

    print(recognized_faces)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()