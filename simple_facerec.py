import face_recognition
import cv2
import os
import glob
import numpy as np


class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []

        # Resize frame for a faster speed
        self.frame_resizing = 0.25
        
        # Tolerance: equilibrado entre precisão e flexibilidade
        self.tolerance = 0.55

    def load_encoding_images(self, images_path):
        """
        Load encoding images from path
        :param images_path:
        :return:
        """
        # Load Images
        images_path = glob.glob(os.path.join(images_path, "*.*"))

        print("{} encoding images found.".format(len(images_path)))

        # Store image encoding and names
        for img_path in images_path:
            img = cv2.imread(img_path)
            
            # Validar se a imagem foi carregada
            if img is None:
                print(f"Warning: Could not load image {img_path}")
                continue
                
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Get the filename only from the initial file path.
            basename = os.path.basename(img_path)
            (filename, ext) = os.path.splitext(basename)
            
            # Get encoding
            encodings = face_recognition.face_encodings(rgb_img)
            
            # Validar se encontrou pelo menos uma face
            if len(encodings) == 0:
                print(f"Warning: No face found in {img_path}")
                continue
                
            img_encoding = encodings[0]

            # Store file name and file encoding
            self.known_face_encodings.append(img_encoding)
            self.known_face_names.append(filename)
        print("Encoding images loaded")

    def detect_known_faces(self, frame):
        small_frame = cv2.resize(
            frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        
        # Convert the image from BGR color (which OpenCV uses) to RGB color 
        # (which face_recognition uses)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            name = "Unknown"
            
            # Calcular distâncias para todas as faces conhecidas
            face_distances = face_recognition.face_distance(
                self.known_face_encodings, face_encoding)
            
            # Encontrar a menor distância
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                best_distance = face_distances[best_match_index]
                
                # Só aceitar se a distância for menor que tolerance
                if best_distance < self.tolerance:
                    name = self.known_face_names[best_match_index]
                    print(f"Match: {name} (distance: {best_distance:.3f})")
                else:
                    print(f"Rejected: closest was {self.known_face_names[best_match_index]} but distance {best_distance:.3f} > {self.tolerance}")
            
            face_names.append(name)

        # Convert to numpy array to adjust coordinates with frame resizing 
        # quickly
        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing
        return face_locations.astype(int), face_names
