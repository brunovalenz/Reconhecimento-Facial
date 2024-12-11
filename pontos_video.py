import cv2  
import dlib
import matplotlib.pyplot as plt

# Inicializa a captura de vídeo da câmera
video = cv2.VideoCapture(0)

# Inicializa o detector de rostos HOG (Histogram of Oriented Gradients) da dlib
HOG_detector = dlib.get_frontal_face_detector()

# Carrega o preditor de pontos faciais (68 pontos) da dlib
face_landmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    ret, frame = video.read()

    # Converte o frame capturado para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecta rostos no frame em escala de cinza usando o detector HOG
    faces = HOG_detector(gray)

    for face in faces:

        # Prediz os pontos faciais (landmarks) para o rosto detectado
        face_landmarks = face_landmark(gray, face)

        # Itera sobre os 68 pontos faciais
        for n in range(0, 68):
            # Obtém as coordenadas (x, y) do ponto facial n
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y

            # Desenha um círculo pequeno no frame na posição do ponto facial
            cv2.circle(frame, (x, y), 2, (0, 255, 0), 2)

    # Exibe o frame com os pontos faciais

    cv2.imshow('Video', frame)

    # Sair com 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Encerrar captura
video.release()
cv2.destroyAllWindows()