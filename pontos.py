import cv2  
import dlib
import matplotlib.pyplot as plt

# Inicializa o detector de rostos HOG (Histogram of Oriented Gradients) da dlib
HOG_detector = dlib.get_frontal_face_detector()

# Carrega o preditor de pontos faciais (68 pontos) da dlib
face_landmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Carrega uma imagem do caminho especificado
path = "exemplos/"
image = cv2.imread(path + "ex2.jpg")

# Converte a imagem carregada para escala de cinza
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detecta rostos na imagem em escala de cinza usando o detector HOG
face = HOG_detector(gray)

# Prediz os pontos faciais (landmarks) para o primeiro rosto detectado
face_landmarks = face_landmark(gray, face[0])

# Itera sobre os 68 pontos faciais
for n in range(0, 68):
    # Obtém as coordenadas (x, y) do ponto facial n
    x = face_landmarks.part(n).x
    y = face_landmarks.part(n).y

    # Desenha um círculo pequeno na imagem original na posição do ponto facial
    cv2.circle(image, (x, y), 2, (0, 255, 0), 12)

# Exibe a imagem com os pontos faciais
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()