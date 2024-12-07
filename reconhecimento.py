import face_recognition
import cv2
import os

# Caminho da pasta com as imagens conhecidas
known_folder = "rostos"

# Listas para armazenar codificações de rostos conhecidos e nomes
known_encodings = []
known_names = []

# Carregar e codificar cada imagem na pasta conhecida
for filename in os.listdir(known_folder):
    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
        # Carregar imagem e obter codificação
        image_path = os.path.join(known_folder, filename)
        image = face_recognition.load_image_file(image_path)
        
        # Verificar se a imagem tem pelo menos um rosto
        encodings = face_recognition.face_encodings(image)
        if encodings:  # Se a lista não estiver vazia
            known_encodings.append(encodings[0])

            # Usar o nome do arquivo (sem extensão) como nome da pessoa
            name = os.path.splitext(filename)[0]
            known_names.append(name)

# Iniciar a captura de vídeo
video_capture = cv2.VideoCapture(0)

while True:
    # Capturar frame
    ret, frame = video_capture.read()

    # Redimensionar o frame para processamento mais rápido
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Encontrar rostos no frame redimensionado
    face_locations = face_recognition.face_locations(small_frame)
    face_encodings = face_recognition.face_encodings(small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Ajustar coordenadas para o tamanho original do frame
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Verificar correspondências
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
        name = "Desconhecido"
        color = (0, 0, 255)  # Vermelho para rosto desconhecido

        # Se houver correspondência, pegar o nome do primeiro rosto correspondente
        if True in matches:
            match_index = matches.index(True)
            name = known_names[match_index]
            color = (0, 255, 0)  # Verde para rosto reconhecido

        # Desenhar retângulo ao redor do rosto
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # Exibir nome do rosto abaixo do retângulo
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)

    # Exibir o resultado
    cv2.imshow('Video', frame)

    # Sair com 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Encerrar captura
video_capture.release()
cv2.destroyAllWindows()
