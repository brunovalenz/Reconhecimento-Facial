import face_recognition
import cv2

# Inicializa a captura de vídeo da câmera
video_capture = cv2.VideoCapture(0)

# Carrega a imagem conhecida e gera sua codificação facial
known_image = face_recognition.load_image_file("conhecido.jpg")
known_encodings = face_recognition.face_encodings(known_image)
if not known_encodings:
    print("Nenhum rosto foi detectado na imagem conhecida. Tente com outra imagem.")
    video_capture.release()
    exit()

# Verifica se o rosto foi detectado na imagem conhecida
if len(known_encodings) > 0:
    known_encoding = known_encodings[0]
    known_name = "Nome da Pessoa"  # Altere para o nome desejado
else:
    print("Nenhum rosto foi detectado na imagem conhecida. Tente com outra imagem.")
    video_capture.release()
    exit()

while True:
    # Captura um único frame de vídeo
    ret, frame = video_capture.read()

    # Reduz o tamanho do frame para 1/4 para processamento mais rápido
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Converte a imagem de BGR (OpenCV) para RGB (face_recognition)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Localiza todos os rostos no frame atual de vídeo
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    # Processa cada rosto detectado no frame de vídeo
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces([known_encoding], face_encoding)
        name = "Desconhecido"

        if True in matches:
            name = known_name

        # Escala as coordenadas do rosto de volta para o tamanho original do frame
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Desenha um retângulo ao redor do rosto
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Desenha um retângulo com o nome abaixo do rosto
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Exibe o frame resultante
    cv2.imshow('Video', frame)

    # Pressione 'q' no teclado para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a captura de vídeo e fecha todas as janelas
video_capture.release()
cv2.destroyAllWindows()
