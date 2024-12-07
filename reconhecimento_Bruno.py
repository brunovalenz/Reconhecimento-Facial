import face_recognition
import cv2

# Iniciar a captura de vídeo
video_capture = cv2.VideoCapture(0)

# Carregar e codificar o rosto conhecido
known_image = face_recognition.load_image_file("conhecido.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]

while True:
    # Capturar frame
    ret, frame = video_capture.read()

    # Encontrar rostos no frame
    face_locations = face_recognition.face_locations(frame, model="cnn")
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for face_encoding in face_encodings:
        # Verificar correspondência
        matches = face_recognition.compare_faces([known_encoding], face_encoding)

        if matches[0]:
            print("Rosto reconhecido!")
            cv2.putText(frame, "Reconhecido", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            print("Rosto não reconhecido.")
            cv2.putText(frame, "Desconhecido", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Exibir o resultado
    cv2.imshow('Video', frame)

    # Sair com 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Encerrar captura
video_capture.release()
cv2.destroyAllWindows()
