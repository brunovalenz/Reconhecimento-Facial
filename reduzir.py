import cv2

# Carrega a imagem em alta resolução
image_path = "conhecido.jpg"
image = cv2.imread(image_path)

# Redimensiona a imagem para 500x500 pixels
resized_image = cv2.resize(image, (500, 500))

# Salva a imagem redimensionada
cv2.imwrite("resized.jpg", resized_image)
