import cv2

# Carrega a imagem em alta resolução
image_path = "samples/"
image_path = image_path + "6.jpeg"
image = cv2.imread(image_path)

# Redimensiona a imagem para 500x500 pixels
resized_image = cv2.resize(image, (500, 500))

# Salva a imagem redimensionada
cv2.imwrite("resized.jpg", resized_image)
