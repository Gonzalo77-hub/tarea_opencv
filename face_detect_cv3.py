import cv2
import sys

# Get user supplied values
#cimagen_a_escanear = "pantallazo.png"
# algoritmo_cascada = "haarcascade_frontalface_default.xml"
imagen_a_escanear = sys.argv[1]
algoritmo_cascada = sys.argv[2]

# Create the haar cascade
cascada_cara = cv2.CascadeClassifier(algoritmo_cascada)

# Read the image
imagen = cv2.imread(imagen_a_escanear) # lee la imagen a escanear y guarda la informacion
gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY) # convierte la imagen en una escala de grises para que el algoritmo lo pueda procesar

# Detect faces in the image
faces = cascada_cara.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
    #flags = cv2.CV_HAAR_SCALE_IMAGE
)

print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(imagen, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(imagen, 'Cara', (x,y-10), 2, 0.7,(255,0,0),2,cv2.LINE_AA)

cv2.imshow("Faces found", imagen)
cv2.waitKey(0)
