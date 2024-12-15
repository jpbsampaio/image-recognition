import cv2

imagem = cv2.imread('content/img1.jpg')

classificador_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
classificador_olho = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
classificador_sorriso = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

faces = classificador_face.detectMultiScale(cinza, scaleFactor=1.3, minNeighbors=5)

for (x, y, largura, altura) in faces:
    cv2.rectangle(imagem, (x, y), (x+largura, y+altura), (255, 0, 0), 2)
    regiao_cinza = cinza[y:y+altura, x:x+largura]
    regiao_cor = imagem[y:y+altura, x:x+largura]

    olhos = classificador_olho.detectMultiScale(regiao_cinza)
    for (ex, ey, elargura, ealtura) in olhos:
        cv2.rectangle(regiao_cor, (ex, ey), (ex+elargura, ey+ealtura), (0, 255, 0), 2)

    sorrisos = classificador_sorriso.detectMultiScale(regiao_cinza, scaleFactor=1.8, minNeighbors=20)
    for (sx, sy, slargura, saltura) in sorrisos:
        cv2.rectangle(regiao_cor, (sx, sy), (sx+slargura, sy+saltura), (0, 0, 255), 2)

cv2.imshow('rosto', imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()