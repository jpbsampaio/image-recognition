import cv2

cascata_rosto = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cascata_olho = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
cascata_sorriso = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

captura = cv2.VideoCapture(0)

while captura.isOpened():
    ret, quadro = captura.read()
    if not ret:
        break

    cinza = cv2.cvtColor(quadro, cv2.COLOR_BGR2GRAY)

    rostos = cascata_rosto.detectMultiScale(cinza, scaleFactor=1.3, minNeighbors=5)
    for (x, y, l, a) in rostos:
        cv2.rectangle(quadro, (x, y), (x+l, y+a), (255, 0, 0), 2)
        regiao_cinza = cinza[y:y+a, x:x+l]
        regiao_cor = quadro[y:y+a, x:x+l]

        olhos = cascata_olho.detectMultiScale(regiao_cinza, scaleFactor=1.1, minNeighbors=10)
        for (ex, ey, el, ea) in olhos:
            cv2.rectangle(regiao_cor, (ex, ey), (ex+el, ey+ea), (0, 255, 0), 2)

        sorrisos = cascata_sorriso.detectMultiScale(regiao_cinza, scaleFactor=1.8, minNeighbors=20)
        for (sx, sy, sl, sa) in sorrisos:
            cv2.rectangle(regiao_cor, (sx, sy), (sx+sl, sy+sa), (0, 0, 255), 2)

    cv2.imshow('video', quadro)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

captura.release()
cv2.destroyAllWindows()
