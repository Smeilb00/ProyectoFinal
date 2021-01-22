import numpy
import pickle
import cv2
import os
import time


def run(detect, model, recon, inputle, input, umbralConfianza, t3):
    # Cargamos el detector entrenado
    print("[INFO] Cargamos el detector.")
    protoPath = os.path.sep.join([detect, "deploy.prototxt"])
    modelPath = os.path.sep.join([detect, "res10_300x300_ssd_iter_140000.caffemodel"])
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    # Cargamos ahora el modelo
    print("[INFO] Cargando el modelo de reconocimiento.")
    embedder = cv2.dnn.readNetFromTorch(model)

    # Ahora cargamos el sistema de reconocimiento entrenado
    recognizer = pickle.loads(open(recon, "rb").read())
    le = pickle.loads(open(inputle, "rb").read())

    # Cargamos la imagen en la que vamos a realizar el reconocimiento
    imagen = cv2.imread(input)
    (h, w) = imagen.shape[:2]

    # Contruimos el blob de esta imagen
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(imagen, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    # Ahora detectamos las caras en esta imagen y las guardamos
    detector.setInput(imageBlob)
    detections = detector.forward()

    # Todas estas detecciones las tenemos que comparar con el datapool
    for  i in range(0, detections.shape[2]):
        confianza = detections[0, 0, i, 2]
        if confianza > umbralConfianza:
            box = detections[0, 0, i, 3:7] * numpy.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extraemos los ROI
            face = imagen[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]
            # Comprobamos que cumpla el tamaño minimo
            if fW < 20 or fH < 20:
                continue
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # Clasificamos para reconocer las caras
            preds = recognizer.predict_proba(vec)[0]
            j = numpy.argmax(preds)
            proba = preds[j]
            nombre = le.classes_[j]

            # Por ultimo dibujamos la caja que lo contiene
            texto = "{}: {:.2f}%".format(nombre, proba * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(imagen, (startX, startY), (endX, endY), (255, 0, 0), 2)
            cv2.putText(imagen, nombre, (startX, startY - 6), cv2.FONT_HERSHEY_COMPLEX, 0.45, (0, 0, 200), 1)
    # show the output image
    print("[INFO] Reconocer todas las caras dentro de la imagen introducida tardó {} segundos.\n".format(time.time() - t3))
    cv2.imshow("Image", imagen)
    cv2.waitKey(0)