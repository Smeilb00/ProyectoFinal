# -*- coding: latin-1 -*-
import cv2
import face_recognition
import os
import pickle
import time


def run():
    print("Sistema de reconocimiento facial T5.")
    print("Comienza el contador de tiempo.")
    arrayImages = []
    path = os.getcwd()
    pathingdatapool = path + "\datapool"
    tiempo1 = time.time()
    if not os.path.exists(path + '\output\saved_encodings.dat'):
        print("Se van a generar encodings.")
        faceEncodings = encoding_generator(arrayImages, pathingdatapool, path)
        print("Han tardado " + str(time.time()-tiempo1) + " segundos en geneararse los encodings.")
    else:
        with open(path + '\output\saved_encodings.dat', 'rb') as f:
            print("Los encodings ya estaban guardados en el archivo correspondiente.")
            faceEncodings = pickle.load(f)
            print("Han tardado " + str(time.time()-tiempo1) + " segundos en cargarse los encodings.")

    # Definimos la lista de nombres conocidos con el siguiente método
    nombres_conocidos = namelist_generator(os.listdir(pathingdatapool))

    # Necesitamos una fuente para escribir el recuadro
    font = cv2.FONT_HERSHEY_COMPLEX

    # La imagen donde se buscan las caras por ahora se carga directamente
    image_bw = cv2.cvtColor(cv2.imread(path + '\input\\imagen_input.jpg'),cv2.COLOR_BGR2GRAY)
    cv2.imwrite(path + '\input\imagen_inputbw.jpg', image_bw)

    img = face_recognition.load_image_file(path + '\input\imagen_inputbw.jpg')

    # Generamos los 3 arrays para los parámetros de los rostros de la imagen
    loc_rostros = face_recognition.face_locations(img)
    encoding_rostros = face_recognition.face_encodings(img, loc_rostros)

    # LLamamos al método que compara las caras encontradas con las que tenia almacenadas
    nombres_rostros = finding_faces(pathingdatapool, encoding_rostros, faceEncodings)

    # Dibujamos los cuadrados al rededor
    draw_squares(img, font, loc_rostros, nombres_rostros)

    # Abrimos una ventana con el resultado:
    cv2.imshow('Output', img)
    cv2.imwrite(path + '\output\Output.jpg', img)
    print("\nMostrando resultado. Pulsa cualquier tecla para salir.\n")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def encoding_generator(arrayimages, pathinghome, path):
    # Este método se encarga de generar el encoding de cada cara
    # y lo exporta a un archivo .dat
    faceEncodings = []
    i = 0
    pathfinal = os.listdir(pathinghome)
    for person in pathfinal:
        file = os.listdir(pathinghome + "\\" +  person)
        if (file.__len__() > 0):
            for imagen in file:
                arrayimages.append(face_recognition.load_image_file(pathinghome + "\\" + person + "\\" + imagen))

                # Para cada imagen generamos su encoding correspondiente
                faceEncodings.append(face_recognition.face_encodings(arrayimages[i])[0])
                i += 1

    with open(path + '\output\saved_encodings.dat', 'wb') as f:
        pickle.dump(faceEncodings, f)
    return faceEncodings


def namelist_generator(directorylist):
    # Este método genera una lista de nombres conocidos en función de
    # las carpetas existentes en la carpeta raíz datapool
    print("Se va a generar una lista de nombres conocidos.")
    tiempo2 = time.time()
    namelist = []
    for carpeta in directorylist:
        namelist.append(carpeta)
    print("Se ha generado una lista de nombres conocidos tardando un total de " + str(time.time() - tiempo2) + " segundos.")

    return namelist


def finding_faces(pathinghome, encoding_rostros, faceencodings):
    print("Se van a buscar las caras conocidas.")
    tiempo3 = time.time()
    nombres_rostros = []
    for encoding in encoding_rostros:
        # Array de Trues y Falses en funcion de si encuentra las caras en las que ya conoce
        coincidencias = face_recognition.compare_faces(faceencodings, encoding, tolerance=0.5)

        if True in coincidencias:
            encontrado = coincidencias.index(True)
            index = 0
            for carpeta in os.listdir(pathinghome):
                if not os.path.isfile(carpeta):
                    encontrado = encontrado - os.listdir(pathinghome+"\\"+carpeta).__len__()
                    if encontrado < 0:
                        nombre = carpeta.__str__()
                        break
        else:
            nombre = "???"

        nombres_rostros.append(nombre)
    print("Han tardado " + str(time.time()-tiempo3) + " segundos en encontrar las caras.")
    return nombres_rostros


def draw_squares(img, font, loc_rostros, nombres_rostros):
    for (top, right, bottom, left), nombre in zip(loc_rostros, nombres_rostros):

        # Cambiar el color segun el nombre:
        if nombre != "???":
            color = (0, 255, 0)  # Verde
        else:
            color = (0, 0, 255)  # Rojo

        # Dibujar los recuadros alrededor del rostro:
        cv2.rectangle(img, (left, top), (right, bottom), color, 2)
        cv2.rectangle(img, (left, bottom - 20), (right, bottom), color, -1)

        # Escribir el nombre de la persona:
        cv2.putText(img, nombre, (left, bottom - 6), font, 0.6, (0, 0, 0), 1)