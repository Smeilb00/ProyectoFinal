# -*- coding: latin-1 -*-
import cv2
import face_recognition
import os
def run():
    # Cargamos todas las imagenes del datapool utilizando un bucle para recorrer las subcarpetas
    arrayImages = []
    faceEncodings = []
    i = 0
    pathing = r"C:\Users\Sbita\PycharmProjects\Test1\Datapool\\"
    pathfinal = os.listdir(pathing)
    for person in pathfinal:
        print(person)
        file = os.listdir(pathing + person)
        if (file.__len__() > 0):

            for imagen in file:
                arrayImages.append(face_recognition.load_image_file(pathing + person + "/" + imagen))

                # Para cada imagen generamos su encoding correspondiente
                faceEncodings.append(face_recognition.face_encodings(arrayImages[i])[0])
                i += 1


    # Creamos un array con los encodings y otro con sus respectivos nombres:

    nombres_conocidos = [
        "Ben Afflek",
        "Albert Einstein",
        "Elton John",
        "Jerry Seinfeld",
        "Madonna",
        "Mindy Kailing",
        "Paul Langevin",
        "Max Planck",
    ]

    # Cargamos una fuente de texto:
    font = cv2.FONT_HERSHEY_COMPLEX

    # Cargamos la imagen donde hay que identificar los rostros:
    img = face_recognition.load_image_file('Input\imagen_input.jpg')
    # (Para probar la segunda imagen hay que cambiar el argumento de la funci�n por 'imagen_input2.jpg')

    # Definir tres arrays, que servir�n para guardar los par�metros de los rostros que se encuentren en la imagen:
    loc_rostros = []  # Localizacion de los rostros en la imagen (contendr� las coordenadas de los recuadros que las contienen)
    encodings_rostros = []  # Encodings de los rostros
    nombres_rostros = []  # Nombre de la persona de cada rostro

    # Localizamos cada rostro de la imagen y extraemos sus encodings:
    loc_rostros = face_recognition.face_locations(img)
    encodings_rostros = face_recognition.face_encodings(img, loc_rostros)

    # Recorremos el array de encodings que hemos encontrado:
    for encoding in encodings_rostros:

        # Buscamos si hay alguna coincidencia con alg�n encoding conocido:
        coincidencias = face_recognition.compare_faces(faceEncodings, encoding, tolerance=0.48)

        # El array 'coincidencias' es ahora un array de booleanos.
        # Si contiene algun 'True', es que ha habido alguna coincidencia:
        if True in coincidencias:
            encontrado = coincidencias.index(True)
            index = 0
            for carpeta in os.listdir(pathing):
                if not os.path.isfile(carpeta):
                    encontrado = encontrado - os.listdir(pathing + carpeta).__len__()
                    if encontrado < 0:
                        nombre = carpeta.__str__()
                        break

        # Si no hay ning�n 'True' en el array 'coincidencias', no se ha podido identificar el rostro:
        else:
            nombre = "???"

        # A�adimos el nombre de la persona identificada en el array de nombres:
        nombres_rostros.append(nombre)

    # Dibujamos un recuadro rojo alrededor de los rostros desconocidos, y uno verde alrededor de los conocidos:
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

    # Abrimos una ventana con el resultado:
    cv2.imshow('Output', img)
    print("\nMostrando resultado. Pulsa cualquier tecla para salir.\n")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
