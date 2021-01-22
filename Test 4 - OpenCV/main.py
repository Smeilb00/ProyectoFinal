import argparse
import time
import os
import AnalizeDatapool, Trainer, Recognize


# Leemos todos los argumentos que necesitamos
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--datapool", required=True,
                help="path to input directory of faces + images")
ap.add_argument("-e", "--embeddings", required=True,
                help="path to output serialized db of facial embeddings")
ap.add_argument("-d", "--detector", required=True,
                help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding_model", required=True,
                help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-r", "--recognizer", required=True,
                help="path to output model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,
                help="path to output label encoder")
ap.add_argument("-in", "--image", required=True,
                help="path to input image")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# Ahora procedemos a crear una estructura en función del datapool introducido:
t1 = time.time()
if not os.path.exists(args["embeddings"]):
    AnalizeDatapool.run(args["detector"], args["embedding_model"], args["datapool"], args["confidence"], args["embeddings"])

print("[INFO] Generar un archivo de datos en base al datapool tardó {} segundos.\n".format(time.time() - t1))

# Entrenamos el sistema
t2 = time.time()
Trainer.run(args["embeddings"], args["recognizer"], args["le"])
print("[INFO] Entrenar el sistema de reconocimiento tardó {} segundos.\n".format(time.time() - t2))

# Ahora ejecutamos el sistema de reconocimiento
t3 = time.time()
Recognize.run(args["detector"], args["embedding_model"], args["recognizer"], args["le"], args["image"], args["confidence"], t3)

