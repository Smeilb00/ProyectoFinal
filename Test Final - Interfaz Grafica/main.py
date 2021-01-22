from tkinter import *
from tkinter import filedialog
from tkinter import ttk
import AnalizeDatapool, Trainer, Recognize
import time, datetime, os


rutaInput = ""
rutaDatapool = "C:/Users/Sbita/PycharmProjects/Test1/Datapool"
rutaOutput = "C:/Users/Sbita/PycharmProjects/Test1/Test Final - Interfaz Grafica/Output"


def ventanaTiempos(ts):
    root = Tk()
    root.title("Face Recognition")
    root.geometry("390x260")
    root.resizable(0, 0)
    root.iconbitmap("..\Input\proconsi.ico")

    tiempos = ttk.Treeview(root, columns="ts")
    tiempos.heading("#0", text="Tiempos")
    tiempos.heading("ts", text="Tiempo en segundos")
    tiempos.column("#0", minwidth=0, width=220)
    tiempos.column("ts", minwidth=0, width=130)
    item = tiempos.insert("", "end", text="Tiempo inicial: ", values=str(ts["t0"]))
    item = tiempos.insert("", "end", text="Leer datapool: ", values=str(ts["t1"]))
    item = tiempos.insert("", "end", text="Entrenar el sistema: ", values=str(ts["t2"]))
    item = tiempos.insert("", "end", text="Reconocer caras dentro de la imagen: ", values=str(ts["t3"]))
    tiempos.place(x=20, y=20)

    root.mainloop()

def uploadInput():
    global rutaInput
    path = filedialog.askopenfilename(initialdir=os.getcwd()+"/../")
    rutaInput = path

def uploadDatapool():
    path = filedialog.askdirectory(initialdir=os.getcwd()+"/../")
    rutaDatapool = path

def uploadOutput():
    path = filedialog.askdirectory(initialdir=os.getcwd()+"/../")
    rutaOutput = path
    print(rutaOutput)

def start(kernel):
    t0 = time.time()
    tiempos = {"t0": datetime.datetime.now().strftime("%H:%M:%S"), "t1": "", "t2": "", "t3": ""}
    if not os.path.exists(rutaOutput + "/embeddings.pickle"):
        AnalizeDatapool.run("Face_Detection_Model/", "openface.nn4.small2.v1.t7", rutaDatapool + "/", 0.95,
                            rutaOutput + "/embeddings.pickle")
    tiempos["t1"] = time.time() - t0
    t0 = time.time()
    if kernel.get() == 1:
        kern = "linear"
    elif kernel.get() == 2:
        kern = "rbf"
    else:
        kern = "poly"
    Trainer.run(rutaOutput + "/embeddings.pickle", rutaOutput + "/recognizer.pickle", rutaOutput + "/le.pickle", kern)
    tiempos["t2"] = time.time() - t0

    t0 = time.time()
    tfinal = Recognize.run("face_detection_model/", "openface.nn4.small2.v1.t7", rutaOutput + "/recognizer.pickle", rutaOutput + "/le.pickle",
                  rutaInput, 0.95, rutaOutput,t0)
    tiempos["t3"] = tfinal
    ventanaTiempos(tiempos)

def main():
    root = Tk()
    root.title("Face Recognition")
    root.geometry("370x200")
    #root.resizable(0, 0)
    root.iconbitmap("..\Input\proconsi.ico")

    # Añadimos objetos que utilizaremos después
    labelInput = Label(root, text="Imagen en la que se quiere realizar el reconocimiento:")
    labelInput.place(x=20, y=20)

    butUploadFile = Button(root, text="Select", command=uploadInput)
    butUploadFile.place(x=310, y=20)

    labelOutput = Label(root, text="Directorio en el que se guardará el resultado:")
    labelOutput.place(x=20, y=50)
    butUploadFolder1 = Button(root, text="Select", command=uploadOutput)
    butUploadFolder1.place(x=310, y=50)

    labelDatapool = Label(root, text="Directorio en el que se encuentra el datapool:")
    labelDatapool.place(x=20, y=80)
    butUploadFolder2 = Button(root, text="Select", command=uploadDatapool)
    butUploadFolder2.place(x=310, y=80)

    labelDatapool = Label(root, text="Algoritmo que se desea utilizar:")
    labelDatapool.place(x=20, y=110)

    kernel = IntVar()

    radioLinear = Radiobutton(root, text="Linear", variable=kernel, value=1)
    radioLinear.place(x=200, y=110)

    radioRBF = Radiobutton(root, text="RBF", variable=kernel, value=2)
    radioRBF.place(x=260, y=110)

    radioRBF = Radiobutton(root, text="Poly", variable=kernel, value=3)
    radioRBF.place(x=305, y=110)

    buttStart = Button(root, text="Iniciar Reconocimiento", command=lambda :start(kernel)).place(x=120, y=170)

    root.mainloop()

main()
