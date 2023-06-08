import os
import cv2
import numpy as np
from tqdm import tqdm
from skimage import exposure, transform
import matplotlib.pyplot as plt
import h5py


DATADIR = "C:/Users/aleji/OneDrive/Escritorio/Sistemas inteligentes computacionales/test"
CATEGORIES = ["pistols", "rifles"]
IMAGE_SIZE = 100


def generar_datos():
    data = []
    label_counter = 0  # Contador para asignar números a las etiquetas
    for categoria in CATEGORIES:
        path = os.path.join(DATADIR, categoria)
        for imagen_nombre in tqdm(os.listdir(path), desc=categoria):
            try:
                imagen_ruta = os.path.join(path, imagen_nombre)
                imagen = cv2.imread(imagen_ruta)
                imagen = cv2.resize(imagen, (IMAGE_SIZE, IMAGE_SIZE)) 
                imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)               
                imagen = imagen.reshape(IMAGE_SIZE, IMAGE_SIZE, 1)
                data.append([imagen, label_counter])        
            except Exception as e:
                pass
        label_counter += 1
    
    np.random.shuffle(data)
    x = []
    y = []
    for par in tqdm(data, desc="Procesamiento"):
        x.append(par[0])
        y.append(par[1])
    
    x = np.array(x).astype('float32')  / 255
    
    with h5py.File('test.hdf5', 'w') as hf:
        # Crear grupos para X e Y
        x_grp = hf.create_group('X')
        y_grp = hf.create_group('Y')
        
        # Almacenar datos en cada grupo
        x_grp.create_dataset('imagenes', data=x)
        y_grp.create_dataset('etiquetas', data=y)
        
    print("test.hdf5 creado")

if __name__ == "__main__":
    generar_datos()
