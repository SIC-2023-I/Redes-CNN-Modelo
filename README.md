# Redes-CNN-Modelo

## Funcionamiento

El programa se encarga de entrenar una red neuronal convolucional para la clasificación de imágenes de dos tipos de arams. Para ello, se utiliza un dataset de 8000 imágenes, 4000 de cada clase. El dataset se divide en un 85% imágenes para entrenamiento y 15% para validación. El programa se encarga de entrenar la red neuronal con las imágenes de entrenamiento y posteriormente se evalúa el modelo con las imágenes de validación. Finalmente, se muestra la precisión del modelo.

## Ejecución

Para ejecutar el programa de debe tener instalado:

- Python 3.6 o superior
- Tensorflow 2.0 o superior
- Keras 2.3.1 o superior
- Numpy 1.18.1 o superior
- Matplotlib 3.1.3 o superior

Para ejecutar el programa se debe ejecutar el siguiente comando:

```bash
python -m http.server 8000  # Python 3
```

Después de ejecutar el comando, se debe abrir el navegador y escribir la siguiente dirección:

```bash
http://localhost:8000/
```