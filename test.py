import h5py
import tensorflow as tf

#prueba con menos datos
with h5py.File('tes.hdf5', 'r') as hf:
    # Leer datos de los grupos X e Y
    x = hf['X/imagenes'][:]
    y = hf['Y/etiquetas'][:]

y = tf.keras.utils.to_categorical(y, 2)
modelo_new = tf.keras.models.load_model('model.h5')
loss, acc = modelo_new.evaluate(x, y, verbose=0)
print(loss,acc)