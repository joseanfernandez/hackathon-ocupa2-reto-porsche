#### 00 Comprobar TensorFlow en Python
```bash
!python -c 'import tensorflow as tf; print(tf.__version__)'
```
Si no está agegado:
```bash
pip install tensorflowjs
```
#### 01 InicioCrear carpeta 
Se crea en root por defecto
```bash
mkdir ~/reentrenar

pwd

cd /root/reentrenar

pwd
```
#### Traer imagen reentrenada a la carpeta
```bash
!curl -LO https://github.com/tensorflow/hub/raw/master/examples/image_retraining/retrain.py

ls
```
#### Crear dataset de imágenes
```bash
mkdir dataset

ls

mkdir dataset/porche/ dataset/no-porche/

ls dataset
```
#### Subir set de imagenes

#### Reentrenar modelo:

#### Solo con el dataset
```bash
ls

!python retrain.py --image_dir ./dataset
```
#### Reenternar modelo dependiente de mibilenet model[texto del enlace](https://)
```bash
!python retrain.py \
    --image_dir ./dataset \
    --tfhub_module https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/2
```
#### Modelo reentrenado en carpeta temporal: /tmp/output_graph.pb
- Archivo: **output_graph.pb**
- Archivo: **output_labels.txt**

```bash
ls /tmp
```
#### Reentrenar modelo: 
```bash
mkdir salida

!tensorflowjs_converter \
    --input_format=tf_hub \
    --output_format=tfjs_graph_model \
    --saved_model_tags=serve \
    /tmp/output_graph.pb \
    salida
```
#### Probando modelo con una imagen: prueba.jpg
```bash
!curl -LO https://github.com/tensorflow/tensorflow/raw/master/tensorflow/examples/label_image/label_image.py
!python label_image.py \
--graph=/tmp/output_graph.pb --labels=/tmp/output_labels.txt \
--input_layer=Placeholder \
--output_layer=final_result \
--image=prueba.jpg
```