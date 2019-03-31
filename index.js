/**
 * @license
 * Copyright 2018-2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tf from '@tensorflow/tfjs';

import {IMAGENET_CLASSES} from './imagenet_classes';

/* El Modelo */
const MOBILENET_MODEL_PATH = 'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json';
//const MOBILENET_MODEL_PATH = 'modelo_reenternado.pb';

const IMAGE_SIZE = 224;
const TOPK_PREDICTIONS = 10;

let mobilenet;
const mobilenetDemo = async () => {
  status('Cargando Modelo...');

  mobilenet = await tf.loadLayersModel(MOBILENET_MODEL_PATH);

  // Calienta el modelo. Esto no es necesario, pero hace que la primera predicción
  // más rápido. Llamada `dispose` para liberar la memoria WebGL asignada para el retorno
  // valor de "predecir".
  mobilenet.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])).dispose();

  status('');

  // Haga una predicción a través del cat.jpg alojado localmente.
  const catElement = document.getElementById('porche');
  if (catElement.complete && catElement.naturalHeight !== 0) {
    predict(catElement);
    catElement.style.display = '';
  } else {
    catElement.onload = () => {
      predict(catElement);
      catElement.style.display = '';
    }
  }

  document.getElementById('file-container').style.display = '';
};

/**
 * A partir de un elemento de imagen, realiza una predicción a través de mobilenet devolviendo el
 * probabilidades de las mejores clases K.
 */
async function predict(imgElement) {
  status('Predicting...');

  // La primera hora de inicio incluye el tiempo que se tarda en extraer la imagen
  // del HTML y procesarlo previamente, además de la llamada predict().
  const startTime1 = performance.now();
  // La segunda hora de arranque excluye la extracción y el preprocesamiento, y
  // incluye sólo la llamada predict().
  let startTime2;
  const logits = tf.tidy(() => {
    // tf.browser.fromPixels() devuelve un Tensor de un elemento de imagen.
    const img = tf.browser.fromPixels(imgElement).toFloat();

    const offset = tf.scalar(127.5);
    // Normaliza la imagen de [0, 255] a [-1, 1].
    const normalized = img.sub(offset).div(offset);

    // Reestructurar a un lote de un solo elemento para que podamos pasarlo a predecir.
    const batched = normalized.reshape([1, IMAGE_SIZE, IMAGE_SIZE, 3]);

    startTime2 = performance.now();
    // Haga una predicción a través de mobilenet.
    return mobilenet.predict(batched);
  });

  // Convertir logits en probabilidades y nombres de clases.
  const classes = await getTopKClasses(logits, TOPK_PREDICTIONS);
  const totalTime1 = performance.now() - startTime1;
  const totalTime2 = performance.now() - startTime2;
  status(`Hecho en ${Math.floor(totalTime1)} ms ` +
      `(sin incluir procesamiento: ${Math.floor(totalTime2)} ms)`);

  // Muestre las clases en el DOM.
  showResults(imgElement, classes);
}

/**
 * Calcula las probabilidades de las clases topK dadas logits computando
 * softmax para obtener probabilidades y luego ordenarlas.
 * @param logits Tensor representando los registros de MobileNet.
 * @param topK El número de las mejores predicciones a mostrar.
 */
export async function getTopKClasses(logits, topK) {
  const values = await logits.data();

  const valuesAndIndices = [];
  for (let i = 0; i < values.length; i++) {
    valuesAndIndices.push({value: values[i], index: i});
  }
  valuesAndIndices.sort((a, b) => {
    return b.value - a.value;
  });
  const topkValues = new Float32Array(topK);
  const topkIndices = new Int32Array(topK);
  for (let i = 0; i < topK; i++) {
    topkValues[i] = valuesAndIndices[i].value;
    topkIndices[i] = valuesAndIndices[i].index;
  }

  const topClassesAndProbs = [];
  for (let i = 0; i < topkIndices.length; i++) {
    topClassesAndProbs.push({
      className: IMAGENET_CLASSES[topkIndices[i]],
      probability: topkValues[i]
    })
  }
  return topClassesAndProbs;
}

//
// UI
//

function showResults(imgElement, classes) {
  const predictionContainer = document.createElement('div');
  predictionContainer.className = 'pred-container';

  const imgContainer = document.createElement('div');
  imgContainer.appendChild(imgElement);
  predictionContainer.appendChild(imgContainer);

  const probsContainer = document.createElement('div');
  for (let i = 0; i < classes.length; i++) {
    const row = document.createElement('div');
    row.className = 'row';

    const classElement = document.createElement('div');
    classElement.className = 'cell';
    classElement.innerText = classes[i].className;
    row.appendChild(classElement);

    const probsElement = document.createElement('div');
    probsElement.className = 'cell';
    probsElement.innerText = classes[i].probability.toFixed(3);
    row.appendChild(probsElement);

    probsContainer.appendChild(row);
  }
  predictionContainer.appendChild(probsContainer);

  predictionsElement.insertBefore(
      predictionContainer, predictionsElement.firstChild);
}

const filesElement = document.getElementById('files');
filesElement.addEventListener('change', evt => {
  let files = evt.target.files;
  // Display thumbnails & issue call to predict each image.
  for (let i = 0, f; f = files[i]; i++) {
    // Only process image files (skip non image files)
    if (!f.type.match('image.*')) {
      continue;
    }
    let reader = new FileReader();
    const idx = i;
    // Closure to capture the file information.
    reader.onload = e => {
      // Fill the image & call predict.
      let img = document.createElement('img');
      img.src = e.target.result;
      img.width = IMAGE_SIZE;
      img.height = IMAGE_SIZE;
      img.onload = () => predict(img);
    };

    // Read in the image file as a data URL.
    reader.readAsDataURL(f);
  }
});

const demoStatusElement = document.getElementById('status');
const status = msg => demoStatusElement.innerText = msg;

const predictionsElement = document.getElementById('predictions');

mobilenetDemo();
