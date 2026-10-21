# Reconocedor de dígitos MNIST

Aplicación de escritorio en **Python** para **entrenar, usar, evaluar y comparar redes neuronales** que reconocen dígitos manuscritos (0–9). Está construida con **TensorFlow/Keras** y una interfaz gráfica en **Tkinter**, y reúne en una sola ventana todo el ciclo de trabajo de un modelo de aprendizaje profundo.

> Proyecto de la asignatura **Redes Neuronales** · Ingeniería de Sistemas · Universidad Nacional del Altiplano.

---

## Tabla de contenido

1. [Características](#características)
2. [Requisitos](#requisitos)
3. [Instalación](#instalación)
4. [Ejecución](#ejecución)
5. [Estructura del proyecto](#estructura-del-proyecto)
6. [Manual de uso](#manual-de-uso)
7. [Arquitecturas disponibles](#arquitecturas-disponibles)
8. [Hiperparámetros: qué hace cada uno](#hiperparámetros-qué-hace-cada-uno)
9. [Mejor configuración probada](#mejor-configuración-probada)
10. [Cómo funciona por dentro](#cómo-funciona-por-dentro)
11. [Archivos que genera la aplicación](#archivos-que-genera-la-aplicación)
12. [Atajos de teclado](#atajos-de-teclado)
13. [Solución de problemas](#solución-de-problemas)
14. [Autor](#autor)

---

## Características

- **5 pestañas** que siguen el flujo de trabajo: Entrenar → Reconocer → Evaluar → Dataset → Experimentos.
- **4 arquitecturas**: MLP, LeNet-5, CNN simple y CNN profunda.
- **6 optimizadores**: Adam, SGD, SGD + Momentum, RMSprop, Nadam y Adagrad.
- **4 programaciones de la tasa de aprendizaje**: constante, decaimiento exponencial, coseno y reducción en meseta.
- Regularización con **Dropout** y **L2**, **early stopping** y **aumento de datos**.
- **Curvas en vivo** de pérdida, precisión y tasa de aprendizaje, con registro por época.
- **Barrido automático** de hiperparámetros (entrena varias veces cambiando una variable).
- **Lienzo de dibujo** con predicción en **tiempo real**, barras de probabilidad e historial.
- **Mapas de activación** de las capas convolucionales.
- **Evaluación completa**: precisión, recall, F1 por dígito, matriz de confusión y errores más seguros.
- **Historial de experimentos** persistente, comparación de curvas y exportación a CSV.
- El entrenamiento corre en un **hilo aparte**: la interfaz nunca se congela.

## Requisitos

### Software

| Requisito | Versión |
|---|---|
| Python | 3.10 – 3.12 (recomendado) |
| Tkinter | Incluido con Python en Windows y macOS; en Linux se instala aparte |
| Conexión a internet | Solo la primera vez, para descargar MNIST |

### Dependencias de Python (`requirements.txt`)

```txt
tensorflow>=2.16
numpy
matplotlib
pillow
```

| Paquete | Uso |
|---|---|
| `tensorflow` (Keras) | Construcción, entrenamiento, evaluación y guardado de modelos |
| `numpy` | Matrices, métricas y matriz de confusión |
| `matplotlib` | Gráficas incrustadas en la interfaz |
| `pillow` | Dibujo del lienzo, reescalado y conversión de imágenes |

### Hardware

No requiere GPU. Con CPU, la CNN profunda con el 100 % de los datos y 25 épocas tardó unos 16 minutos; la CNN simple con 20 % de datos y 3 épocas tarda solo unos segundos.

## Instalación

### 1. Obtener el proyecto

```bash
git clone https://github.com/USUARIO/reconocedor-mnist.git
cd reconocedor-mnist
```

(o descomprimir el `.zip` y entrar a la carpeta `reconocedor_mnist`).

### 2. Crear un entorno virtual (recomendado)

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar las dependencias

```bash
pip install -r requirements.txt
```

### 4. Linux: instalar Tkinter si falta

```bash
sudo apt install python3-tk
```

## Ejecución

```bash
python main.py
```

Aparece una pantalla de carga mientras se importa TensorFlow (unos segundos) y después la ventana principal. **La primera vez** se descarga MNIST automáticamente; el botón *Entrenar* se activa al terminar.

## Estructura del proyecto

```text
reconocedor_mnist/
├── main.py                  # Punto de entrada y pantalla de carga
├── app.py                   # Ventana principal, datos, modelo activo y cola de eventos
├── config.py                # Rutas, tema visual y valores por defecto
├── nucleo.py                # Modelos, optimizadores, preprocesado y métricas
├── entrenador.py            # Entrenamiento en un hilo aparte
├── ui_utils.py              # Estilo ttk y funciones auxiliares de interfaz
├── requirements.txt         # Dependencias
├── tabs/
│   ├── __init__.py
│   ├── tab_entrenamiento.py # 1 · Entrenar
│   ├── tab_reconocer.py     # 2 · Reconocer
│   ├── tab_evaluacion.py    # 3 · Evaluar
│   ├── tab_dataset.py       # 4 · Dataset
│   └── tab_experimentos.py  # 5 · Experimentos
│
├── modelo_mnist.keras       # (se genera) modelo activo
├── modelos/                 # (se genera) un .keras por entrenamiento
└── experimentos.json        # (se genera) historial de experimentos
```

| Archivo | Responsabilidad |
|---|---|
| `main.py` | Crea la pantalla de carga y lanza la aplicación |
| `app.py` | Clase `App`: pestañas, carga de MNIST, modelo activo, menú, registro de experimentos y despacho de eventos |
| `config.py` | Constantes: rutas, colores, listas de opciones y valores por defecto (`DEFECTOS`) |
| `nucleo.py` | `construir_modelo`, `crear_optimizador`, `preparar_imagen`, `matriz_confusion`, `metricas_por_clase` |
| `entrenador.py` | Clase `Entrenador` (hilo): compila, entrena, evalúa y guarda cada configuración |
| `ui_utils.py` | Tema oscuro de ttk y helpers (`combo`, `deslizador`, `incrustar`, …) |

## Manual de uso

### Inicio rápido (2 minutos)

1. Ejecuta `python main.py`.
2. En **1 · Entrenar** pulsa **▶ Entrenar** (los valores por defecto ya funcionan).
3. Al terminar, pulsa **Ir a 2 · Reconocer →**.
4. Dibuja un número en el lienzo y mira la predicción.

---

### Pestaña 1 · Entrenar

Reúne todo lo que se puede modificar antes y durante un entrenamiento. Un **banner** superior indica el estado (preparando datos, entrenando, sin modelo, o modelo listo).

**Grupo «Modelo y optimización»**

| Control | Qué hace |
|---|---|
| Arquitectura | MLP, LeNet-5, CNN simple o CNN profunda |
| Optimizador | Cómo se actualizan los pesos |
| Tasa de aprendizaje | Tamaño del paso en cada actualización (0.1 … 0.0001, editable) |
| Programación del LR | Cómo cambia la tasa durante el entrenamiento |
| Dropout | Porcentaje de neuronas que se apagan al entrenar (0 – 0.7) |
| Regularización L2 | Penalización de pesos grandes (0, 0.0001, 0.001, 0.01) |

**Grupo «Entrenamiento y datos»**

| Control | Qué hace |
|---|---|
| Épocas | Pasadas completas por los datos (1 – 200) |
| Tamaño de batch | Imágenes por actualización (32 – 512) |
| Validación (%) | Parte de los datos reservada para vigilar el aprendizaje (5 – 30 %) |
| Datos usados (%) | Fracción de las 60 000 imágenes que se emplea (5 – 100 %) |
| Early stopping + paciencia | Detiene el entrenamiento si `val_accuracy` no mejora y restaura los mejores pesos |
| Aumento de datos | Rotación, traslación y zoom aleatorios solo durante el entrenamiento |
| Semilla | Fija la aleatoriedad para reproducir resultados |

**Grupo «Acciones»**

| Botón | Qué hace |
|---|---|
| ▶ Entrenar | Valida el formulario y entrena un modelo |
| ■ Detener | Corta el entrenamiento en el siguiente lote |
| Ver capas | Muestra el `summary()` de la arquitectura elegida en el registro |
| Ejecutar barrido | Entrena varias veces cambiando **una** variable |

**Barrido automático.** Elige la variable (tasa de aprendizaje, optimizador, arquitectura, dropout o batch) y escribe los valores separados por comas, por ejemplo `0.01, 0.001, 0.0001`. El resto de parámetros se toma del formulario. Al terminar, **el mejor modelo según la precisión de prueba pasa a ser el modelo activo**.

**Resultados.** Se dibujan tres gráficas (pérdida, precisión y tasa de aprendizaje) y un registro con una línea por época:

```text
Época  3/25  loss 0.0412  acc 0.9871  │ val_loss 0.0388  val_acc 0.9893  │ lr 9.37e-04
```

Al final se evalúa con las 10 000 imágenes de prueba y el modelo se guarda en `modelos/`.

**Cómo interpretar las curvas**

- `loss` baja pero `val_loss` sube → **sobreajuste** (más dropout, L2 o aumento de datos).
- Ambas altas → **subajuste** (más épocas, red más grande o mayor tasa de aprendizaje).
- Curvas casi iguales y bajas → el modelo generaliza bien.

---

### Pestaña 2 · Reconocer

Dibuja un dígito con el ratón y la red lo reconoce **en tiempo real**.

| Elemento | Descripción |
|---|---|
| Lienzo (400×400) | Fondo negro, trazo blanco |
| Grosor (10 – 60) | Ancho del pincel; un trazo grueso se parece más a MNIST |
| Tiempo real | Predice mientras se dibuja (≈25 veces por segundo); si se desactiva, predice al soltar el ratón |
| Limpiar / Deshacer | Borra todo / elimina el último trazo |
| Guardar | Exporta el dibujo como PNG |
| Abrir imagen | Reconoce una imagen externa (PNG, JPG, BMP) |
| Predicción | Dígito más probable y su confianza |
| Barras | Probabilidad de cada dígito (la ganadora en verde) |
| «Lo que ve la red» | La imagen 28×28 que recibe la red tras el preprocesado |
| Historial | Últimos 30 reconocimientos (dígito, confianza y hora) |

**Preprocesado.** Antes de predecir, el dibujo se transforma para parecerse a MNIST: escala de grises → inversión si el fondo es claro → eliminación de ruido → recorte del trazo → reescalado a 20×20 → centrado en 28×28 por **centro de masa** → normalización a [0, 1].

**Confianza.** La red siempre devuelve probabilidades que suman 1, incluso si el trazo no es un dígito. Una confianza alta indica un dígito claro; una confianza baja o barras repartidas indican un dibujo ambiguo.

**Mapas de activación.** El botón abre una ventana con la entrada 28×28 y, para las tres primeras capas convolucionales, **ocho filtros** de cada una (mapa de colores *magma*; zonas claras = activación alta). Sirven para:

- Interpretar qué rasgos detecta cada filtro (bordes, curvas, cruces).
- Comprobar que la red aprendió características útiles.
- Diagnosticar por qué falla un dígito mal dibujado o descentrado.

Las capas iniciales detectan trazos simples; las más profundas combinan esos trazos en formas más abstractas. Con un **MLP** no hay mapas (no tiene convoluciones) y la aplicación lo avisa.

---

### Pestaña 3 · Evaluar

Pulsa **Evaluar modelo activo** para medirlo con las **10 000 imágenes de prueba**, que nunca se usaron para entrenar.

- **Resumen:** exactitud, pérdida y número de errores.
- **Tabla por dígito:** precisión, recall, F1 y soporte, más la fila *Macro*.
- **Matriz de confusión 10×10:** filas = real, columnas = predicho; la diagonal son los aciertos.
- **15 errores más seguros:** imágenes mal clasificadas con mayor confianza (`R:real P:predicho`).

Definiciones, para cada clase *k*:

```text
Precisión = TP / (TP + FP)        Recall = TP / (TP + FN)
F1 = 2 · Precisión · Recall / (Precisión + Recall)
```

---

### Pestaña 4 · Dataset

Explora los datos con los que se entrena: **60 000 imágenes de entrenamiento** y **10 000 de prueba**, de 28×28 píxeles en escala de grises y 10 clases. La barra superior muestra además el píxel medio y su desviación.

| Vista | Para qué sirve |
|---|---|
| Muestras aleatorias | Verificar imágenes y etiquetas y ver la variedad de escritura (botón *Otras muestras*) |
| Distribución de clases | Comprobar que el conjunto está balanceado (≈ 5 400 – 6 700 imágenes por dígito) |
| Dígito promedio | Ver el «prototipo» de cada número |

---

### Pestaña 5 · Experimentos

Cada entrenamiento se registra en `experimentos.json` y **se conserva al cerrar la aplicación**.

- **Tabla:** arquitectura, optimizador, LR, programación, épocas, batch, dropout, aumento, precisión de validación y de prueba, parámetros y tiempo.
- **Seleccionar filas:** superpone sus curvas para compararlas.
- **Doble clic / Usar:** carga ese modelo como modelo activo.
- **Eliminar:** borra el experimento del historial.
- **Exportar CSV:** guarda la tabla para Excel o informes.

---

### Menú

| Menú | Opción | Acción |
|---|---|---|
| Archivo | Cargar modelo… | Abre un `.keras` o `.h5` y lo activa |
| Archivo | Guardar modelo activo como… | Copia el modelo activo a otra ruta |
| Ayuda | Guía rápida / Atajos / Acerca de | Información de uso |

## Arquitecturas disponibles

| Arquitectura | Capas | Parámetros |
|---|---|---:|
| **MLP** | Flatten → Dense(256, ReLU) → Dropout → Dense(128, ReLU) → Dropout → Dense(10, softmax) | 235 146 |
| **LeNet-5** | Conv(6, 5×5, tanh) → AvgPool → Conv(16, 5×5, tanh) → AvgPool → Flatten → Dense(120) → Dense(84) → Dropout → Dense(10, softmax) | 61 706 |
| **CNN simple** | Conv(32) → MaxPool → Conv(64) → MaxPool → Flatten → Dropout → Dense(10, softmax) | 34 826 |
| **CNN profunda** | [Conv(32) → BN]×2 → MaxPool → Dropout → [Conv(64) → BN]×2 → MaxPool → Dropout → Flatten → Dense(128) → BN → Dropout → Dense(10, softmax) | 469 098 |

- **MLP:** línea base; ignora la estructura espacial de la imagen.
- **LeNet-5:** arquitectura histórica de Yann LeCun (1998), creada para dígitos.
- **CNN simple:** buen equilibrio entre velocidad y precisión (opción por defecto).
- **CNN profunda:** la más precisa y la más lenta; usa Batch Normalization.

**Funciones de activación:** ReLU en MLP y CNN, `tanh` en LeNet-5 y **softmax** en la capa de salida de todas.

## Hiperparámetros: qué hace cada uno

### Optimizadores

| Optimizador | Descripción |
|---|---|
| Adam | Momento + tasas adaptativas por parámetro; converge rápido y con poco ajuste (por defecto) |
| SGD | Descenso de gradiente clásico; simple, pero lento y sensible a la tasa |
| SGD + Momentum | SGD con momento 0.9; atraviesa mejor valles y mesetas |
| RMSprop | Divide el gradiente por una media móvil de su magnitud |
| Nadam | Adam con momento de Nesterov |
| Adagrad | Reduce la tasa de parámetros muy actualizados |

### Programación de la tasa de aprendizaje

| Programación | Comportamiento |
|---|---|
| Constante | No cambia |
| Decaimiento exponencial | Multiplica la tasa por 0.85 cada época (de forma continua) |
| Coseno | Desciende suavemente hasta casi 0 al terminar el entrenamiento |
| Reducir en meseta | Si `val_loss` no mejora en 1 época, multiplica la tasa por 0.5 (mínimo 1e-6) |

### Aumento de datos

Rotación de hasta ±14°, traslación del 8 % y zoom del 10 %, **solo al entrenar**. Ayuda a reconocer dibujos hechos con el ratón, que no son idénticos a los de MNIST.

## Mejor configuración probada

Resultado obtenido con la configuración siguiente:

| Parámetro | Valor | Motivo |
|---|---|---|
| Arquitectura | CNN profunda | Mayor capacidad; separa dígitos parecidos |
| Optimizador | Adam | Converge rápido y estable |
| Tasa de aprendizaje | 0.001 | Valor estándar para Adam |
| Programación | Coseno | Pasos grandes al inicio y ajuste fino al final |
| Dropout | 0.30 | Evita memorizar sin frenar el aprendizaje |
| Regularización L2 | 0.0001 | Penalización ligera de pesos grandes |
| Épocas | 25 | Permite un descenso gradual de la tasa |
| Batch | 128 | Equilibrio entre estabilidad y velocidad |
| Validación | 10 % | 6 000 imágenes para vigilar el aprendizaje |
| Datos usados | 100 % | 54 000 para entrenar + 6 000 para validar |
| Early stopping | Sí, paciencia 10 | Tolera pequeñas fluctuaciones |
| Aumento de datos | Sí | Mejora el reconocimiento de dibujos a mano |
| Semilla | 44 | Reproducibilidad |

| Resultado | Valor |
|---|---|
| Precisión en prueba (10 000 imágenes) | **99.58 %** |
| Pérdida en prueba | 0.0422 |
| Parámetros | 469 098 |
| Tiempo de entrenamiento (CPU) | ≈ 956 s (≈ 16 min) |

Las curvas de entrenamiento y validación casi coinciden, por lo que **no hay sobreajuste**.

**Procedimiento sugerido:**

1. Probar con los valores rápidos por defecto (20 % de datos, 3 épocas) para verificar que todo funciona.
2. Usar un barrido de tasa de aprendizaje y de arquitectura con pocos datos.
3. Entrenar la configuración final con el 100 % de los datos.
4. Revisar **Evaluar** y probar dibujando en **Reconocer**.

## Cómo funciona por dentro

- **Datos:** `tf.keras.datasets.mnist` se carga en un hilo y se normaliza a [0, 1] con forma `(N, 28, 28, 1)`.
- **Modelo:** `construir_modelo(arq, dropout, l2, aumento)` crea un `tf.keras.Sequential`.
- **Compilación:** pérdida `sparse_categorical_crossentropy` y métrica `accuracy`.
- **Entrenamiento en hilo:** `Entrenador` (hereda de `threading.Thread`) se comunica con la interfaz **solo mediante una `queue.Queue`**. La interfaz consulta la cola periódicamente y reparte cada mensaje a su pestaña.

| Mensaje | Significado |
|---|---|
| `datos` / `datos_error` | MNIST cargado o fallo de descarga |
| `run_ini` | Empieza el entrenamiento *i* de *N* |
| `batch` | Avance dentro de la época (barra de progreso) |
| `epoca` | Fin de época: se envían las curvas para redibujarlas |
| `fin` | Entrenamiento terminado, con sus resultados |
| `error` | Se produjo una excepción |
| `hilo_fin` | El hilo terminó; se reactivan los botones |

- **Modelo activo:** un único modelo (`app.modelo`) es el que usan *Reconocer* y *Evaluar*.

## Archivos que genera la aplicación

| Ruta | Contenido |
|---|---|
| `modelos/<arquitectura>_AAAAMMDD_HHMMSS.keras` | Un modelo por cada entrenamiento (nunca se sobrescribe) |
| `modelo_mnist.keras` | Modelo activo; se carga al abrir la aplicación |
| `experimentos.json` | Historial con configuración, curvas y resultados |
| MNIST | Se descarga una vez en la caché de Keras |

## Atajos de teclado

| Atajo | Acción |
|---|---|
| `Ctrl + Z` | Deshacer el último trazo (pestaña Reconocer) |
| `Ctrl + L` | Limpiar el lienzo (pestaña Reconocer) |

## Solución de problemas

| Problema | Solución |
|---|---|
| `ModuleNotFoundError: tensorflow` | Activa el entorno virtual y ejecuta `pip install -r requirements.txt` |
| `No module named 'tkinter'` (Linux) | `sudo apt install python3-tk` |
| TensorFlow no se instala | Usa Python 3.10 – 3.12 (versiones muy nuevas pueden no ser compatibles) |
| El botón *Entrenar* está desactivado | MNIST aún se descarga; requiere internet la primera vez |
| Falla la descarga de MNIST | Revisa la conexión o el firewall y vuelve a abrir la aplicación |
| La ventana tarda en abrir | Es normal: TensorFlow tarda unos segundos en importarse |
| *Reconocer* muestra «Aún no hay un modelo» | Entrena uno en la pestaña 1 o cárgalo desde *Archivo → Cargar modelo…* |
| «Mapas de activación» no muestra nada | El modelo activo es un MLP; usa una arquitectura convolucional |
| Reconoce mal un dígito dibujado | Usa un trazo más grueso, dibuja el número grande y entrena con **aumento de datos** |
| El entrenamiento es muy lento | Reduce las épocas o el porcentaje de datos, o usa la CNN simple |

## Autor

**Rojas Luque Franco** · Código 230494 · Semestre VIII
Ingeniería de Sistemas · Facultad de Ingeniería Mecánica Eléctrica, Electrónica y Sistemas
Universidad Nacional del Altiplano
Docente: Ing. Romero Flores Robert Antonio
