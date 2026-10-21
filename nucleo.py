"""Núcleo de la red neuronal: modelos, optimizadores, preprocesado y métricas."""
import numpy as np
import tensorflow as tf
from PIL import Image


def construir_modelo(arq, dropout=0.3, l2=0.0, aumento=False):
    L = tf.keras.layers
    reg = tf.keras.regularizers.l2(l2) if l2 > 0 else None

    def conv(f, k=3, **kw):
        return L.Conv2D(f, k, kernel_regularizer=reg, **kw)

    def den(n, **kw):
        return L.Dense(n, kernel_regularizer=reg, **kw)

    capas = [L.Input(shape=(28, 28, 1))]
    if aumento:
        capas += [L.RandomRotation(0.04), L.RandomTranslation(0.08, 0.08), L.RandomZoom(0.1)]

    if arq == "MLP":
        capas += [L.Flatten(), den(256, activation="relu"), L.Dropout(dropout),
                  den(128, activation="relu"), L.Dropout(dropout)]
    elif arq == "LeNet-5":
        capas += [conv(6, 5, padding="same", activation="tanh"), L.AveragePooling2D(2),
                  conv(16, 5, activation="tanh"), L.AveragePooling2D(2), L.Flatten(),
                  den(120, activation="tanh"), den(84, activation="tanh"),
                  L.Dropout(dropout)]
    elif arq == "CNN simple":
        capas += [conv(32, activation="relu"), L.MaxPooling2D(2),
                  conv(64, activation="relu"), L.MaxPooling2D(2),
                  L.Flatten(), L.Dropout(dropout)]
    else:  # CNN profunda
        capas += [conv(32, padding="same", activation="relu"), L.BatchNormalization(),
                  conv(32, padding="same", activation="relu"), L.BatchNormalization(),
                  L.MaxPooling2D(2), L.Dropout(dropout * 0.8),
                  conv(64, padding="same", activation="relu"), L.BatchNormalization(),
                  conv(64, padding="same", activation="relu"), L.BatchNormalization(),
                  L.MaxPooling2D(2), L.Dropout(dropout * 0.8),
                  L.Flatten(), den(128, activation="relu"), L.BatchNormalization(),
                  L.Dropout(dropout)]
    capas.append(den(10, activation="softmax"))
    nombre = arq.lower().replace(" ", "_").replace("-", "_")
    return tf.keras.Sequential(capas, name=nombre)


def crear_optimizador(cfg, pasos_por_epoca):
    lr = cfg["lr"]
    S, O = tf.keras.optimizers.schedules, tf.keras.optimizers
    if cfg["sched"] == "Decaimiento exponencial":
        lr = S.ExponentialDecay(lr, pasos_por_epoca, 0.85)
    elif cfg["sched"] == "Coseno":
        lr = S.CosineDecay(lr, max(1, pasos_por_epoca * cfg["epocas"]))
    fabrica = {
        "Adam": lambda: O.Adam(lr), "SGD": lambda: O.SGD(lr),
        "SGD + Momentum": lambda: O.SGD(lr, momentum=0.9),
        "RMSprop": lambda: O.RMSprop(lr), "Nadam": lambda: O.Nadam(lr),
        "Adagrad": lambda: O.Adagrad(lr),
    }
    return fabrica[cfg["opt"]]()


def lr_actual(modelo):
    """Tasa de aprendizaje vigente (también cuando hay una programación)."""
    try:
        lr = modelo.optimizer.learning_rate
        if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
            lr = lr(modelo.optimizer.iterations)
        return float(np.array(lr))
    except Exception:
        return float("nan")


def preparar_imagen(img):
    """Imagen cualquiera -> array 28x28 estilo MNIST (blanco sobre negro, centrado)."""
    a = np.array(img.convert("L"), dtype="float32")
    if a.mean() > 127:                       # fondo claro -> invertir
        a = 255 - a
    a[a < 30] = 0
    ys, xs = np.where(a > 0)
    if xs.size == 0:
        return np.zeros((28, 28), dtype="float32")
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    alto, ancho = a.shape
    if max(alto, ancho) < 15:                # trazo demasiado pequeño
        return np.zeros((28, 28), dtype="float32")

    k = 20.0 / max(alto, ancho)              # el dígito ocupa 20x20 como en MNIST
    peq = Image.fromarray(a.astype("uint8")).resize(
        (max(1, round(ancho * k)), max(1, round(alto * k))), Image.LANCZOS)
    lienzo = Image.new("L", (28, 28), 0)
    lienzo.paste(peq, ((28 - peq.width) // 2, (28 - peq.height) // 2))

    m = np.array(lienzo, dtype="float32")    # centrar por centro de masa
    total = m.sum()
    if total > 0:
        cy = (m.sum(axis=1) * np.arange(28)).sum() / total
        cx = (m.sum(axis=0) * np.arange(28)).sum() / total
        m = np.roll(m, (round(14 - cy), round(14 - cx)), axis=(0, 1))
    return m / 255.0


def matriz_confusion(y, p):
    cm = np.zeros((10, 10), dtype=int)
    np.add.at(cm, (y, p), 1)
    return cm


def metricas_por_clase(cm):
    diag = np.diag(cm).astype(float)
    precision = np.divide(diag, cm.sum(axis=0), out=np.zeros(10), where=cm.sum(axis=0) > 0)
    recall = np.divide(diag, cm.sum(axis=1), out=np.zeros(10), where=cm.sum(axis=1) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(10), where=(precision + recall) > 0)
    return precision, recall, f1, cm.sum(axis=1)
