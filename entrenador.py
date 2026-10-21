"""Entrenamiento en un hilo aparte; avisa a la interfaz mediante una cola."""
import copy
import threading
import time
import traceback
from datetime import datetime

import tensorflow as tf

from config import DIR_MODELOS
from nucleo import construir_modelo, crear_optimizador, lr_actual


class Entrenador(threading.Thread):
    """Entrena una o varias configuraciones y avisa a la interfaz por una cola."""

    def __init__(self, configs, datos, cola, parar):
        super().__init__(daemon=True)
        self.configs, self.datos, self.cola, self.parar = configs, datos, cola, parar

    def run(self):
        try:
            for i, cfg in enumerate(self.configs):
                if self.parar.is_set():
                    break
                self.cola.put(("run_ini", (i, len(self.configs), cfg)))
                self.cola.put(("fin", self.entrenar(cfg)))
        except Exception as ex:
            traceback.print_exc()
            self.cola.put(("error", f"{type(ex).__name__}: {ex}"))
        self.cola.put(("hilo_fin", None))

    def entrenar(self, cfg):
        (x_tr, y_tr), (x_te, y_te) = self.datos
        tf.keras.utils.set_random_seed(cfg["semilla"])
        n = max(100, int(len(x_tr) * cfg["subset"] / 100))
        x, y = x_tr[:n], y_tr[:n]
        n_val = int(n * cfg["val"] / 100)
        pasos = max(1, (n - n_val) // cfg["batch"])

        modelo = construir_modelo(cfg["arq"], cfg["dropout"], cfg["l2"], cfg["aug"])
        modelo.compile(optimizer=crear_optimizador(cfg, pasos),
                       loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        cola, parar, t0 = self.cola, self.parar, time.time()
        hist = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": [], "lr": []}

        class Progreso(tf.keras.callbacks.Callback):
            epoca = 0

            def on_epoch_begin(cb, epoch, logs=None):
                cb.epoca = epoch

            def on_train_batch_end(cb, batch, logs=None):
                if parar.is_set():
                    cb.model.stop_training = True
                if batch % max(1, pasos // 30) == 0:
                    cola.put(("batch", (cb.epoca, min(1.0, (batch + 1) / pasos))))

            def on_epoch_end(cb, epoch, logs=None):
                logs = logs or {}
                for k in ("loss", "accuracy", "val_loss", "val_accuracy"):
                    hist[k].append(float(logs.get(k, float("nan"))))
                hist["lr"].append(lr_actual(cb.model))
                cola.put(("epoca", {"epoca": epoch + 1, "total": cfg["epocas"],
                                    "hist": copy.deepcopy(hist)}))

        callbacks = [Progreso()]
        if cfg["es"]:
            callbacks.append(tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=cfg["paciencia"], restore_best_weights=True))
        if cfg["sched"] == "Reducir en meseta":
            callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=1, min_lr=1e-6))

        modelo.fit(x, y, epochs=cfg["epocas"], batch_size=cfg["batch"],
                   validation_split=cfg["val"] / 100, callbacks=callbacks, verbose=0)

        perdida, precision = modelo.evaluate(x_te, y_te, batch_size=512, verbose=0)
        DIR_MODELOS.mkdir(exist_ok=True)
        ruta = DIR_MODELOS / f"{modelo.name}_{datetime.now():%Y%m%d_%H%M%S}.keras"
        modelo.save(ruta)
        return {"cfg": cfg, "hist": hist, "modelo": modelo, "ruta": ruta,
                "test_acc": float(precision), "test_loss": float(perdida),
                "params": int(modelo.count_params()), "tiempo": time.time() - t0,
                "etiqueta": ruta.name}
