
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tkinter as tk


def main():
    raiz = tk.Tk()
    raiz.withdraw()

    splash = tk.Toplevel(raiz)
    splash.overrideredirect(True)
    splash.configure(bg="#131316", highlightthickness=1, highlightbackground="#c9a45c")
    ancho, alto = 440, 150
    splash.geometry(f"{ancho}x{alto}+{(splash.winfo_screenwidth() - ancho) // 2}"
                    f"+{(splash.winfo_screenheight() - alto) // 2}")
    tk.Label(splash, text="Reconocedor de dígitos", bg="#131316", fg="#ececf1",
             font=("Segoe UI", 20, "bold")).pack(pady=(32, 4))
    tk.Label(splash, text="Cargando TensorFlow…", bg="#131316", fg="#8e8e9a",
             font=("Segoe UI", 11)).pack()
    splash.update()

    from app import App         

    splash.destroy()
    raiz.deiconify()
    App(raiz)
    raiz.mainloop()


if __name__ == "__main__":
    main()
