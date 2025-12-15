import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageDraw, ImageTk, ImageOps
import numpy as np
import cv2
import time
import os

USE_REAL_MODELS = False
MNIST_MODEL_PATH = "mnist_model.tflite"
FACE_MODEL_PATH = "ferplus_model_pd_best.tflite"


class ModelHandler:

    def __init__(self, model_path):
        self.interpreter = None
        if USE_REAL_MODELS and os.path.exists(model_path):
            import tensorflow as tf
            try:
                self.interpreter = tf.lite.Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                print(f"Załadowano model: {model_path}")
            except Exception as e:
                print(f"Błąd ładowania modelu: {e}")

    def predict(self, input_data):
        if self.interpreter:
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            return self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        else:

            time.sleep(0.5)
            return np.random.rand(10)

class ExplainWindow(tk.Toplevel):
    def __init__(self, parent, image_pil, title="Wyjaśnienie AI"):
        super().__init__(parent)
        self.title(title)
        self.geometry("450x650")
        self.configure(bg="#F8F9FA")

        self.layers = []
        self.current_layer_index = 0
        self.last_toast_time = 0
        self.TOAST_COOLDOWN = 5.0 

        self.generate_layers(image_pil)

        self.create_widgets()
        self.update_ui()

    def create_widgets(self):

        header_frame = tk.Frame(self, bg="#F8F9FA")
        header_frame.pack(pady=10, fill='x')

        btn_back = tk.Button(header_frame, text="←", font=("Arial", 14), command=self.destroy, bg="#E0E0E0",
                             relief="flat")
        btn_back.pack(side="left", padx=10)

        lbl_title = tk.Label(header_frame, text="Wizualizacja Sieci", font=("Arial", 14, "bold"), bg="#F8F9FA",
                             fg="#333333")
        lbl_title.pack(side="left", expand=True)

        self.card_frame = tk.Frame(self, bg="white", bd=2, relief="raised")
        self.card_frame.pack(pady=10, padx=20, fill='both', expand=True)

        self.lbl_layer_name = tk.Label(self.card_frame, text="", font=("Arial", 16, "bold"), bg="white", fg="#2C3E50")
        self.lbl_layer_name.pack(pady=(20, 10))

        self.canvas_visual = tk.Canvas(self.card_frame, width=260, height=260, bg="#EEEEEE", highlightthickness=0)
        self.canvas_visual.pack(pady=10)

        self.lbl_layer_desc = tk.Label(self.card_frame, text="", font=("Arial", 11), bg="white", fg="#555555",
                                       wraplength=350, justify="center")
        self.lbl_layer_desc.pack(pady=20, padx=10)

        nav_frame = tk.Frame(self, bg="#F8F9FA")
        nav_frame.pack(side="bottom", pady=20, fill='x')

        self.btn_prev = tk.Button(nav_frame, text="<", font=("Arial", 18), width=4, command=self.prev_layer)
        self.btn_prev.pack(side="left", padx=30)

        self.lbl_indicator = tk.Label(nav_frame, text="1 / 4", font=("Arial", 12, "bold"), bg="#F8F9FA")
        self.lbl_indicator.pack(side="left", expand=True)

        self.btn_next = tk.Button(nav_frame, text=">", font=("Arial", 18), width=4, bg="#3498DB", fg="white",
                                  command=self.next_layer)
        self.btn_next.pack(side="right", padx=30)

        self.lbl_toast = tk.Label(self, text="", bg="#333333", fg="white", font=("Arial", 10))

    def show_toast(self, message):

        current_time = time.time()
        if current_time - self.last_toast_time < self.TOAST_COOLDOWN:
            return

        self.lbl_toast.config(text=message)
        self.lbl_toast.place(relx=0.5, rely=0.9, anchor="center")
        self.last_toast_time = current_time

        self.after(2000, lambda: self.lbl_toast.place_forget())

    def update_ui(self):
        data = self.layers[self.current_layer_index]
        self.lbl_layer_name.config(text=data['name'])
        self.lbl_layer_desc.config(text=data['desc'])
        self.lbl_indicator.config(text=f"{self.current_layer_index + 1} / {len(self.layers)}")

        img = data['image']
        img = img.resize((260, 260), Image.NEAREST)
        self.photo_img = ImageTk.PhotoImage(img)
        self.canvas_visual.create_image(130, 130, image=self.photo_img)

    def prev_layer(self):
        if self.current_layer_index > 0:
            self.current_layer_index -= 1
            self.update_ui()
        else:
            self.show_toast("To jest Warstwa wejściowa!")

    def next_layer(self):
        if self.current_layer_index < len(self.layers) - 1:
            self.current_layer_index += 1
            self.update_ui()
        else:
            self.show_toast("To jest Warstwa wyjściowa!")

    def generate_layers(self, input_pil):

        input_gray = input_pil.convert('L')
        img_arr = np.array(input_gray)

        self.layers.append({
            "name": "Warstwa wejściowa",
            "image": input_gray,
            "desc": "To surowe dane, które widzi model. Każdy piksel ma wartość liczbową odpowiadającą jasności."
        })

        vertical_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        horizontal_filter = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        vert_map = cv2.filter2D(img_arr, -1, vertical_filter)
        vert_pil = Image.fromarray(vert_map.astype('uint8'))

        self.layers.append({
            "name": "Warstwa 1",
            "image": vert_pil,
            "desc": "Wykrywanie cech pionowych. Neurony w tej warstwie aktywują się tam, gdzie widzą pionowe krawędzie."
        })

        horiz_map = cv2.filter2D(img_arr, -1, horizontal_filter)
        horiz_pil = Image.fromarray(horiz_map.astype('uint8'))

        self.layers.append({
            "name": "Warstwa 2",
            "image": horiz_pil,
            "desc": "Wykrywanie cech poziomych. Neurony reagują na górne i dolne krawędzie obiektu."
        })

        combined_map = np.maximum(vert_map, horiz_map)
        combined_pil = Image.fromarray(combined_map.astype('uint8'))

        self.layers.append({
            "name": "Warstwa wyjściowa",
            "image": combined_pil,
            "desc": "Model łączy wykryte proste cechy (linie) w bardziej złożone kształty, co pozwala mu podjąć decyzję."
        })

class MnistFrame(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        self.model = ModelHandler(MNIST_MODEL_PATH)

        tk.Label(self, text="Rysuj cyfrę (0-9)", font=("Arial", 16)).pack(pady=10)

        self.canvas_width = 280
        self.canvas_height = 280
        self.canvas = tk.Canvas(self, width=self.canvas_width, height=self.canvas_height, bg="white", cursor="cross")
        self.canvas.pack(pady=10)

        self.image1 = Image.new("RGB", (self.canvas_width, self.canvas_height), "white")
        self.draw = ImageDraw.Draw(self.image1)

        self.canvas.bind("<B1-Motion>", self.paint)

        self.lbl_result = tk.Label(self, text="Wynik: ?", font=("Arial", 14))
        self.lbl_result.pack(pady=5)

        btn_frame = tk.Frame(self)
        btn_frame.pack(pady=10)

        tk.Button(btn_frame, text="Wyczyść", command=self.clear_canvas).grid(row=0, column=0, padx=5)
        tk.Button(btn_frame, text="Klasyfikuj", command=self.classify, bg="#DDDDDD").grid(row=0, column=1, padx=5)
        tk.Button(btn_frame, text="Wyjaśnij", command=self.explain, bg="#3498DB", fg="white").grid(row=0, column=2,
                                                                                                   padx=5)

        tk.Button(self, text="Wróć do Menu", command=lambda: controller.show_frame("MenuFrame")).pack(pady=20)

    def paint(self, event):
        x1, y1 = (event.x - 10), (event.y - 10)
        x2, y2 = (event.x + 10), (event.y + 10)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", width=0)
        self.draw.ellipse([x1, y1, x2, y2], fill="black")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image1 = Image.new("RGB", (self.canvas_width, self.canvas_height), "white")
        self.draw = ImageDraw.Draw(self.image1)
        self.lbl_result.config(text="Wynik: ?")

    def get_processed_image(self):
        img = self.image1.resize((28, 28), Image.BILINEAR).convert('L')
        img = ImageOps.invert(img)
        return img

    def classify(self):
        img = self.get_processed_image()

        input_data = np.array(img, dtype=np.float32) / 255.0
        input_data = np.expand_dims(input_data, axis=0)
        input_data = np.expand_dims(input_data, axis=-1)

        try:
            output = self.model.predict(input_data.reshape(1, 28, 28, 1))
            prediction = np.argmax(output)
            confidence = np.max(output) * 100
            self.lbl_result.config(text=f"Cyfra: {prediction} ({confidence:.1f}%)")
        except Exception as e:
            mock_res = np.random.randint(0, 9)
            self.lbl_result.config(text=f"Cyfra (Mock): {mock_res}")

    def explain(self):
        img = self.get_processed_image()
        ExplainWindow(self, img, title="Explainer: MNIST")

class FaceFrame(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.model = ModelHandler(FACE_MODEL_PATH)
        self.emotions = ["Neutralny", "Radość", "Zaskoczenie", "Smutek", "Złość", "Obrzydzenie", "Strach", "Pogarda"]
        self.current_image = None

        tk.Label(self, text="Rozpoznawanie Emocji", font=("Arial", 16)).pack(pady=10)

        self.lbl_image = tk.Label(self, text="Brak obrazu", bg="#EEEEEE", width=40, height=15)
        self.lbl_image.pack(pady=10)

        self.lbl_result = tk.Label(self, text="Emocja: ?", font=("Arial", 14, "bold"))
        self.lbl_result.pack(pady=5)

        btn_frame = tk.Frame(self)
        btn_frame.pack(pady=10)

        tk.Button(btn_frame, text="Wczytaj Zdjęcie", command=self.load_image).grid(row=0, column=0, padx=5)
        tk.Button(btn_frame, text="Klasyfikuj", command=self.classify).grid(row=0, column=1, padx=5)
        tk.Button(btn_frame, text="Wyjaśnij", command=self.explain, bg="#3498DB", fg="white").grid(row=0, column=2,
                                                                                                   padx=5)

        tk.Button(self, text="Wróć do Menu", command=lambda: controller.show_frame("MenuFrame")).pack(pady=20)

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if path:
            img = Image.open(path)

            display_img = img.copy()
            display_img.thumbnail((300, 300))
            self.photo = ImageTk.PhotoImage(display_img)
            self.lbl_image.config(image=self.photo, text="")

            self.current_image = img.resize((48, 48), Image.BILINEAR).convert('L')
            self.lbl_result.config(text="Emocja: Gotowy")

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def classify(self):
        if self.current_image is None:
            messagebox.showwarning("Błąd", "Najpierw wczytaj zdjęcie!")
            return

        input_data = np.array(self.current_image, dtype=np.float32)
        input_data = np.expand_dims(input_data, axis=0)
        input_data = np.expand_dims(input_data, axis=-1)

        try:
            raw_output = self.model.predict(input_data)
            probs = self.softmax(raw_output)
            best_idx = np.argmax(probs)

            emotion = self.emotions[best_idx] if best_idx < len(self.emotions) else "Nieznany"
            prob = probs[best_idx] * 100

            color = "green" if emotion == "Radość" else "red" if emotion == "Złość" else "black"
            self.lbl_result.config(text=f"{emotion} ({prob:.1f}%)", fg=color)

        except Exception:
            mock_idx = np.random.randint(0, len(self.emotions))
            self.lbl_result.config(text=f"MOCK: {self.emotions[mock_idx]}", fg="blue")

    def explain(self):
        if self.current_image:
            ExplainWindow(self, self.current_image, title="Explainer: TWARZ")
        else:
            messagebox.showwarning("Błąd", "Brak obrazu do wyjaśnienia")

class AIExplainerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI Explainer (Python Edition)")
        self.geometry("400x500")

        self.container = tk.Frame(self)
        self.container.pack(side="top", fill="both", expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for F in (MenuFrame, MnistFrame, FaceFrame):
            page_name = F.__name__
            frame = F(parent=self.container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("MenuFrame")

    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()


class MenuFrame(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        tk.Label(self, text="AI Explainer", font=("Arial", 24, "bold")).pack(pady=50)

        tk.Button(self, text="MNIST (Cyfry)", font=("Arial", 14), width=20, height=2,
                  command=lambda: controller.show_frame("MnistFrame")).pack(pady=10)

        tk.Button(self, text="Rozpoznawanie Emocji", font=("Arial", 14), width=20, height=2,
                  command=lambda: controller.show_frame("FaceFrame")).pack(pady=10)

        tk.Button(self, text="Wyjście", command=controller.quit).pack(pady=50)


if __name__ == "__main__":
    app = AIExplainerApp()
    app.mainloop()