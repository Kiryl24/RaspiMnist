import tkinter as tk
from tkinter import font
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageTk
import os


class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Rozpoznawanie Cyfr")

        window_width = 700
        window_height = 1000
        self.root.geometry(f"{window_width}x{window_height}")
        self.root.configure(bg='#f0f0f0')

        model_path = 'mnist_model.keras'
        if not os.path.exists(model_path):
            print(f"No model in {model_path}")
            self.root.destroy()
            return

        self.model = tf.keras.models.load_model(model_path)

        self.strokes = []
        self.current_stroke = []
        self.is_classified = False
        self.tk_image_ref = None
        self.result_font = font.Font(family='Helvetica', size=24, weight='bold')
        self.result_label = tk.Label(root, text="Narysuj cyfrę od 0 do 9)",
                                     bg='#f0f0f0', fg='#333333',
                                     font=self.result_font, pady=10)
        self.result_label.pack(side=tk.TOP, fill=tk.X)

        self.canvas_size = 450

        self.canvas = tk.Canvas(root, width=self.canvas_size, height=self.canvas_size,
                                bg='white', cursor="cross",
                                highlightthickness=4, highlightbackground='black', highlightcolor='black')

        self.canvas.pack(pady=10)

        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.end_draw)

        self.preview_frame = tk.Frame(root, bg='#f0f0f0')
        self.preview_frame.pack(pady=5)

        self.preview_label_text = tk.Label(self.preview_frame, text="Co widzi sieć:",
                                           bg='#f0f0f0', font=('Helvetica', 12))


        self.preview_image_label = tk.Label(self.preview_frame, bg='#dddddd', width=20, height=10)

        self.btn_frame = tk.Frame(root, bg='#f0f0f0')
        self.btn_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=30, padx=20)

        self.btn_font = font.Font(family='Helvetica', size=16)
        self.undo_btn = tk.Button(self.btn_frame, text="Cofnij", command=self.undo,
                                  bg='#e74c3c', fg='white', font=self.btn_font,
                                  height=2, width=10)
        self.undo_btn.pack(side=tk.LEFT, padx=10)

        self.classify_btn = tk.Button(self.btn_frame, text="Klasyfikuj", command=self.handle_classify_click,
                                      bg='#2ecc71', fg='white', font=self.btn_font,
                                      height=2, width=15)
        self.classify_btn.pack(side=tk.RIGHT, padx=10)

    def start_draw(self, event):
        if self.is_classified:
            self.reset_canvas()
        self.current_stroke = [(event.x, event.y)]

    def draw(self, event):
        if not self.current_stroke:
            return
        x, y = event.x, event.y
        last_x, last_y = self.current_stroke[-1]

        self.canvas.create_line((last_x, last_y, x, y),
                                width=25, fill='black',
                                capstyle=tk.ROUND, smooth=True)
        self.current_stroke.append((x, y))

    def end_draw(self, event):
        if len(self.current_stroke) > 1:
            self.strokes.append(self.current_stroke)
        self.current_stroke = []

    def undo(self):
        if self.is_classified:
            self.reset_canvas()
            return
        if self.strokes:
            self.strokes.pop()
            self.redraw_canvas()

    def redraw_canvas(self):
        self.canvas.delete("all")
        for stroke in self.strokes:
            if len(stroke) > 1:
                coords = [coord for point in stroke for coord in point]
                self.canvas.create_line(coords, width=25, fill='black',
                                        capstyle=tk.ROUND, joinstyle=tk.ROUND)

    def handle_classify_click(self):
        if not self.is_classified:
            self.classify()
        else:
            self.reset_canvas()

    def classify(self):
        if not self.strokes:
            self.result_label.config(text="Najpierw coś narysuj!")
            return

        image = Image.new("L", (self.canvas_size, self.canvas_size), color=0)
        draw = ImageDraw.Draw(image)

        for stroke in self.strokes:
            if len(stroke) > 1:
                # Używamy width=25, ale przy skalowaniu w dół linie staną się cieńsze
                draw.line(stroke, fill=255, width=25, joint='curve')

        img_resized = image.resize((28, 28), resample=Image.Resampling.LANCZOS)

        preview_img = img_resized.resize((140, 140), resample=Image.Resampling.NEAREST)
        self.tk_image_ref = ImageTk.PhotoImage(preview_img)

        # Wyświetlenie podglądu
        self.preview_label_text.pack()
        self.preview_image_label.config(image=self.tk_image_ref, width=140, height=140)
        self.preview_image_label.pack()

        img_array = np.array(img_resized)
        img_array = img_array / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        prediction = self.model.predict(img_array)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        self.result_label.config(text=f"Cyfra: {predicted_class} | Pewność: {confidence:.1f}%")
        self.classify_btn.config(text="Nowa klasyfikacja", bg='#3498db')
        self.is_classified = True

    def reset_canvas(self):
        self.strokes = []
        self.canvas.delete("all")
        self.result_label.config(text="Narysuj cyfrę (0-9)")
        self.classify_btn.config(text="Klasyfikuj", bg='#2ecc71')

        # Ukrywanie podglądu
        self.preview_label_text.pack_forget()
        self.preview_image_label.pack_forget()
        self.preview_image_label.config(image='')

        self.is_classified = False


if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()