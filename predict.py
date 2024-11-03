import os
import tensorflow as tf
from tkinter import Tk, Label, Button, filedialog, messagebox, simpledialog, Listbox, Scrollbar, END, Frame, Text
from PIL import Image, ImageTk


# Завантаження моделі
def load_trained_model(model_path):
    if not os.path.exists (model_path):
        messagebox.showerror ("Помилка", f"Директорію моделі не знайдено: {model_path}")
        exit ()
    return tf.keras.models.load_model (model_path)


# Функція для вибору моделі
def select_model():
    global model, model_path
    model_name = model_listbox.get (model_listbox.curselection ())
    model_path = os.path.join (models_dir, model_name)
    model = load_trained_model (model_path)
    messagebox.showinfo ("Модель завантажена", f"Ви вибрали модель: {model_name}")


# Передбачення зображення
def predict_image(model, image_path):
    img = tf.io.read_file (image_path)
    img = tf.image.decode_jpeg (img, channels=1)  # Встановлюємо 1 канал для чорно-білого зображення
    img = tf.image.resize (img, [224, 224]) / 255.0  # Нормалізація
    img = tf.expand_dims (img, axis=0)  # Додаємо розмір batch

    predictions = model.predict (img)
    predicted_class_index = tf.argmax (predictions[0]).numpy ()
    class_names = ['COVID', 'Легеневе затемнення', 'Пацієнт не хворий', 'Вірусна пневмонія']

    # Виведення всіх ймовірностей
    output_text.delete (1.0, END)  # Очищуємо поле виводу
    for i, class_name in enumerate (class_names):
        output_text.insert (END, f"{class_name}: {predictions[0][i]:.4f}\n")

    return class_names[predicted_class_index], predictions[0]


# Логування невірних передбачень
def log_incorrect_prediction(image_path, incorrect_label, correct_label):
    with open ("incorrect_predictions_log.txt", "a") as log_file:
        log_file.write (f"{image_path},{incorrect_label},{correct_label}\n")


# Класифікація зображення
def classify_image():
    if 'img_path' not in globals ():
        messagebox.showwarning ("Помилка", "Будь ласка, завантажте зображення!")
        return

    result, probabilities = predict_image (model, img_path)
    feedback = messagebox.askquestion ("Підтвердження", f"Передбачення: {result}. Це правильно?", icon='question',
                                       type='yesno')

    if feedback == 'no':
        correct_label = simpledialog.askstring ("Введіть правильний клас",
                                                "Який клас є правильним? (COVID, Легеневе затемнення, Норма, Вірусна пневмонія)")
        output_text.insert (END, f"\nКористувач вказав правильний клас: {correct_label}")
        log_incorrect_prediction ( result, correct_label)


# Завантаження зображення
def upload_image():
    global img_path
    img_path = filedialog.askopenfilename (title="Виберіть зображення", filetypes=[("Image Files", "*.jpg;*.png")])
    if img_path:
        img = Image.open (img_path).convert ("L")  # Конвертуємо в чорно-білий формат, якщо потрібно
        img.thumbnail ((250, 250))
        img_display = ImageTk.PhotoImage (img)
        img_label.configure (image=img_display)
        img_label.image = img_display

        # Зберігаємо тимчасове зображення
        img.save ("temp_image.jpg")  # Використовуємо новий шлях для передбачення
        img_path = "temp_image.jpg"


# Створення основного вікна GUI
root = Tk ()
root.title ("Класифікація медичних зображень")
root.geometry ("450x700")
root.configure (bg="#F0F8FF")  # Фоновий колір

# Путь до директорії з моделями
models_dir = 'C:/Users/clash/PycharmProjects/NeuroPy/models'

# Створення списку для вибору моделі
model_label = Label (root, text="Виберіть модель:", font=("Arial", 12), bg="#F0F8FF")
model_label.pack (pady=10)

model_listbox = Listbox (root, selectmode="single", height=5, font=("Arial", 10))
model_listbox.pack (pady=5)

# Додаємо моделі з директорії в список
for model_name in os.listdir (models_dir):
    if model_name.endswith ('.keras'):
        model_listbox.insert (END, model_name)

# Кнопка для вибору моделі
select_model_btn = Button (root, text="Завантажити вибрану модель", command=select_model, font=("Arial", 12),
                           bg="#4CAF50", fg="white")
select_model_btn.pack (pady=15)

# Кнопка для завантаження зображення
upload_btn = Button(root, text="Завантажити зображення", command=upload_image, font=("Arial", 12), bg="#2196F3", fg="white")
upload_btn.pack(pady=15)

# Елемент для відображення зображення
img_label = Label(root, bg="#F0F8FF")
img_label.pack(pady=10)

# Кнопка для класифікації зображення
classify_btn = Button(root, text="Класифікувати зображення", command=classify_image, font=("Arial", 12), bg="#FF9800", fg="white")
classify_btn.pack(pady=15)

# Поле для виводу передбачень і ймовірностей
output_label = Label(root, text="Результати передбачення:", font=("Time New Roman", 12), bg="#F0F8FF")
output_label.pack(pady=5)

output_frame = Frame(root, bg="#F0F8FF")
output_frame.pack(pady=5)

output_text = Text(output_frame, height=8, width=50, wrap='word', font=("Arial", 10))
output_text.pack(side="left", fill="y")

scrollbar = Scrollbar(output_frame, command=output_text.yview)
scrollbar.pack(side="right", fill="y")

output_text.config(yscrollcommand=scrollbar.set)

# Запуск головного циклу
root.mainloop()