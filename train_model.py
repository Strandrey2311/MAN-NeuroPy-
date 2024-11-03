import os
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

dataset_path = 'C:/Users/clash/PycharmProjects/NeuroPy/dataset/archive/COVID-19_Radiography_Dataset'
if not os.path.exists(dataset_path):
    print(f"Ошибка: Директория с данными не найдена: {dataset_path}")
    exit()

mask_paths = []
labels = []
classes = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']

for label in classes:
    mask_dir = os.path.join(dataset_path, label, 'masks')
    if os.path.exists(mask_dir):
        images = [mask for mask in os.listdir(mask_dir) if mask.endswith(('.jpg', '.png'))]
        for mask in images:
            mask_paths.append(os.path.join(mask_dir, mask))
            labels.append(classes.index(label))


if not mask_paths:
    print("Ошибка: Не удалось найти маски в папках.")
    exit()

def load_and_preprocess_mask(path):
    mask = tf.io.read_file(path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, [224, 224]) / 255.0  # Нормализация
    return mask

mask_dataset = tf.data.Dataset.from_tensor_slices(mask_paths).map(load_and_preprocess_mask)
label_dataset = tf.data.Dataset.from_tensor_slices(tf.cast(labels, tf.int64))

train_masks, val_masks, train_labels, val_labels = train_test_split(mask_paths, labels, test_size=0.2, stratify=labels)

train_ds = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(train_masks).map(load_and_preprocess_mask),
                                tf.data.Dataset.from_tensor_slices(tf.cast(train_labels, tf.int64))))
val_ds = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(val_masks).map(load_and_preprocess_mask),
                              tf.data.Dataset.from_tensor_slices(tf.cast(val_labels, tf.int64))))

train_ds = train_ds.shuffle(buffer_size=len(train_masks)).batch(32)
val_ds = val_ds.batch(32)

model = models.Sequential([
    layers.Input(shape=(224, 224, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),  # Dropout для предотвращения переобучения
    layers.Dense(len(classes), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_ds, epochs=30, validation_data=val_ds)

model.save('C:/Users/clash/PycharmProjects/NeuroPy/models/diagnosis_model(3).keras', save_format='keras')
print("Модель успешно сохранена в формате Keras!")