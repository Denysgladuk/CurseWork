import cv2
import os
import numpy as np
from tkinter import *
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import pickle
from datetime import datetime
import shutil
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Deep Learning
import tensorflow as tf
import keras
from keras.applications import MobileNetV2, ResNet50
from keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from keras.applications.resnet50 import preprocess_input as resnet_preprocess
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization, Input
from keras.layers import RandomFlip, RandomRotation, RandomZoom, RandomBrightness, RandomContrast
from keras.preprocessing.image import img_to_array
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import kagglehub

class FaceRecognitionSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("Інтелектуальна система ідентифікації людей (Dual Model)")
        self.root.geometry("1400x800")
        self.root.configure(bg='#2c3e50')
        
        # Ініціалізація змінних
        self.dataset_path = "dataset"
        self.models_path = "models"
        self.training_data_path = "training_data"
        self.kaggle_dataset_path = None
        
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Дві моделі статі
        self.gender_model_mobilenet = None
        self.gender_model_resnet = None
        
        # Дві моделі розпізнавання
        self.face_encoder_mobilenet = None
        self.face_encoder_resnet = None
        self.face_classifier_mobilenet = None
        self.face_classifier_resnet = None
        self.label_encoder = None
        
        self.person_counter = 0
        
        # Історія навчання для порівняння
        self.training_history = {
            'mobilenet': None,
            'resnet': None
        }
        self.min_recognition_confidence = 0.85
        self.max_gender_samples = 2000
        self.max_celeba_samples_per_gender = 5000
        np.random.seed(42)
        tf.random.set_seed(42)
        
        # Створення необхідних директорій
        os.makedirs(self.dataset_path, exist_ok=True)
        os.makedirs(self.models_path, exist_ok=True)
        os.makedirs(self.training_data_path, exist_ok=True)
        
        # Створення інтерфейсу
        self.create_ui()
        
        # Завантаження моделей
        self.load_models()
        
    def create_ui(self):
        """Створення графічного інтерфейсу"""
        # Заголовок
        title_frame = Frame(self.root, bg='#34495e', bd=5)
        title_frame.pack(side=TOP, fill=X)
        
        title_label = Label(title_frame, text="🎯 Інтелектуальна система ідентифікації (MobileNetV2 vs ResNet50)", 
                            font=("Arial", 22, "bold"), bg='#34495e', fg='white')
        title_label.pack(pady=15)
        
        # Головний контейнер
        main_frame = Frame(self.root, bg='#2c3e50')
        main_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)
        
        # Ліва панель - кнопки управління
        left_frame = Frame(main_frame, bg='#34495e', width=300, bd=2, relief=RIDGE)
        left_frame.pack(side=LEFT, fill=Y, padx=(0, 10))
        left_frame.pack_propagate(False)
        
        Label(left_frame, text="Панель управління", font=("Arial", 16, "bold"), 
              bg='#34495e', fg='white').pack(pady=20)
        
        # Кнопки
        btn_style = {'font': ('Arial', 11), 'bg': '#3498db', 'fg': 'white', 
                     'activebackground': '#2980b9', 'bd': 0, 'padx': 15, 'pady': 8}
        
        Button(left_frame, text="📥 Завантажити датасет", command=self.download_training_dataset, 
               bg='#e74c3c', fg='white', activebackground='#c0392b',
               font=('Arial', 11, 'bold'), bd=0, padx=15, pady=8).pack(pady=8, padx=15, fill=X)
        
        Button(left_frame, text="🎓 Навчити MobileNetV2", command=lambda: self.train_gender_model('mobilenet'),
               bg='#9b59b6', fg='white', activebackground='#8e44ad',
               font=('Arial', 11, 'bold'), bd=0, padx=15, pady=8).pack(pady=8, padx=15, fill=X)
        
        Button(left_frame, text="🎓 Навчити ResNet50", command=lambda: self.train_gender_model('resnet'),
               bg='#e67e22', fg='white', activebackground='#d35400',
               font=('Arial', 11, 'bold'), bd=0, padx=15, pady=8).pack(pady=8, padx=15, fill=X)
        
        Button(left_frame, text="📊 Порівняти моделі", command=self.compare_models,
               bg='#16a085', fg='white', activebackground='#138d75',
               font=('Arial', 11, 'bold'), bd=0, padx=15, pady=8).pack(pady=8, padx=15, fill=X)
        
        Button(left_frame, text="🎓 Навчити розпізнавання", command=self.train_face_recognition_menu,
               **btn_style).pack(pady=8, padx=15, fill=X)
        
        Button(left_frame, text="📸 Розпізнати з фото", command=self.analyze_photo, 
               **btn_style).pack(pady=8, padx=15, fill=X)
        
        Button(left_frame, text="🎥 Розпізнати з камери", command=self.analyze_camera, 
               **btn_style).pack(pady=8, padx=15, fill=X)
        
        Button(left_frame, text="📊 Статистика", command=self.show_statistics, 
               **btn_style).pack(pady=8, padx=15, fill=X)
        
        # Інформаційна панель
        info_frame = LabelFrame(left_frame, text="Лог активності", bg='#34495e', 
                                fg='white', font=("Arial", 10, "bold"))
        info_frame.pack(pady=15, padx=15, fill=BOTH, expand=True)
        
        self.info_text = Text(info_frame, height=12, width=28, bg='#2c3e50', 
                              fg='white', font=("Arial", 9))
        self.info_text.pack(padx=5, pady=5, fill=BOTH, expand=True)
        self.log_message("Система готова до роботи")
        
        # Права панель - відображення результатів
        right_frame = Frame(main_frame, bg='#34495e', bd=2, relief=RIDGE)
        right_frame.pack(side=RIGHT, fill=BOTH, expand=True)
        
        Label(right_frame, text="Результати розпізнавання", font=("Arial", 16, "bold"), 
              bg='#34495e', fg='white').pack(pady=15)
        
        # Область для зображення
        self.image_label = Label(right_frame, bg='#2c3e50', text="Зображення з'явиться тут", 
                                 fg='white', font=("Arial", 14))
        self.image_label.pack(pady=15, padx=20, fill=BOTH, expand=True)
        
        # Область для результатів
        self.result_frame = LabelFrame(right_frame, text="Результат ідентифікації", 
                                       bg='#34495e', fg='white', font=("Arial", 12, "bold"))
        self.result_frame.pack(pady=10, padx=20, fill=X)
        
        self.result_text = Text(self.result_frame, height=5, bg='#2c3e50', 
                                fg='#2ecc71', font=("Arial", 11, "bold"))
        self.result_text.pack(padx=10, pady=10, fill=X)
    
    def log_message(self, message):
        """Додавання повідомлення в лог"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.info_text.insert(END, f"[{timestamp}] {message}\n")
        self.info_text.see(END)
        self.root.update()
    
    def load_models(self):
        """Завантаження моделей"""
        try:
            # Gender models
            mobilenet_path = os.path.join(self.models_path, "gender_model_mobilenet.h5")
            resnet_path = os.path.join(self.models_path, "gender_model_resnet.h5")
            
            if os.path.exists(mobilenet_path):
                self.gender_model_mobilenet = load_model(mobilenet_path)
                self.log_message("✓ MobileNetV2 Gender завантажена")
            
            if os.path.exists(resnet_path):
                self.gender_model_resnet = load_model(resnet_path)
                self.log_message("✓ ResNet50 Gender завантажена")
            
            # Encoders для розпізнавання
            self.log_message("Завантаження encoders...")
            
            # MobileNetV2 encoder
            base_mobilenet = MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=(224, 224, 3),
                pooling='avg'
            )
            base_mobilenet.trainable = False
            self.face_encoder_mobilenet = base_mobilenet
            
            # ResNet50 encoder
            base_resnet = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=(224, 224, 3),
                pooling='avg'
            )
            base_resnet.trainable = False
            self.face_encoder_resnet = base_resnet
            
            self.log_message("✓ Encoders готові")
            
            # Face classifiers
            mobilenet_classifier_path = os.path.join(self.models_path, "face_classifier_mobilenet.h5")
            resnet_classifier_path = os.path.join(self.models_path, "face_classifier_resnet.h5")
            encoder_path = os.path.join(self.models_path, "label_encoder.pkl")
            
            if os.path.exists(mobilenet_classifier_path):
                self.face_classifier_mobilenet = load_model(mobilenet_classifier_path)
                self.log_message("✓ MobileNet Classifier завантажений")
            
            if os.path.exists(resnet_classifier_path):
                self.face_classifier_resnet = load_model(resnet_classifier_path)
                self.log_message("✓ ResNet Classifier завантажений")
            
            if os.path.exists(encoder_path):
                with open(encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)
            
            # Person counter
            counter_path = os.path.join(self.models_path, "person_counter.pkl")
            if os.path.exists(counter_path):
                with open(counter_path, 'rb') as f:
                    self.person_counter = pickle.load(f)
            
            # Training history
            history_path = os.path.join(self.models_path, "training_history.pkl")
            if os.path.exists(history_path):
                with open(history_path, 'rb') as f:
                    self.training_history = pickle.load(f)
                    
        except Exception as e:
            self.log_message(f"Помилка завантаження: {e}")
    
    def download_training_dataset(self):
        """Завантаження датасету"""
        try:
            self.log_message("=" * 40)
            self.log_message("Вибір датасету...")
            
            choice_window = Toplevel(self.root)
            choice_window.title("Вибір датасету")
            choice_window.geometry("500x350")
            choice_window.configure(bg='#2c3e50')
            choice_window.transient(self.root)
            choice_window.grab_set()
            
            Label(choice_window,
                  text="📦 Виберіть датасет для навчання",
                  font=("Arial", 16, "bold"),
                  bg='#2c3e50',
                  fg='white').pack(pady=20)
            
            selected = StringVar(value="")
            
            datasets = [
                ("UTKFace (Вік, Стать, Раса) - Рекомендовано", "jangedoo/utkface-new"),
                ("CelebA (Знаменитості)", "jessicali9530/celeba-dataset"),
            ]
            
            for name, slug in datasets:
                rb = Radiobutton(choice_window,
                                 text=name,
                                 variable=selected,
                                 value=slug,
                                 font=("Arial", 12),
                                 bg='#2c3e50',
                                 fg='white',
                                 selectcolor='#34495e',
                                 activebackground='#2c3e50',
                                 activeforeground='white',
                                 indicatoron=True)
                rb.pack(pady=8, padx=30, anchor=W)
            
            def confirm():
                if selected.get():
                    choice_window.destroy()
                    self.process_dataset_download(selected.get())
                else:
                    messagebox.showwarning("Увага", "Виберіть датасет!")
            
            Button(choice_window,
                   text="Завантажити і обробити",
                   command=confirm,
                   font=("Arial", 12, "bold"),
                   bg='#e74c3c',
                   fg='white',
                   activebackground='#c0392b',
                   bd=0,
                   padx=30,
                   pady=10).pack(pady=25)
            
        except Exception as e:
            self.log_message(f"✗ Помилка: {e}")
            messagebox.showerror("Помилка", str(e))
    
    def process_dataset_download(self, dataset_slug):
        """Завантаження і обробка датасету"""
        try:
            self.log_message(f"Завантаження: {dataset_slug}")
            self.log_message("Це може зайняти кілька хвилин...")
            
            path = kagglehub.dataset_download(dataset_slug)
            self.log_message(f"✓ Завантажено: {path}")
            
            self.log_message("Обробка зображень...")
            self.process_training_data(path, dataset_slug)
            
            messagebox.showinfo("Успіх", 
                              "Датасет успішно завантажено!\n"
                              "Готовий до навчання.")
            
        except Exception as e:
            self.log_message(f"✗ Помилка: {e}")
            messagebox.showerror("Помилка", str(e))
    
    def _load_celeba_gender_map(self, source_path):
        """Завантажує мапу статей з файлу атрибутів CelebA."""
        gender_map = {}
        attr_candidates = [
            os.path.join(source_path, "list_attr_celeba.txt"),
            os.path.join(source_path, "celeba", "list_attr_celeba.txt"),
        ]
        attr_path = next((p for p in attr_candidates if os.path.exists(p)), None)
        if not attr_path:
            self.log_message("✗ Не знайдено файл атрибутів CelebA")
            return gender_map
        
        try:
            with open(attr_path, 'r') as f:
                lines = f.read().splitlines()
            if len(lines) < 3:
                self.log_message("✗ Некоректний файл атрибутів CelebA")
                return gender_map
            attr_names = lines[1].split()
            if "Male" not in attr_names:
                self.log_message("✗ Атрибут 'Male' у CelebA відсутній")
                return gender_map
            gender_idx = attr_names.index("Male") + 1  # +1 бо перша колонка — ім'я файлу
            for line in lines[2:]:
                parts = line.split()
                if len(parts) <= gender_idx:
                    continue
                filename = parts[0]
                gender_flag = parts[gender_idx]
                gender_map[filename] = 'male' if gender_flag == '1' else 'female'
        except Exception as e:
            self.log_message(f"✗ Помилка читання атрибутів CelebA: {e}")
        return gender_map
    
    def _find_celeba_image_root(self, source_path):
        """Шукає директорію із зображеннями CelebA."""
        candidates = [
            os.path.join(source_path, "img_align_celeba"),
            os.path.join(source_path, "img_align_celeba", "img_align_celeba"),
            os.path.join(source_path, "celeba", "img_align_celeba"),
        ]
        for candidate in candidates:
            if os.path.isdir(candidate):
                return candidate
        for root, _, files in os.walk(source_path):
            if any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in files):
                return root
        return None
    
    def process_training_data(self, source_path, dataset_slug):
        """Обробка даних"""
        processed = 0
        gender_data = {'male': 0, 'female': 0}
        error_count = 0
        
        male_folder = os.path.join(self.training_data_path, "male")
        female_folder = os.path.join(self.training_data_path, "female")
        os.makedirs(male_folder, exist_ok=True)
        os.makedirs(female_folder, exist_ok=True)
        
        if dataset_slug == "jessicali9530/celeba-dataset":
            gender_map = self._load_celeba_gender_map(source_path)
            image_root = self._find_celeba_image_root(source_path)
            if not gender_map or not image_root:
                self.log_message("✗ CelebA: відсутні дані для обробки")
                return
            
            limits = {'male': 0, 'female': 0}
            for filename, gender in gender_map.items():
                if limits[gender] >= self.max_celeba_samples_per_gender:
                    continue
                
                img_path = os.path.join(image_root, filename)
                if not os.path.exists(img_path):
                    continue
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                    if len(faces) > 0:
                        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
                        x, y, w, h = faces[0]
                        face = img[y:y+h, x:x+w]
                    else:
                        face = img
                    face_resized = cv2.resize(face, (224, 224))
                    
                    save_folder = female_folder if gender == 'female' else male_folder
                    save_path = os.path.join(save_folder, f"{gender}_{processed:06d}.jpg")
                    cv2.imwrite(save_path, face_resized)
                    
                    limits[gender] += 1
                    gender_data[gender] += 1
                    processed += 1
                    
                    if processed % 500 == 0:
                        self.log_message(f"Оброблено: {processed}")
                        self.root.update()
                except Exception as e:
                    error_count += 1
                    if error_count % 50 == 0:
                        self.log_message(f"✗ Помилки під час обробки: {error_count}")
            self.log_message(f"✓ Оброблено CelebA: {processed}")
            self.log_message(f"Чоловіків: {gender_data['male']}")
            self.log_message(f"Жінок: {gender_data['female']}")
            if error_count:
                self.log_message(f"✗ Пропущено через помилки: {error_count}")
            return
        
        for root, dirs, files in os.walk(source_path):
            for file in files:
                if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                
                try:
                    filepath = os.path.join(root, file)
                    img = cv2.imread(filepath)
                    
                    if img is None:
                        continue
                    
                    if dataset_slug == "jangedoo/utkface-new":
                        face_resized = cv2.resize(img, (224, 224))
                    else:
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                        
                        if len(faces) == 0:
                            continue
                        
                        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
                        x, y, w, h = faces[0]
                        face = img[y:y+h, x:x+w]
                        face_resized = cv2.resize(face, (224, 224))
                    
                    gender = None
                    if dataset_slug == "jangedoo/utkface-new":
                        parts = file.split('_')
                        if len(parts) >= 2:
                            gender = 'male' if parts[1] == '0' else 'female'
                    else:
                        continue
                    
                    if gender:
                        save_folder = female_folder if gender == 'female' else male_folder
                        save_path = os.path.join(save_folder, f"{gender}_{processed:06d}.jpg")
                        cv2.imwrite(save_path, face_resized)
                        
                        gender_data[gender] += 1
                        processed += 1
                        
                        if processed % 100 == 0:
                            self.log_message(f"Оброблено: {processed}")
                            self.root.update()
                
                except Exception as e:
                    error_count += 1
                    if error_count % 50 == 0:
                        self.log_message(f"✗ Помилки під час обробки: {error_count}")
        
        self.log_message(f"✓ Оброблено всього: {processed}")
        self.log_message(f"Чоловіків: {gender_data['male']}")
        self.log_message(f"Жінок: {gender_data['female']}")
        if error_count:
            self.log_message(f"✗ Пропущено через помилки: {error_count}")

    def train_gender_model(self, model_type='mobilenet'):
        """Навчання моделі статі"""
        try:
            self.log_message("=" * 40)
            self.log_message(f"🚀 Навчання {model_type.upper()}...")
            
            male_folder = os.path.join(self.training_data_path, "male")
            female_folder = os.path.join(self.training_data_path, "female")
            
            if not os.path.exists(male_folder) or not os.path.exists(female_folder):
                messagebox.showerror("Помилка", "Спочатку завантажте датасет!")
                return
            
            X = []
            y = []
            
            self.log_message("Завантаження даних...")
            
            def load_images_from_folder(folder, label, max_count):
                count = 0
                files = [f for f in os.listdir(folder) if f.endswith(('.jpg', '.png'))]
                np.random.shuffle(files)
                
                for img_file in files[:max_count]:
                    img_path = os.path.join(folder, img_file)
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, (224, 224))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        X.append(img)
                        y.append(label)
                        count += 1
                return count

            m_count = load_images_from_folder(male_folder, 0, self.max_gender_samples)
            f_count = load_images_from_folder(female_folder, 1, self.max_gender_samples)
            
            self.log_message(f"Дані: {m_count} Male, {f_count} Female")
            if m_count == 0 or f_count == 0:
                messagebox.showerror("Помилка", "Недостатньо даних для навчання (порожня одна з категорій).")
                return
            
            X = np.array(X, dtype='float32')
            y = np.array(y)
            
            # Вибір препроцесінгу
            if model_type == 'mobilenet':
                X = mobilenet_preprocess(X)
                self.log_message("Препроцесінг: MobileNetV2")
            else:
                X = resnet_preprocess(X)
                self.log_message("Препроцесінг: ResNet50")
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Вибір base model
            if model_type == 'mobilenet':
                self.log_message("Архітектура: MobileNetV2")
                base_model = MobileNetV2(
                    input_shape=(224, 224, 3),
                    include_top=False,
                    weights='imagenet'
                )
            else:
                self.log_message("Архітектура: ResNet50")
                base_model = ResNet50(
                    input_shape=(224, 224, 3),
                    include_top=False,
                    weights='imagenet'
                )
            
            base_model.trainable = False
            
            # Data Augmentation
            data_augmentation = Sequential([
                RandomFlip("horizontal"),
                RandomRotation(0.1),
                RandomZoom(0.1),
            ])
            
            # Модель
            inputs = Input(shape=(224, 224, 3))
            x = data_augmentation(inputs)
            x = base_model(x, training=False)
            x = GlobalAveragePooling2D()(x)
            x = Dropout(0.2)(x)
            x = Dense(256, activation='relu')(x)
            x = Dropout(0.2)(x)
            outputs = Dense(1, activation='sigmoid')(x)
            
            model = Model(inputs, outputs)
            
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                          loss='binary_crossentropy',
                          metrics=['accuracy'])
            
            early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            
            self.log_message("Старт навчання...")
            
            history = model.fit(
                X_train, y_train,
                batch_size=32,
                epochs=50,
                validation_data=(X_test, y_test),
                callbacks=[early_stop],
                verbose=1
            )
            
            # Збереження
            if model_type == 'mobilenet':
                model_path = os.path.join(self.models_path, "gender_model_mobilenet.h5")
                self.gender_model_mobilenet = model
            else:
                model_path = os.path.join(self.models_path, "gender_model_resnet.h5")
                self.gender_model_resnet = model
            
            model.save(model_path)
            
            # Збереження історії
            self.training_history[model_type] = history.history
            history_path = os.path.join(self.models_path, "training_history.pkl")
            with open(history_path, 'wb') as f:
                pickle.dump(self.training_history, f)
            
            val_acc = history.history['val_accuracy'][-1]
            self.log_message(f"✓ {model_type.upper()}: {val_acc*100:.1f}%")
            
            messagebox.showinfo("Успіх", f"{model_type.upper()} навчена!\nТочність: {val_acc*100:.1f}%")
            
        except Exception as e:
            self.log_message(f"✗ Помилка: {e}")
            print(e)
            messagebox.showerror("Помилка", str(e))
    
    def compare_models(self):
        """Порівняння моделей"""
        if not self.training_history['mobilenet'] or not self.training_history['resnet']:
            messagebox.showwarning("Увага", "Спочатку навчіть обидві моделі!")
            return
        
        try:
            # Створення вікна порівняння
            compare_window = Toplevel(self.root)
            compare_window.title("Порівняння моделей")
            compare_window.geometry("1200x700")
            compare_window.configure(bg='#2c3e50')
            
            Label(compare_window, text="📊 Порівняння MobileNetV2 vs ResNet50",
                  font=("Arial", 18, "bold"), bg='#2c3e50', fg='white').pack(pady=15)
            
            # Статистика
            stats_frame = Frame(compare_window, bg='#34495e', bd=2, relief=RIDGE)
            stats_frame.pack(pady=10, padx=20, fill=X)
            
            mobilenet_acc = self.training_history['mobilenet']['val_accuracy'][-1] * 100
            resnet_acc = self.training_history['resnet']['val_accuracy'][-1] * 100
            
            mobilenet_loss = self.training_history['mobilenet']['val_loss'][-1]
            resnet_loss = self.training_history['resnet']['val_loss'][-1]
            
            stats_text = f"""
            📱 MobileNetV2:
               • Validation Accuracy: {mobilenet_acc:.2f}%
               • Validation Loss: {mobilenet_loss:.4f}
               • Epochs: {len(self.training_history['mobilenet']['accuracy'])}
            
            🔷 ResNet50:
               • Validation Accuracy: {resnet_acc:.2f}%
               • Validation Loss: {resnet_loss:.4f}
               • Epochs: {len(self.training_history['resnet']['accuracy'])}
            
            🏆 Переможець: {"MobileNetV2" if mobilenet_acc > resnet_acc else "ResNet50" if resnet_acc > mobilenet_acc else "Нічия"}
            """
            
            Label(stats_frame, text=stats_text, font=("Courier", 12), bg='#34495e', 
                  fg='#2ecc71', justify=LEFT).pack(pady=15, padx=20)
            
            # Графіки
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            fig.patch.set_facecolor('#2c3e50')
            
            # Accuracy
            axes[0].plot(self.training_history['mobilenet']['accuracy'], label='MobileNet Train', color='#3498db', linewidth=2)
            axes[0].plot(self.training_history['mobilenet']['val_accuracy'], label='MobileNet Val', color='#3498db', linestyle='--', linewidth=2)
            axes[0].plot(self.training_history['resnet']['accuracy'], label='ResNet Train', color='#e67e22', linewidth=2)
            axes[0].plot(self.training_history['resnet']['val_accuracy'], label='ResNet Val', color='#e67e22', linestyle='--', linewidth=2)
            axes[0].set_title('Accuracy Comparison', fontsize=14, color='white', weight='bold')
            axes[0].set_xlabel('Epoch', fontsize=12, color='white')
            axes[0].set_ylabel('Accuracy', fontsize=12, color='white')
            axes[0].legend(fontsize=10, loc='lower right')
            axes[0].grid(True, alpha=0.3)
            axes[0].set_facecolor('#34495e')
            axes[0].tick_params(colors='white')
            
            # Loss
            axes[1].plot(self.training_history['mobilenet']['loss'], label='MobileNet Train', color='#3498db', linewidth=2)
            axes[1].plot(self.training_history['mobilenet']['val_loss'], label='MobileNet Val', color='#3498db', linestyle='--', linewidth=2)
            axes[1].plot(self.training_history['resnet']['loss'], label='ResNet Train', color='#e67e22', linewidth=2)
            axes[1].plot(self.training_history['resnet']['val_loss'], label='ResNet Val', color='#e67e22', linestyle='--', linewidth=2)
            axes[1].set_title('Loss Comparison', fontsize=14, color='white', weight='bold')
            axes[1].set_xlabel('Epoch', fontsize=12, color='white')
            axes[1].set_ylabel('Loss', fontsize=12, color='white')
            axes[1].legend(fontsize=10, loc='upper right')
            axes[1].grid(True, alpha=0.3)
            axes[1].set_facecolor('#34495e')
            axes[1].tick_params(colors='white')
            
            plt.tight_layout()
            
            canvas = FigureCanvasTkAgg(fig, master=compare_window)
            canvas.draw()
            canvas.get_tk_widget().pack(pady=10, padx=20, fill=BOTH, expand=True)
            
            self.log_message("✓ Порівняння відображено")
            
        except Exception as e:
            self.log_message(f"✗ Помилка: {e}")
            messagebox.showerror("Помилка", str(e))
    
    def train_face_recognition_menu(self):
        """Меню вибору моделі для навчання розпізнавання"""
        choice_window = Toplevel(self.root)
        choice_window.title("Вибір моделі")
        choice_window.geometry("400x250")
        choice_window.configure(bg='#2c3e50')
        choice_window.transient(self.root)
        choice_window.grab_set()
        
        Label(choice_window, text="Виберіть модель для навчання розпізнавання:",
              font=("Arial", 12, "bold"), bg='#2c3e50', fg='white').pack(pady=20)
        
        Button(choice_window, text="📱 MobileNetV2", 
               command=lambda: [choice_window.destroy(), self.train_face_recognition('mobilenet')],
               font=("Arial", 11, "bold"), bg='#9b59b6', fg='white', 
               activebackground='#8e44ad', bd=0, padx=30, pady=10).pack(pady=10)
        
        Button(choice_window, text="🔷 ResNet50", 
               command=lambda: [choice_window.destroy(), self.train_face_recognition('resnet')],
               font=("Arial", 11, "bold"), bg='#e67e22', fg='white', 
               activebackground='#d35400', bd=0, padx=30, pady=10).pack(pady=10)
    
    def train_face_recognition(self, model_type='mobilenet'):
        """Навчання розпізнавання облич"""
        try:
            self.log_message("=" * 40)
            self.log_message(f"Навчання розпізнавання ({model_type.upper()})...")
            
            if not os.path.exists(self.dataset_path) or len(os.listdir(self.dataset_path)) == 0:
                messagebox.showinfo("Інфо", "Немає даних.\nЗавантажте фото для аналізу.")
                return
            
            X = []
            y = []
            
            # Вибір encoder та preprocess
            if model_type == 'mobilenet':
                encoder = self.face_encoder_mobilenet
                preprocess_fn = mobilenet_preprocess
            else:
                encoder = self.face_encoder_resnet
                preprocess_fn = resnet_preprocess
            
            self.log_message("Завантаження даних осіб...")
            
            for person_folder in os.listdir(self.dataset_path):
                person_path = os.path.join(self.dataset_path, person_folder)
                if not os.path.isdir(person_path):
                    continue
                
                for img_file in os.listdir(person_path):
                    if not img_file.endswith(('.jpg', '.jpeg', '.png')):
                        continue
                    
                    img_path = os.path.join(person_path, img_file)
                    img = cv2.imread(img_path)
                    
                    if img is None:
                        continue
                    
                    img_resized = cv2.resize(img, (224, 224))
                    img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                    
                    img_array = img_to_array(img_resized)
                    img_array = np.expand_dims(img_array, axis=0)
                    img_array = preprocess_fn(img_array)
                    
                    embedding = encoder.predict(img_array, verbose=0)
                    
                    X.append(embedding[0])
                    y.append(person_folder)
            
            if len(X) == 0:
                self.log_message("Немає зображень для навчання!")
                return

            X = np.array(X)
            y = np.array(y)
            
            self.log_message(f"✓ Завантажено: {len(X)} фото")
            self.log_message(f"✓ Осіб: {len(set(y))}")
            
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)
            
            num_classes = len(set(y))
            if num_classes < 2:
                messagebox.showinfo("Інфо", "Потрібно щонайменше дві особи для навчання класифікатора.")
                return
            
            val_data = None
            if len(X) > 4:
                try:
                    X_train, X_val, y_train, y_val = train_test_split(
                        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
                    )
                    val_data = (X_val, y_val)
                except Exception:
                    X_train, y_train = X, y_encoded
            else:
                X_train, y_train = X, y_encoded
            
            model = Sequential([
                Dense(512, activation='relu', input_shape=(X.shape[1],)),
                BatchNormalization(),
                Dropout(0.5),
                Dense(256, activation='relu'),
                Dropout(0.3),
                Dense(num_classes, activation='softmax')
            ])
            
            model.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])
            
            callbacks = []
            if val_data:
                callbacks.append(EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True))
            
            history = model.fit(
                X_train, y_train,
                epochs=50,
                batch_size=8,
                validation_data=val_data,
                callbacks=callbacks,
                verbose=1
            )
            
            # Збереження
            if model_type == 'mobilenet':
                model_path = os.path.join(self.models_path, "face_classifier_mobilenet.h5")
                self.face_classifier_mobilenet = model
            else:
                model_path = os.path.join(self.models_path, "face_classifier_resnet.h5")
                self.face_classifier_resnet = model
            
            model.save(model_path)
            
            encoder_path = os.path.join(self.models_path, "label_encoder.pkl")
            with open(encoder_path, 'wb') as f:
                pickle.dump(self.label_encoder, f)
            
            if val_data and 'val_accuracy' in history.history:
                acc = history.history['val_accuracy'][-1]
            else:
                acc = history.history['accuracy'][-1]
            self.log_message(f"✓ Точність: {acc*100:.1f}%")
            
        except Exception as e:
            self.log_message(f"✗ Помилка: {e}")
            messagebox.showerror("Помилка", str(e))
    
    def analyze_photo(self):
        """Аналіз фото з вибором моделі"""
        # Вибір моделі
        choice_window = Toplevel(self.root)
        choice_window.title("Вибір моделі")
        choice_window.geometry("400x250")
        choice_window.configure(bg='#2c3e50')
        choice_window.transient(self.root)
        choice_window.grab_set()
        
        Label(choice_window, text="Виберіть модель для розпізнавання:",
              font=("Arial", 12, "bold"), bg='#2c3e50', fg='white').pack(pady=20)
        
        def analyze_with_model(model_type):
            choice_window.destroy()
            self._analyze_photo_internal(model_type)
        
        Button(choice_window, text="📱 MobileNetV2", 
               command=lambda: analyze_with_model('mobilenet'),
               font=("Arial", 11, "bold"), bg='#9b59b6', fg='white', 
               activebackground='#8e44ad', bd=0, padx=30, pady=10).pack(pady=10)
        
        Button(choice_window, text="🔷 ResNet50", 
               command=lambda: analyze_with_model('resnet'),
               font=("Arial", 11, "bold"), bg='#e67e22', fg='white', 
               activebackground='#d35400', bd=0, padx=30, pady=10).pack(pady=10)
    
    def _analyze_photo_internal(self, model_type):
        """Внутрішня функція аналізу фото"""
        file_path = filedialog.askopenfilename(
            title="Виберіть зображення",
            filetypes=[("Зображення", "*.jpg *.jpeg *.png")]
        )
        
        if not file_path:
            return
        
        try:
            self.log_message("=" * 40)
            self.log_message(f"Аналіз ({model_type.upper()})...")
            
            img = cv2.imread(file_path)
            display_img = img.copy()
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) == 0:
                self.log_message("✗ Обличчя не знайдено")
                messagebox.showwarning("Увага", "Обличчя не знайдено!")
                self.display_image(display_img)
                return
            
            results = []
            
            for idx, (x, y, w, h) in enumerate(faces, 1):
                face = img[y:y+h, x:x+w]
                face_resized = cv2.resize(face, (224, 224))
                
                gender, gender_conf = self.predict_gender(face_resized, model_type)
                person_id, person_conf = self.recognize_person(face_resized, model_type)
                
                if person_id is None:
                    status = "Невідома особа"
                    person_conf = 0.0
                else:
                    status = f"Збіг: {person_conf:.1f}%"
                
                color = (0, 255, 0) if person_id else (0, 165, 255)
                person_label = person_id if person_id else "Unknown"
                cv2.rectangle(display_img, (x, y), (x+w, y+h), color, 3)
                
                gender_en = "Male" if gender == "Male" else "Female" if gender == "Female" else "Unknown"
                
                text_y = y - 30
                if text_y < 20: text_y = y + h + 20
                
                cv2.putText(display_img, f"ID: {person_label}", (x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(display_img, f"{gender_en} {gender_conf:.0f}%", (x, text_y + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(display_img, f"Model: {model_type}", (x, text_y + 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                
                results.append({
                    'person_id': person_id,
                    'gender': gender,
                    'gender_confidence': gender_conf,
                    'status': status,
                    'person_confidence': person_conf,
                    'model': model_type
                })
            
            self.display_image(display_img)
            self.display_results(results)
            
        except Exception as e:
            self.log_message(f"✗ Помилка: {e}")
            print(e)
    
    def predict_gender(self, face_img, model_type='mobilenet'):
        """Визначення статі"""
        model = self.gender_model_mobilenet if model_type == 'mobilenet' else self.gender_model_resnet
        
        if model is None:
            return "Невідомо", 0.0
        
        try:
            face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            img_array = img_to_array(face_img_rgb)
            img_array = np.expand_dims(img_array, axis=0)
            
            if model_type == 'mobilenet':
                img_array = mobilenet_preprocess(img_array)
            else:
                img_array = resnet_preprocess(img_array)
            
            pred = model.predict(img_array, verbose=0)[0][0]
            
            if pred > 0.5:
                gender = "Female"
                confidence = pred * 100
            else:
                gender = "Male"
                confidence = (1 - pred) * 100
            
            return gender, confidence
            
        except:
            return "Невідомо", 0.0
    
    def recognize_person(self, face_img, model_type='mobilenet'):
        """Розпізнавання особи"""
        encoder = self.face_encoder_mobilenet if model_type == 'mobilenet' else self.face_encoder_resnet
        classifier = self.face_classifier_mobilenet if model_type == 'mobilenet' else self.face_classifier_resnet
        
        if classifier is None or self.label_encoder is None:
            return None, 0.0
        
        try:
            face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            img_array = img_to_array(face_img_rgb)
            img_array = np.expand_dims(img_array, axis=0)
            
            if model_type == 'mobilenet':
                img_array = mobilenet_preprocess(img_array)
            else:
                img_array = resnet_preprocess(img_array)
            
            embedding = encoder.predict(img_array, verbose=0)
            
            predictions = classifier.predict(embedding, verbose=0)
            confidence = np.max(predictions) * 100
            
            if confidence < self.min_recognition_confidence * 100:
                return None, 0.0
            
            label_idx = np.argmax(predictions)
            person_folder = self.label_encoder.inverse_transform([label_idx])[0]
            person_id = person_folder.split('_')[-1]
            
            return person_id, confidence
            
        except:
            return None, 0.0
    
    def create_new_person(self, face_img, gender):
        """Створення нової особи"""
        self.person_counter += 1
        person_id = f"{self.person_counter:03d}"
        
        person_folder = f"Person_{person_id}"
        person_path = os.path.join(self.dataset_path, person_folder)
        os.makedirs(person_path, exist_ok=True)
        
        face_path = os.path.join(person_path, f"face_001.jpg")
        cv2.imwrite(face_path, face_img)
        
        metadata = {
            'gender': gender,
            'created': datetime.now().isoformat()
        }
        
        with open(os.path.join(person_path, "metadata.pkl"), 'wb') as f:
            pickle.dump(metadata, f)
        
        with open(os.path.join(self.models_path, "person_counter.pkl"), 'wb') as f:
            pickle.dump(self.person_counter, f)
        
        self.log_message(f"✨ Створено: Особа {person_id}")
        
        return person_id
    
    def analyze_camera(self):
        """Аналіз з камери з вибором моделі"""
        choice_window = Toplevel(self.root)
        choice_window.title("Вибір моделі")
        choice_window.geometry("400x250")
        choice_window.configure(bg='#2c3e50')
        choice_window.transient(self.root)
        choice_window.grab_set()
        
        Label(choice_window, text="Виберіть модель для розпізнавання:",
              font=("Arial", 12, "bold"), bg='#2c3e50', fg='white').pack(pady=20)
        
        def analyze_with_model(model_type):
            choice_window.destroy()
            self._analyze_camera_internal(model_type)
        
        Button(choice_window, text="📱 MobileNetV2", 
               command=lambda: analyze_with_model('mobilenet'),
               font=("Arial", 11, "bold"), bg='#9b59b6', fg='white', 
               activebackground='#8e44ad', bd=0, padx=30, pady=10).pack(pady=10)
        
        Button(choice_window, text="🔷 ResNet50", 
               command=lambda: analyze_with_model('resnet'),
               font=("Arial", 11, "bold"), bg='#e67e22', fg='white', 
               activebackground='#d35400', bd=0, padx=30, pady=10).pack(pady=10)
    
    def _analyze_camera_internal(self, model_type):
        """Внутрішня функція аналізу з камери"""
        model = self.gender_model_mobilenet if model_type == 'mobilenet' else self.gender_model_resnet
        
        if model is None:
            messagebox.showerror("Помилка", f"Спочатку навчіть модель {model_type.upper()}!")
            return
        
        self.log_message(f"Запуск камери ({model_type.upper()})...")
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            messagebox.showerror("Помилка", "Не вдалося відкрити камеру!")
            return
        
        cv2.namedWindow(f"AI Recognition - {model_type.upper()} (Q - Exit)")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                face_resized = cv2.resize(face, (224, 224))
                
                gender, gender_conf = self.predict_gender(face_resized, model_type)
                person_id, person_conf = self.recognize_person(face_resized, model_type)
                
                color = (0, 255, 0) if person_id else (0, 165, 255)
                text = f"ID: {person_id}" if person_id else "New"
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, text, (x, y-40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame, f"{gender} {gender_conf:.0f}%", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            cv2.imshow(f"AI Recognition - {model_type.upper()} (Q - Exit)", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        self.log_message("✓ Камера зупинена")
    
    def display_image(self, cv_img):
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        height, width = cv_img.shape[:2]
        max_width = 900
        max_height = 550
        
        if width > max_width or height > max_height:
            scaling_factor = min(max_width/width, max_height/height)
            new_width = int(width * scaling_factor)
            new_height = int(height * scaling_factor)
            cv_img = cv2.resize(cv_img, (new_width, new_height))
        
        pil_img = Image.fromarray(cv_img)
        photo = ImageTk.PhotoImage(image=pil_img)
        self.image_label.configure(image=photo, text="")
        self.image_label.image = photo
    
    def display_results(self, results):
        self.result_text.delete(1.0, END)
        if not results:
            self.result_text.insert(END, "Обличчя не знайдено\n")
            return
        
        for i, res in enumerate(results, 1):
            self.result_text.insert(END, f"═══ Особа {i} [{res.get('model', 'N/A').upper()}] ═══\n")
            self.result_text.insert(END, f"ID: {res['person_id'] or 'Нова'}\n")
            self.result_text.insert(END, f"Стать: {res['gender']}\n")
            self.result_text.insert(END, f"Впевненість: {res['gender_confidence']:.1f}%\n\n")

    def show_statistics(self):
        try:
            total_persons = len([d for d in os.listdir(self.dataset_path) 
                               if os.path.isdir(os.path.join(self.dataset_path, d))])
            
            total_images = 0
            for person_folder in os.listdir(self.dataset_path):
                person_path = os.path.join(self.dataset_path, person_folder)
                if os.path.isdir(person_path):
                    total_images += len([f for f in os.listdir(person_path) 
                                       if f.endswith(('.jpg', '.png'))])
            
            mobilenet_status = '✓ Навчена' if self.gender_model_mobilenet else '✗ Не навчена'
            resnet_status = '✓ Навчена' if self.gender_model_resnet else '✗ Не навчена'
            
            msg = f"📊 Статистика системи\n\n"
            msg += f"Осіб у базі: {total_persons}\n"
            msg += f"Фото у базі: {total_images}\n\n"
            msg += f"📱 MobileNetV2: {mobilenet_status}\n"
            msg += f"🔷 ResNet50: {resnet_status}"
            
            messagebox.showinfo("Статистика", msg)
            
        except Exception as e:
            messagebox.showerror("Помилка", str(e))

def main():
    root = Tk()
    app = FaceRecognitionSystem(root)
    root.mainloop()

if __name__ == "__main__":
    main()
