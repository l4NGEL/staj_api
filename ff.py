from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import os
import json
from datetime import datetime
import base64
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import io
import pickle
import tensorflow as tf
import shutil
from mtcnn import MTCNN
import random
import math
from scipy import ndimage
from sklearn.metrics import roc_curve, auc
from itertools import combinations
from collections import defaultdict

app = Flask(__name__)
CORS(app)

USERS_DB_FILE = 'users.json'
KNOWN_FACES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'faces')
RECOGNITION_LOG_FILE = 'logs.json'
MOBILFACENET_MODEL_PATH = 'facenet.tflite'
THRESHOLD_LOG_FILE = 'threshold_logs.json'

os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

# Augmentation fonksiyonları
def apply_brightness_contrast(image, brightness_factor=1.0, contrast_factor=1.0):
    """Parlaklık ve kontrast ayarlama"""
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness_factor)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast_factor)
    return image

def apply_noise(image, noise_factor=0.02):
    """Gürültü ekleme"""
    img_array = np.array(image)
    noise = np.random.normal(0, noise_factor * 255, img_array.shape)
    noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img)

def apply_blur(image, blur_factor=1.0):
    """Bulanıklık uygulama"""
    return image.filter(ImageFilter.GaussianBlur(radius=blur_factor))

def apply_sharpening(image, sharpness_factor=1.5):
    """Keskinleştirme uygulama"""
    enhancer = ImageEnhance.Sharpness(image)
    return enhancer.enhance(sharpness_factor)

def apply_geometric_transform(image, rotation_range=10, scale_range=0.1):
    """Geometrik dönüşümler"""
    # Rastgele rotasyon
    angle = random.uniform(-rotation_range, rotation_range)
    # Rastgele ölçekleme
    scale = random.uniform(1 - scale_range, 1 + scale_range)

    # Rotasyon uygula
    if abs(angle) > 1:
        image = image.rotate(angle, expand=True)

    # Ölçekleme uygula
    if abs(scale - 1.0) > 0.05:
        new_size = (int(image.width * scale), int(image.height * scale))
        image = image.resize(new_size, Image.Resampling.LANCZOS)

    return image

def apply_lighting_simulation(image, lighting_factor=0.3):
    """Aydınlatma simülasyonu"""
    img_array = np.array(image)

    # Rastgele aydınlatma maskesi oluştur
    height, width = img_array.shape[:2]
    y, x = np.ogrid[:height, :width]

    # Rastgele merkez
    center_x = random.uniform(0, width)
    center_y = random.uniform(0, height)

    # Mesafe hesapla
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    max_distance = np.sqrt(width**2 + height**2)

    # Aydınlatma maskesi
    lighting_mask = 1 - (distance / max_distance) * lighting_factor
    lighting_mask = np.clip(lighting_mask, 0.7, 1.3)

    # Maskeyi 3 kanala genişlet
    if len(img_array.shape) == 3:
        lighting_mask = np.stack([lighting_mask] * 3, axis=2)

    # Aydınlatma uygula
    lighted_img = np.clip(img_array * lighting_mask, 0, 255).astype(np.uint8)
    return Image.fromarray(lighted_img)

def create_augmented_versions(image, num_versions=8):
    """Bir görüntüden birden fazla augmented versiyon oluştur"""
    augmented_images = []

    for i in range(num_versions):
        aug_img = image.copy()

        # Rastgele augmentation kombinasyonları
        if random.random() < 0.7:
            brightness = random.uniform(0.8, 1.2)
            contrast = random.uniform(0.8, 1.2)
            aug_img = apply_brightness_contrast(aug_img, brightness, contrast)

        if random.random() < 0.5:
            noise_factor = random.uniform(0.01, 0.03)
            aug_img = apply_noise(aug_img, noise_factor)

        if random.random() < 0.3:
            blur_factor = random.uniform(0.5, 1.5)
            aug_img = apply_blur(aug_img, blur_factor)

        if random.random() < 0.4:
            sharpness = random.uniform(1.2, 1.8)
            aug_img = apply_sharpening(aug_img, sharpness)

        if random.random() < 0.6:
            aug_img = apply_geometric_transform(aug_img)

        if random.random() < 0.4:
            lighting = random.uniform(0.2, 0.4)
            aug_img = apply_lighting_simulation(aug_img, lighting)

        augmented_images.append(aug_img)

    return augmented_images

def get_augmentation_stats():
    """Augmentation istatistiklerini döndür"""
    return {
        'augmentation_types': [
            'brightness_contrast',
            'noise',
            'blur',
            'sharpening',
            'geometric_transform',
            'lighting_simulation'
        ],
        'default_versions_per_image': 8,
        'supported_transformations': [
            'Parlaklık ve kontrast ayarlama',
            'Gürültü ekleme',
            'Bulanıklık uygulama',
            'Keskinleştirme',
            'Geometrik dönüşümler (rotasyon, ölçekleme)',
            'Aydınlatma simülasyonu'
        ]
    }

def auto_rotate_image(image):
    """
    Görüntüyü EXIF verilerine göre otomatik olarak döndürür
    """
    try:
        # EXIF verilerini kontrol et
        if hasattr(image, '_getexif') and image._getexif() is not None:
            exif = image._getexif()
            if exif is not None:
                orientation = exif.get(274)  # EXIF orientation tag
                if orientation is not None:
                    # Orientation değerine göre döndür
                    if orientation == 3:
                        image = image.rotate(180, expand=True)
                    elif orientation == 6:
                        image = image.rotate(270, expand=True)
                    elif orientation == 8:
                        image = image.rotate(90, expand=True)
    except Exception as e:
        print(f"EXIF rotasyon hatası: {e}")

    return image

def fix_corrupted_json():
    """
    Bozuk JSON dosyasını düzeltir
    """
    try:
        if not os.path.exists(USERS_DB_FILE):
            print("users_db.json dosyası bulunamadı, yeni dosya oluşturuluyor...")
            with open(USERS_DB_FILE, 'w', encoding='utf-8') as f:
                json.dump([], f, ensure_ascii=False, indent=2)
            return True

        # Dosyayı oku ve temizle
        with open(USERS_DB_FILE, 'r', encoding='utf-8') as f:
            content = f.read()

        print(f"Dosya boyutu: {len(content)} karakter")

        # Sadece ilk geçerli JSON array'i al
        try:
            first_bracket = content.index('[')
            last_bracket = content.rindex(']')
            valid_json = content[first_bracket:last_bracket+1]

            # JSON'u parse et
            users = json.loads(valid_json)
            print(f"Geçerli kullanıcı sayısı: {len(users)}")

            # Temiz JSON'u tekrar kaydet
            with open(USERS_DB_FILE, 'w', encoding='utf-8') as f:
                json.dump(users, f, ensure_ascii=False, indent=2)

            print("JSON dosyası başarıyla temizlendi")
            return True

        except ValueError as e:
            print(f"JSON parse hatası: {e}")
            # Dosya tamamen bozuksa, yeni dosya oluştur
            with open(USERS_DB_FILE, 'w', encoding='utf-8') as f:
                json.dump([], f, ensure_ascii=False, indent=2)
            print("Yeni JSON dosyası oluşturuldu")
            return True

    except Exception as e:
        print(f"JSON düzeltme hatası: {e}")
        return False

class FaceRecognitionAPI:
    def __init__(self):
        self.mobilfacenet_interpreter = None
        self.input_details = None
        self.output_details = None
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_ids = []
        self.known_embeddings_array = None
        self.users_data_cache = []
        self.auto_threshold_enabled = True  # Otomatik threshold aktif
        self.optimal_threshold = 0.7437  # Başlangıç threshold değeri
        self.threshold_history = []  # Threshold geçmişi

        # MTCNN yüz tespit edici
        self.mtcnn_detector = MTCNN()

        # Tanınan kişileri takip etmek için set
        self.recognized_persons = set()
        # Gerçek zamanlı tanıma kayıtları
        self.realtime_recognition_logs = []
        # Performans optimizasyonu için cache
        self.last_cache_update = None

        self.load_mobilfacenet_model(MOBILFACENET_MODEL_PATH)
        self.load_known_faces()
        self.update_cache()

    def update_cache(self):
        """
        Performans için cache'i günceller
        """
        try:
            # Embeddings array'ini önceden hesapla
            if len(self.known_face_encodings) > 0:
                self.known_embeddings_array = np.array(self.known_face_encodings)

            # Users data cache'ini güncelle
            with open(USERS_DB_FILE, 'r', encoding='utf-8') as f:
                self.users_data_cache = json.load(f)

            self.last_cache_update = datetime.now()
        except Exception as e:
            print(f"Cache güncelleme hatası: {e}")

    def load_mobilfacenet_model(self, model_path):
        self.mobilfacenet_interpreter = tf.lite.Interpreter(model_path=model_path)
        self.mobilfacenet_interpreter.allocate_tensors()
        self.input_details = self.mobilfacenet_interpreter.get_input_details()
        self.output_details = self.mobilfacenet_interpreter.get_output_details()

    def get_embedding(self, face_img):
        try:
            # Yüz boyutunu kontrol et
            height, width = face_img.shape[:2]
            print(f"Yüz boyutu: {width}x{height}")

            # Minimum boyut kontrolü
            if width < 80 or height < 80:
                print(f"Yüz çok küçük: {width}x{height}, minimum 80x80 gerekli")
                return None

            # Model için 112x112'ye yeniden boyutlandır
            img = cv2.resize(face_img, (160, 160), interpolation=cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype('float32') / 255.0
            img = np.expand_dims(img, axis=0)

            self.mobilfacenet_interpreter.set_tensor(self.input_details[0]['index'], img)
            self.mobilfacenet_interpreter.invoke()
            embedding = self.mobilfacenet_interpreter.get_tensor(self.output_details[0]['index'])
            embedding = embedding.flatten()

            # L2 normalizasyonu uygula
            embedding = embedding / np.linalg.norm(embedding)

            return embedding
        except Exception as e:
            print(f"Embedding oluşturma hatası: {e}")
            return None

    def calculate_optimal_threshold(self, test_embeddings, known_embeddings):
        """Otomatik olarak optimal threshold değerini hesaplar"""
        try:
            if len(test_embeddings) == 0 or len(known_embeddings) == 0:
                return self.optimal_threshold

            # Tüm embedding'ler arasındaki mesafeleri hesapla
            distances = []
            for test_emb in test_embeddings:
                for known_emb in known_embeddings:
                    dist = float(np.linalg.norm(test_emb - known_emb))  # float32'yi float'a çevir
                    distances.append(dist)

            if len(distances) == 0:
                return self.optimal_threshold

            print(f"{len(distances)} çift oluşturuldu.")
            print(f"{len(self.known_face_encodings)} adet encoding kullanıldı.")

            # Mesafeleri sırala
            distances.sort()

            # Percentile tabanlı threshold hesaplama
            # %95 percentile kullan (daha sıkı eşleşme)
            percentile_95 = float(np.percentile(distances, 95))  # float32'yi float'a çevir
            percentile_90 = float(np.percentile(distances, 90))  # float32'yi float'a çevir
            percentile_85 = float(np.percentile(distances, 85))  # float32'yi float'a çevir

            # Ortalama ve standart sapma
            mean_dist = float(np.mean(distances))  # float32'yi float'a çevir
            std_dist = float(np.std(distances))  # float32'yi float'a çevir

            # Farklı threshold stratejileri
            threshold_candidates = [
                percentile_95,
                percentile_90,
                percentile_85,
                mean_dist + std_dist,
                mean_dist + 0.5 * std_dist,
                self.optimal_threshold  # Mevcut değer
            ]

            # En iyi threshold'u seç (en düşük değer)
            optimal_threshold = min(threshold_candidates)

            # Threshold'u sınırla (çok düşük veya yüksek olmasın)
            optimal_threshold = max(0.3, min(0.8, optimal_threshold))

            # Threshold geçmişini güncelle
            self.threshold_history.append({
                'timestamp': datetime.now().isoformat(),
                'threshold': optimal_threshold,
                'mean_distance': mean_dist,
                'std_distance': std_dist,
                'percentile_95': percentile_95,
                'percentile_90': percentile_90,
                'percentile_85': percentile_85
            })

            # Geçmişi 100 kayıtla sınırla
            if len(self.threshold_history) > 100:
                self.threshold_history = self.threshold_history[-100:]

            self.optimal_threshold = optimal_threshold
            print(f"Optimal threshold hesaplandı: {optimal_threshold:.4f}")
            return optimal_threshold

        except Exception as e:
            print(f"Threshold hesaplama hatası: {e}")
            return self.optimal_threshold

    def get_adaptive_threshold(self):
        """Adaptif threshold değerini döndürür"""
        if not self.auto_threshold_enabled:
            return 0.7437  # Sabit değer

        # Son threshold geçmişinden ortalama al
        if len(self.threshold_history) > 0:
            recent_thresholds = [float(h['threshold']) for h in self.threshold_history[-10:]]  # float32'yi float'a çevir
            return float(np.mean(recent_thresholds))  # float32'yi float'a çevir

        return self.optimal_threshold

    def calculate_threshold_via_roc(self):
        """
        ROC eğrisi kullanarak tüm veritabanı üzerinden optimal threshold hesaplar
        """
        try:
            print("ROC tabanlı threshold hesaplaması başlatıldı...")

            # Embeddings'leri kişilere göre grupla
            embeddings_dict = defaultdict(list)
            for name, id_no, emb in zip(self.known_face_names, self.known_face_ids, self.known_face_encodings):
                embeddings_dict[id_no].append(emb)

            if len(embeddings_dict) < 2:
                print("ROC için yeterli kullanıcı yok (en az 2 kullanıcı gerekli)")
                return self.optimal_threshold

            y_true = []
            distances = []

            # Pozitif çiftler (aynı kişi)
            print("Pozitif çiftler hesaplanıyor...")
            for emb_list in embeddings_dict.values():
                if len(emb_list) >= 2:  # En az 2 embedding gerekli
                    for emb1, emb2 in combinations(emb_list, 2):
                        dist = np.linalg.norm(emb1 - emb2)
                        distances.append(dist)
                        y_true.append(1)

            # Negatif çiftler (farklı kişiler)
            print("Negatif çiftler hesaplanıyor...")
            person_ids = list(embeddings_dict.keys())
            for i in range(len(person_ids)):
                for j in range(i + 1, len(person_ids)):
                    for emb1 in embeddings_dict[person_ids[i]]:
                        for emb2 in embeddings_dict[person_ids[j]]:
                            dist = np.linalg.norm(emb1 - emb2)
                            distances.append(dist)
                            y_true.append(0)

            if not distances or not y_true:
                print("ROC için yeterli veri yok")
                return self.optimal_threshold

            print(f"Toplam {len(distances)} çift hesaplandı ({sum(y_true)} pozitif, {len(y_true) - sum(y_true)} negatif)")
            print(f"{len(distances)} çift oluşturuldu.")
            print(f"{len(self.known_face_encodings)} adet encoding kullanıldı.")

            # ROC eğrisi hesapla
            fpr, tpr, thresholds = roc_curve(y_true, [-d for d in distances])
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = abs(thresholds[optimal_idx])

            # AUC hesapla
            roc_auc = auc(fpr, tpr)

            print(f"✅ ROC üzerinden hesaplanan yeni threshold: {optimal_threshold:.4f}")
            print(f"📊 AUC: {roc_auc:.4f}")
            print(f"📈 TPR: {tpr[optimal_idx]:.4f}, FPR: {fpr[optimal_idx]:.4f}")

            # Threshold'u sınırla
            self.optimal_threshold = max(0.3, min(0.8, optimal_threshold))
            self.auto_threshold_enabled = True  # ROC ile belirlenen threshold'lar otomatik threshold olarak kabul edilir

            # Threshold geçmişine ekle
            self.threshold_history.append({
                'timestamp': datetime.now().isoformat(),
                'threshold': self.optimal_threshold,
                'method': 'roc_curve',
                'auc': roc_auc,
                'tpr': float(tpr[optimal_idx]),
                'fpr': float(fpr[optimal_idx]),
                'total_pairs': len(distances),
                'positive_pairs': sum(y_true),
                'negative_pairs': len(y_true) - sum(y_true)
            })

            # Geçmişi 100 kayıtla sınırla
            if len(self.threshold_history) > 100:
                self.threshold_history = self.threshold_history[-100:]

            return self.optimal_threshold

        except Exception as e:
            print(f"ROC tabanlı threshold hesaplama hatası: {e}")
            return self.optimal_threshold

    def compare_embeddings(self, emb1, emb2, threshold=None):
        """
        İki embedding arasındaki mesafeyi hesaplar
        """
        if threshold is None:
            threshold = self.get_adaptive_threshold()

        try:
            dist = np.linalg.norm(emb1 - emb2)
            return dist < threshold
        except Exception as e:
            print(f"Embedding karşılaştırma hatası: {e}")
            return False

    def compare_embeddings_batch(self, query_embedding, known_embeddings, threshold=None):
        """
        Tek seferde tüm bilinen yüzlerle karşılaştırma yapar (daha hızlı)
        """
        if threshold is None:
            threshold = self.get_adaptive_threshold()

        try:
            if len(known_embeddings) == 0:
                return -1, float('inf')

            # Tüm mesafeleri hesapla
            distances = np.linalg.norm(known_embeddings - query_embedding, axis=1)

            # En yakın eşleşmeyi bul
            min_idx = np.argmin(distances)
            min_distance = float(distances[min_idx])  # float32'yi float'a çevir

            return min_idx, min_distance

        except Exception as e:
            print(f"Batch embedding karşılaştırma hatası: {e}")
            return -1, float('inf')

    def detect_faces(self, image_array):
        """
        MTCNN kullanarak yüz tespiti yapar
        """
        try:
            # BGR'den RGB'ye çevir (MTCNN RGB bekler)
            rgb_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

            # MTCNN ile yüz tespiti
            faces = self.mtcnn_detector.detect_faces(rgb_image)

            # MTCNN formatını OpenCV formatına çevir
            opencv_faces = []
            for face in faces:
                x, y, w, h = face['box']
                confidence = face['confidence']

                # Güven skoru kontrolü
                if confidence > 1.0:  # Yüksek güven skoru
                    opencv_faces.append((x, y, w, h))

            return np.array(opencv_faces)
        except Exception as e:
            print(f"MTCNN yüz tespit hatası: {e}")
            return np.array([])

    def detect_faces_with_rotation(self, image_array):
        """
        MTCNN ile görüntüyü farklı açılarda döndürerek yüz tespit etmeye çalışır
        """
        # Orijinal görüntüde yüz tespit etmeyi dene
        faces = self.detect_faces(image_array)
        if len(faces) > 0:
            return faces, 0  # 0 derece (orijinal)

        # 90 derece döndür
        rotated_90 = cv2.rotate(image_array, cv2.ROTATE_90_CLOCKWISE)
        faces = self.detect_faces(rotated_90)
        if len(faces) > 0:
            return faces, 90

        # 180 derece döndür
        rotated_180 = cv2.rotate(image_array, cv2.ROTATE_180)
        faces = self.detect_faces(rotated_180)
        if len(faces) > 0:
            return faces, 180

        # 270 derece döndür
        rotated_270 = cv2.rotate(image_array, cv2.ROTATE_90_COUNTERCLOCKWISE)
        faces = self.detect_faces(rotated_270)
        if len(faces) > 0:
            return faces, 270

        return [], 0  # Hiçbir açıda yüz bulunamadı

    def get_face_with_rotation(self, image_array, faces, rotation_angle):
        """
        MTCNN ile döndürülmüş görüntüden yüzü çıkarır ve hizalar
        """
        if len(faces) == 0:
            return None

        # İlk yüzü al
        (x, y, w, h) = faces[0]

        # Döndürülmüş görüntüden yüzü kes
        face_img = image_array[y:y+h, x:x+w]

        # Yüzü orijinal yöne döndür
        if rotation_angle == 90:
            face_img = cv2.rotate(face_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif rotation_angle == 180:
            face_img = cv2.rotate(face_img, cv2.ROTATE_180)
        elif rotation_angle == 270:
            face_img = cv2.rotate(face_img, cv2.ROTATE_90_CLOCKWISE)

        return face_img

    def detect_faces_with_landmarks(self, image_array):
        """
        MTCNN ile yüz tespiti ve landmark'ları alır - geliştirilmiş versiyon
        """
        try:
            # BGR'den RGB'ye çevir (MTCNN RGB bekler)
            rgb_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

            # MTCNN ile yüz tespiti ve landmark'lar
            faces = self.mtcnn_detector.detect_faces(rgb_image)

            print(f"MTCNN ile {len(faces)} yüz tespit edildi")

            # Eğer MTCNN yüz bulamazsa, Haar Cascade dene
            if len(faces) == 0:
                print("MTCNN yüz bulamadı, Haar Cascade deneniyor...")
                faces = self.detect_faces_with_haar_cascade(image_array)

                # Haar Cascade sonuçlarını MTCNN formatına çevir
                converted_faces = []
                for face in faces:
                    x, y, w, h = face
                    # Basit landmark'lar oluştur
                    landmarks = {
                        'left_eye': (x + w//4, y + h//3),
                        'right_eye': (x + 3*w//4, y + h//3),
                        'nose': (x + w//2, y + h//2),
                        'mouth_left': (x + w//4, y + 2*h//3),
                        'mouth_right': (x + 3*w//4, y + 2*h//3)
                    }
                    converted_faces.append({
                        'box': [x, y, w, h],
                        'keypoints': landmarks,
                        'confidence': 0.8
                    })
                faces = converted_faces
                print(f"Haar Cascade ile {len(faces)} yüz tespit edildi")

            return faces
        except Exception as e:
            print(f"Yüz tespit hatası: {e}")
            return []

    def detect_faces_with_haar_cascade(self, image_array):
        """
        Haar Cascade ile yüz tespiti (MTCNN başarısız olduğunda)
        """
        try:
            # Gri tonlamaya çevir
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

            # Haar Cascade yüz tespit edici
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

            # Yüz tespiti
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            print(f"Haar Cascade ile {len(faces)} yüz tespit edildi")
            return faces

        except Exception as e:
            print(f"Haar Cascade hatası: {e}")
            return []

    def align_face(self, image_array, landmarks):
        """
        Yüz landmark'larını kullanarak yüzü hizalar - siyah kenarlar olmadan
        """
        try:
            # MTCNN landmark formatını kontrol et
            if isinstance(landmarks, dict):
                # Göz noktalarını al
                left_eye = landmarks.get('left_eye', landmarks.get('leftEye'))
                right_eye = landmarks.get('right_eye', landmarks.get('rightEye'))

                if left_eye is None or right_eye is None:
                    print("Göz landmark'ları bulunamadı, orijinal görüntü döndürülüyor")
                    return image_array

                # Gözler arası açıyı hesapla
                eye_angle = np.degrees(np.arctan2(
                    right_eye[1] - left_eye[1],
                    right_eye[0] - left_eye[0]
                ))

                # Çok küçük açılar için hizalama yapma (5 dereceden az)
                if abs(eye_angle) < 5:
                    print("Açı çok küçük, hizalama yapılmıyor")
                    return image_array

                # Gözler arası merkez nokta
                eye_center = (
                    int((left_eye[0] + right_eye[0]) / 2),
                    int((left_eye[1] + right_eye[1]) / 2)
                )

                # Görüntüyü genişlet (siyah kenarları önlemek için)
                height, width = image_array.shape[:2]
                diagonal = int(np.sqrt(width**2 + height**2))

                # Yeni boyutlar
                new_width = diagonal
                new_height = diagonal

                # Yeni görüntü oluştur
                new_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

                # Eski görüntüyü yeni görüntünün merkezine yerleştir
                x_offset = (new_width - width) // 2
                y_offset = (new_height - height) // 2
                new_image[y_offset:y_offset+height, x_offset:x_offset+width] = image_array

                # Yeni merkez nokta
                new_center = (x_offset + eye_center[0], y_offset + eye_center[1])

                # Rotasyon matrisi oluştur
                rotation_matrix = cv2.getRotationMatrix2D(new_center, eye_angle, 1.0)

                # Görüntüyü döndür
                aligned_face = cv2.warpAffine(new_image, rotation_matrix, (new_width, new_height))

                # Siyah kenarları kaldır
                aligned_face = self.remove_black_borders(aligned_face)

                return aligned_face
            else:
                print("Landmark formatı tanınmadı, orijinal görüntü döndürülüyor")
                return image_array

        except Exception as e:
            print(f"Yüz hizalama hatası: {e}")
            return image_array

    def remove_black_borders(self, image):
        """
        Görüntüdeki siyah kenarları kaldırır - daha agresif yaklaşım
        """
        try:
            # Gri tonlamaya çevir
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # Çok düşük eşik değeri kullan (çok agresif)
            _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

            # Morfolojik işlemler uygula
            kernel = np.ones((5,5), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

            # Konturları bul
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # En büyük konturu al
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)

                # Minimum boyut kontrolü
                min_size = 30
                if w < min_size or h < min_size:
                    return image

                # Margin ekle (çok az margin)
                margin = 2
                x = max(0, x - margin)
                y = max(0, y - margin)
                w = min(image.shape[1] - x, w + 2 * margin)
                h = min(image.shape[0] - y, h + 2 * margin)

                # Kırp
                cropped = image[y:y+h, x:x+w]

                # Kırpılan görüntünün boyutunu kontrol et
                if cropped.shape[0] > 0 and cropped.shape[1] > 0:
                    return cropped

            return image
        except Exception as e:
            print(f"Siyah kenar kaldırma hatası: {e}")
            return image

    def process_image_with_rotation(self, image_data_base64):
        """
        Base64 görüntüyü alır, rotasyonunu düzeltir ve numpy array'e çevirir
        """
        try:
            # Base64'ten PIL Image'e çevir
            image_data = base64.b64decode(image_data_base64.split(',')[1] if ',' in image_data_base64 else image_data_base64)
            image = Image.open(io.BytesIO(image_data))

            # Görüntü modunu kontrol et ve düzelt
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Otomatik rotasyon uygula
            image = auto_rotate_image(image)

            # PIL Image'i numpy array'e çevir
            image_array = np.array(image)

            # RGB'den BGR'ye çevir (OpenCV için)
            if len(image_array.shape) == 3:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

            # Görüntü boyutunu kontrol et ve logla
            height, width = image_array.shape[:2]
            print(f"İşlenen görüntü boyutu: {width}x{height}")

            return image_array
        except Exception as e:
            print(f"Görüntü işleme hatası: {e}")
            return None

    def load_known_faces(self):
        self.known_face_encodings.clear()
        self.known_face_names.clear()
        self.known_face_ids.clear()

        if os.path.exists(USERS_DB_FILE):
            try:
                with open(USERS_DB_FILE, 'r', encoding='utf-8') as f:
                    users_data = json.load(f)
                print(f"load_known_faces: {len(users_data)} kullanıcı yüklendi")
            except json.JSONDecodeError as e:
                print(f"load_known_faces JSON hatası: {e}")
                print("JSON dosyası düzeltiliyor...")
                if fix_corrupted_json():
                    try:
                        with open(USERS_DB_FILE, 'r', encoding='utf-8') as f:
                            users_data = json.load(f)
                        print(f"Düzeltilmiş dosyadan {len(users_data)} kullanıcı yüklendi")
                    except Exception as e2:
                        print(f"Düzeltilmiş dosya okuma hatası: {e2}")
                        users_data = []
                else:
                    print("JSON düzeltme başarısız")
                    users_data = []
            except Exception as e:
                print(f"load_known_faces genel hatası: {e}")
                users_data = []
        else:
            print("users_db.json dosyası bulunamadı")
            users_data = []

        for user in users_data:
            user_dir = os.path.join(KNOWN_FACES_DIR, user['id_no'])
            encodings_file = os.path.join(user_dir, 'encodings.pkl')
            if os.path.exists(encodings_file):
                try:
                    with open(encodings_file, 'rb') as ef:
                        encodings = pickle.load(ef)
                        for enc in encodings:
                            self.known_face_encodings.append(enc)
                            self.known_face_names.append(user['name'])
                            self.known_face_ids.append(user['id_no'])
                    print(f"Kullanıcı {user['name']} yüklendi: {len(encodings)} encoding")
                except Exception as e:
                    print(f"Encoding yükleme hatası ({user['name']}): {e}")

        print(f"Toplam yüklenen encoding: {len(self.known_face_encodings)}")

        # Cache'i güncelle
        self.update_cache()

    def add_user(self, name, images_base64, id_no=None, birth_date=None):
        try:
            print(f"Add user başlatıldı: {name}, {id_no}")

            # id_no 11 hane kontrolü
            if not id_no or not str(id_no).isdigit() or len(str(id_no)) != 11:
                return False, "Kimlik numarası 11 haneli olmalıdır."

            # Mevcut kullanıcı kontrolü
            if os.path.exists(USERS_DB_FILE):
                with open(USERS_DB_FILE, 'r', encoding='utf-8') as f:
                    users_data = json.load(f)
                if any(user['id_no'] == id_no for user in users_data):
                    return False, "Bu kimlik numarasına sahip kullanıcı zaten kayıtlı."
            else:
                users_data = []

            user_id = str(id_no)
            user_dir = os.path.join(KNOWN_FACES_DIR, user_id)

            # Kullanıcı klasörünü oluştur
            try:
                os.makedirs(user_dir, exist_ok=True)
                print(f"Kullanıcı klasörü oluşturuldu: {user_dir}")
            except Exception as e:
                print(f"Klasör oluşturma hatası: {e}")
                return False, f"Klasör oluşturulamadı: {e}"

            encodings = []
            saved_photos = 0

            for idx, img_b64 in enumerate(images_base64):
                print(f"Resim {idx+1} işleniyor...")

                # Rotasyon düzeltmesi ile görüntüyü işle
                image_array = self.process_image_with_rotation(img_b64)
                if image_array is None:
                    print(f"Resim {idx+1} işlenemedi")
                    continue

                # MTCNN ile yüz tespiti ve landmark'lar - sadece orijinal görüntüde
                faces_with_landmarks = self.detect_faces_with_landmarks(image_array)
                rotation_angle = 0

                # Sadece orijinal görüntüde yüz tespiti yap, başka rotasyonlar deneme
                if len(faces_with_landmarks) == 0:
                    print(f"Resim {idx+1} için orijinal görüntüde yüz bulunamadı")
                    continue

                # İlk yüzü al
                face_data = faces_with_landmarks[0]
                x, y, w, h = face_data['box']
                landmarks = face_data['keypoints']

                # Yüzü kes (daha az margin - daha kesin yüz alanı)
                margin = int(min(w, h) * 0.15)  # %15 margin (daha az margin)
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(image_array.shape[1], x + w + margin)
                y2 = min(image_array.shape[0], y + h + margin)

                face_img = image_array[y1:y2, x1:x2]

                # Siyah kenarları kaldır
                face_img = self.remove_black_borders(face_img)

                # Landmark'ları yeni koordinatlara göre ayarla
                adjusted_landmarks = {}
                for key, point in landmarks.items():
                    adjusted_landmarks[key] = (point[0] - x1, point[1] - y1)

                # Yüz hizalama uygula
                aligned_face = self.align_face(face_img, adjusted_landmarks)

                # Siyah kenarları kaldır
                aligned_face = self.remove_black_borders(aligned_face)

                # Ana embedding'i oluştur
                main_embedding = self.get_embedding(aligned_face)
                if main_embedding is None:
                    print(f"Resim {idx+1} için embedding oluşturulamadı - yüz çözünürlüğü yetersiz")
                    continue

                encodings.append(main_embedding)

                # Orijinal yüzü kaydet (orijinal yönde)
                try:
                    face_rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
                    face_pil = Image.fromarray(face_rgb)
                    original_photo_path = os.path.join(user_dir, f'{idx+1}_original.jpg')
                    face_pil.save(original_photo_path)
                    saved_photos += 1
                    print(f"Orijinal fotoğraf kaydedildi: {original_photo_path}")
                except Exception as e:
                    print(f"Orijinal fotoğraf kaydetme hatası: {e}")

                # Augmentation uygula - orijinal yüz için
                try:
                    # Orijinal yüzü PIL Image'e çevir
                    face_rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
                    face_pil = Image.fromarray(face_rgb)

                    # Augmented versiyonlar oluştur
                    augmented_versions = create_augmented_versions(face_pil, num_versions=4)

                    # Augmented versiyonları kaydet ve embedding'lerini oluştur
                    for aug_idx, aug_img in enumerate(augmented_versions):
                        try:
                            # Augmented embedding oluştur
                            aug_array = np.array(aug_img)
                            aug_bgr = cv2.cvtColor(aug_array, cv2.COLOR_RGB2BGR)
                            aug_embedding = self.get_embedding(aug_bgr)

                            if aug_embedding is not None:
                                encodings.append(aug_embedding)

                            # Augmented fotoğrafı kaydet
                            aug_path = os.path.join(user_dir, f'{idx+1}_aug{aug_idx+1}.jpg')
                            aug_img.save(aug_path)
                            saved_photos += 1
                            print(f"Augmented fotoğraf {aug_idx+1} kaydedildi: {aug_path}")

                        except Exception as e:
                            print(f"Augmented fotoğraf {aug_idx+1} kaydetme hatası: {e}")
                            continue

                except Exception as e:
                    print(f"Augmentation hatası: {e}")
                    continue

            if not encodings:
                return False, "Hiçbir resimde yüz tespit edilemedi"

            # Embeddings'i kaydet
            try:
                encodings_path = os.path.join(user_dir, 'encodings.pkl')
                with open(encodings_path, 'wb') as ef:
                    pickle.dump(encodings, ef)
                print(f"Embeddings kaydedildi: {encodings_path}")
            except Exception as e:
                print(f"Embeddings kaydetme hatası: {e}")
                return False, f"Embeddings kaydedilemedi: {e}"

            # JSON'a kullanıcı bilgilerini kaydet
            try:
                self.save_users_db(user_id, name, id_no, birth_date)
                print(f"Kullanıcı JSON'a kaydedildi: {user_id}")
            except Exception as e:
                print(f"JSON kaydetme hatası: {e}")
                return False, f"Kullanıcı bilgileri kaydedilemedi: {e}"

            # Memory'ye ekle
            for enc in encodings:
                self.known_face_encodings.append(enc)
                self.known_face_names.append(name)
                self.known_face_ids.append(user_id)

            # Cache'i güncelle
            try:
                self.update_cache()
                print("Cache güncellendi")
            except Exception as e:
                print(f"Cache güncelleme hatası: {e}")

            log_recognition(user_id=user_id, name=name, result='success', image_b64=None, action='kayıt')
            print(f"Kullanıcı başarıyla eklendi: {name}, {saved_photos} fotoğraf kaydedildi")
            return True, f"Kullanıcı {name} başarıyla eklendi ({saved_photos} fotoğraf)"

        except Exception as e:
            print(f"Add user genel hatası: {e}")
            return False, f"Hata: {str(e)}"

    def save_users_db(self, user_id, name, id_no=None, birth_date=None):
        try:
            print(f"save_users_db başlatıldı: {user_id}, {name}")

            users_data = []
            if os.path.exists(USERS_DB_FILE):
                try:
                    with open(USERS_DB_FILE, 'r', encoding='utf-8') as f:
                        users_data = json.load(f)
                    print(f"Mevcut kullanıcı sayısı: {len(users_data)}")
                except json.JSONDecodeError as e:
                    print(f"JSON okuma hatası (extra data): {e}")
                    print("JSON dosyası düzeltiliyor...")
                    if fix_corrupted_json():
                        # Düzeltilmiş dosyayı tekrar oku
                        try:
                            with open(USERS_DB_FILE, 'r', encoding='utf-8') as f:
                                users_data = json.load(f)
                            print(f"Düzeltilmiş dosyadan okunan kullanıcı sayısı: {len(users_data)}")
                        except Exception as e2:
                            print(f"Düzeltilmiş dosya okuma hatası: {e2}")
                            users_data = []
                    else:
                        print("JSON düzeltme başarısız, yeni liste oluşturuluyor")
                        users_data = []
                except Exception as e:
                    print(f"JSON okuma hatası: {e}")
                    users_data = []

            # Yeni kullanıcı bilgilerini ekle
            new_user = {
                'id': user_id,
                'name': name,
                'id_no': id_no,
                'birth_date': birth_date,
                'created_at': datetime.now().isoformat()
            }
            users_data.append(new_user)

            print(f"Yeni kullanıcı eklendi: {new_user}")
            print(f"Toplam kullanıcı sayısı: {len(users_data)}")

            # JSON dosyasına kaydet
            try:
                with open(USERS_DB_FILE, 'w', encoding='utf-8') as f:
                    json.dump(users_data, f, ensure_ascii=False, indent=2)
                print(f"JSON dosyasına başarıyla kaydedildi: {USERS_DB_FILE}")
            except Exception as e:
                print(f"JSON yazma hatası: {e}")
                raise e

        except Exception as e:
            print(f"save_users_db genel hatası: {e}")
            raise e

    def recognize_face(self, face_image_base64):
        try:
            # Rotasyon düzeltmesi ile görüntüyü işle
            image_array = self.process_image_with_rotation(face_image_base64)
            if image_array is None:
                return False, "Görüntü işlenemedi"

            # MTCNN ile yüz tespiti ve landmark'lar - sadece orijinal görüntüde
            faces_with_landmarks = self.detect_faces_with_landmarks(image_array)
            rotation_angle = 0

            # Sadece orijinal görüntüde yüz tespiti yap, başka rotasyonlar deneme
            if len(faces_with_landmarks) == 0:
                print("Orijinal görüntüde yüz bulunamadı")
                return False, "Yüz tespit edilemedi"

            if len(faces_with_landmarks) == 0:
                return False, "Yüz tespit edilemedi"

            # Cache'den users data'yı al
            if self.users_data_cache is None:
                with open(USERS_DB_FILE, 'r', encoding='utf-8') as f:
                    self.users_data_cache = json.load(f)

            # İlk yüzü al
            face_data = faces_with_landmarks[0]
            x, y, w, h = face_data['box']
            landmarks = face_data['keypoints']

            # Yüzü kes (daha az margin - daha kesin yüz alanı)
            margin = int(min(w, h) * 0.15)  # %15 margin (daha az margin)
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(image_array.shape[1], x + w + margin)
            y2 = min(image_array.shape[0], y + h + margin)

            face_img = image_array[y1:y2, x1:x2]

            # Siyah kenarları kaldır
            face_img = self.remove_black_borders(face_img)

            # Landmark'ları yeni koordinatlara göre ayarla
            adjusted_landmarks = {}
            for key, point in landmarks.items():
                adjusted_landmarks[key] = (point[0] - x1, point[1] - y1)

            # Yüz hizalama uygula
            aligned_face = self.align_face(face_img, adjusted_landmarks)

            # Siyah kenarları kaldır
            aligned_face = self.remove_black_borders(aligned_face)

            # Ana embedding'i oluştur
            main_embedding = self.get_embedding(aligned_face)
            if main_embedding is None:
                return False, "Yüz çözünürlüğü yetersiz veya embedding oluşturulamadı"

            # Otomatik threshold hesaplama
            threshold_calculation_details = {
                'auto_threshold_enabled': self.auto_threshold_enabled,
                'test_embeddings_count': 1,
                'rotations_tried': [],
                'optimal_threshold': None,
                'calculation_method': 'adaptive'
            }
            
            if self.auto_threshold_enabled and len(self.known_face_encodings) > 0:
                # Test embedding'lerini hazırla
                test_embeddings = [main_embedding]

                # Farklı rotasyonlarda da deneme yap
                rotations = [90, 180, 270]
                for rotation in rotations:
                    try:
                        if rotation == 90:
                            rotated_face = cv2.rotate(aligned_face, cv2.ROTATE_90_CLOCKWISE)
                        elif rotation == 180:
                            rotated_face = cv2.rotate(aligned_face, cv2.ROTATE_180)
                        elif rotation == 270:
                            rotated_face = cv2.rotate(aligned_face, cv2.ROTATE_90_COUNTERCLOCKWISE)

                        rotated_embedding = self.get_embedding(rotated_face)
                        if rotated_embedding is not None:
                            test_embeddings.append(rotated_embedding)
                            threshold_calculation_details['rotations_tried'].append(rotation)
                    except Exception as e:
                        print(f"Rotasyon {rotation}° embedding hatası: {e}")
                        continue

                threshold_calculation_details['test_embeddings_count'] = len(test_embeddings)
                
                # Optimal threshold hesapla
                optimal_threshold = self.calculate_optimal_threshold(test_embeddings, self.known_embeddings_array)
                threshold_calculation_details['optimal_threshold'] = optimal_threshold
                threshold_calculation_details['calculation_method'] = 'optimal'
                print(f"Kullanılan threshold: {optimal_threshold:.4f}")
            else:
                optimal_threshold = self.get_adaptive_threshold()
                threshold_calculation_details['optimal_threshold'] = optimal_threshold
                threshold_calculation_details['calculation_method'] = 'adaptive'

            # Tüm embedding'ler için en iyi eşleşmeyi bul
            best_match_idx = -1
            min_distance = float('inf')
            match_confidence = 0.0

            # Ana embedding ile test et
            if self.known_embeddings_array is not None and len(self.known_embeddings_array) > 0:
                current_match_idx, current_distance = self.compare_embeddings_batch(
                    main_embedding, self.known_embeddings_array, threshold=optimal_threshold
                )

                if current_match_idx != -1 and current_distance < min_distance:
                    best_match_idx = current_match_idx
                    min_distance = current_distance
                    # Güven skorunu hesapla
                    match_confidence = max(0, 1 - (current_distance / optimal_threshold))

            # En iyi eşleşmeyi kontrol et
            if best_match_idx != -1 and min_distance < optimal_threshold:
                user_id = self.known_face_ids[best_match_idx]
                user_info = next((u for u in self.users_data_cache if u['id'] == user_id), None)

                if user_info is None:
                    return False, "Kullanıcı bilgisi bulunamadı"

                # Threshold log kaydet (başarılı tanıma)
                log_threshold_event(
                    user_id=user_id,
                    name=user_info['name'],
                    threshold_used=optimal_threshold,
                    distance=min_distance,
                    confidence=match_confidence,
                    success=True,
                    image_b64=face_image_base64,
                    threshold_details=threshold_calculation_details
                )

                # Eğer bu kişi zaten tanınmışsa, tekrar tanıma
                if user_id in self.recognized_persons:
                    return False, user_info

                # Tanınan kişiyi set'e ekle
                self.recognized_persons.add(user_id)

                # Tanınan kişinin fotoğrafını kaydet ve detayları logla
                photo_details = self.save_recognition_photo(user_id, image_array, aligned_face)
                
                # Fotoğraf detaylarını log'a ekle
                recognition_details = {
                    'user_id': user_id,
                    'name': user_info['name'],
                    'id_no': user_info.get('id_no'),
                    'birth_date': user_info.get('birth_date'),
                    'timestamp': datetime.now().isoformat(),
                    'image_base64': face_image_base64,
                    'confidence': match_confidence,
                    'distance': min_distance,
                    'threshold_details': threshold_calculation_details,
                    'photo_details': photo_details,
                    'face_detection': {
                        'face_box': {'x': x, 'y': y, 'w': w, 'h': h},
                        'face_crop': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
                        'landmarks': landmarks,
                        'aligned_face_size': aligned_face.shape if aligned_face is not None else None
                    },
                    'recognition_log': {
                        'captured_full_image': True,
                        'captured_face_crop': True,
                        'full_image_description': 'Orijinal çekilen fotoğrafın tamamı',
                        'face_crop_description': 'Sadece yüz kısmı, hizalanmış ve kırpılmış',
                        'log_type': 'recognition_success'
                    }
                }

                # Gerçek zamanlı tanıma kaydı ekle (sadece bir kez ekle)
                if not any(log['user_id'] == user_id for log in self.realtime_recognition_logs):
                    recognition_record = {
                        'user_id': user_id,
                        'name': user_info['name'],
                        'id_no': user_info.get('id_no'),
                        'birth_date': user_info.get('birth_date'),
                        'timestamp': datetime.now().isoformat(),
                        'image_base64': face_image_base64,
                        'confidence': match_confidence,
                        'distance': min_distance,
                        'threshold_details': threshold_calculation_details,
                        'photo_details': photo_details,
                        'recognition_log': {
                            'captured_full_image': True,
                            'captured_face_crop': True,
                            'full_image_description': 'Orijinal çekilen fotoğrafın tamamı',
                            'face_crop_description': 'Sadece yüz kısmı, hizalanmış ve kırpılmış',
                            'log_type': 'realtime_recognition'
                        }
                    }
                    self.realtime_recognition_logs.append(recognition_record)

                # Detaylı log mesajları
                print(f"✅ Tanıma başarılı: {user_info['name']} (ID: {user_id})")
                if photo_details:
                    print(f"📸 Fotoğraf detayları:")
                    print(f"  - Tam resim: {photo_details.get('image_info', {}).get('full_image_dimensions', 'N/A')} ({photo_details.get('file_sizes', {}).get('full_image_bytes', 0)} bytes)")
                    print(f"  - Yüz kırpma: {photo_details.get('image_info', {}).get('face_crop_dimensions', 'N/A')} ({photo_details.get('file_sizes', {}).get('face_crop_bytes', 0)} bytes)")
                    print(f"  - Kayıt yolu: {photo_details.get('timestamp_dir', 'N/A')}")

                log_recognition(user_id, user_info['name'], 'success', face_image_base64, action='tanıma')
                return True, user_info

            # Threshold log kaydet (başarısız tanıma)
            if best_match_idx != -1:
                # En yakın eşleşme var ama threshold'u geçmedi
                closest_user_id = self.known_face_ids[best_match_idx]
                closest_user_info = next((u for u in self.users_data_cache if u['id'] == closest_user_id), None)
                closest_name = closest_user_info['name'] if closest_user_info else "Bilinmeyen"

                log_threshold_event(
                    user_id=closest_user_id,
                    name=closest_name,
                    threshold_used=optimal_threshold,
                    distance=min_distance,
                    confidence=match_confidence,
                    success=False,
                    image_b64=face_image_base64,
                    threshold_details=threshold_calculation_details
                )
            else:
                # Hiç eşleşme bulunamadı
                log_threshold_event(
                    user_id="unknown",
                    name="Tanınmayan Kişi",
                    threshold_used=optimal_threshold,
                    distance=float('inf'),
                    confidence=0.0,
                    success=False,
                    image_b64=face_image_base64,
                    threshold_details=threshold_calculation_details
                )

            log_recognition(None, None, 'fail', face_image_base64, action='tanıma')
            return False, "Tanınmayan kişi"
        except Exception as e:
            return False, f"Hata: {str(e)}"

    def reset_recognition_session(self):
        """Yeni tanıma oturumu başlatmak için tanınan kişileri temizle"""
        self.recognized_persons.clear()
        self.realtime_recognition_logs.clear()
        return True, "Tanıma oturumu sıfırlandı"

    def get_realtime_recognition_logs(self):
        """Gerçek zamanlı tanıma kayıtlarını döndür"""
        return self.realtime_recognition_logs

    def save_recognition_photo(self, user_id, full_image, face_img):
        """Tanınan kişinin fotoğrafını zaman damgası ile kaydet"""
        try:
            # Zaman damgası oluştur
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Kullanıcının recognition_logs klasörünü oluştur
            user_dir = os.path.join(KNOWN_FACES_DIR, user_id)
            recognition_logs_dir = os.path.join(user_dir, 'recognition_logs')
            timestamp_dir = os.path.join(recognition_logs_dir, timestamp)

            os.makedirs(timestamp_dir, exist_ok=True)

            # Tam resmi kaydet (orijinal çekilen fotoğraf)
            full_image_pil = Image.fromarray(full_image)
            full_image_path = os.path.join(timestamp_dir, 'full_image.jpg')
            full_image_pil.save(full_image_path, quality=95)

            # Yüz kısmını kaydet (hizalanmış yüz) - sadece embedding için
            face_pil = Image.fromarray(face_img)
            face_image_path = os.path.join(timestamp_dir, 'face_crop.jpg')
            face_pil.save(face_image_path, quality=95)

            # Tam resmin base64'ünü kaydet (görüntüleme için)
            full_img_bytes = cv2.imencode('.jpg', full_image)[1].tobytes()
            full_base64 = base64.b64encode(full_img_bytes).decode('utf-8')

            # Yüz kısmının base64'ünü kaydet
            face_img_bytes = cv2.imencode('.jpg', face_img)[1].tobytes()
            face_base64 = base64.b64encode(face_img_bytes).decode('utf-8')

            # Dosya boyutlarını al
            full_image_size = os.path.getsize(full_image_path)
            face_crop_size = os.path.getsize(face_image_path)

            # Tanıma bilgilerini JSON dosyasına kaydet
            recognition_info = {
                'timestamp': datetime.now().isoformat(),
                'user_id': user_id,
                'files': {
                    'full_image': 'full_image.jpg',
                    'face_crop': 'face_crop.jpg',
                    'full_base64': full_base64,  # Orijinal fotoğrafın base64'ü
                    'face_base64': face_base64   # Yüz kısmının base64'ü
                },
                'photo_details': {
                    'full_image_size': full_image.shape,
                    'face_crop_size': face_img.shape,
                    'full_image_path': full_image_path,
                    'face_crop_path': face_image_path,
                    'timestamp_dir': timestamp_dir,
                    'file_sizes': {
                        'full_image_bytes': full_image_size,
                        'face_crop_bytes': face_crop_size
                    },
                    'image_info': {
                        'full_image_dimensions': f"{full_image.shape[1]}x{full_image.shape[0]}",
                        'face_crop_dimensions': f"{face_img.shape[1]}x{face_img.shape[0]}",
                        'full_image_channels': full_image.shape[2] if len(full_image.shape) > 2 else 1,
                        'face_crop_channels': face_img.shape[2] if len(face_img.shape) > 2 else 1
                    }
                },
                'recognition_log': {
                    'captured_full_image': True,
                    'captured_face_crop': True,
                    'full_image_description': 'Orijinal çekilen fotoğrafın tamamı',
                    'face_crop_description': 'Sadece yüz kısmı, hizalanmış ve kırpılmış',
                    'saved_at': timestamp,
                    'log_type': 'recognition_photo'
                }
            }

            info_path = os.path.join(timestamp_dir, 'recognition_info.json')
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(recognition_info, f, ensure_ascii=False, indent=2)

            print(f"Tanıma fotoğrafı kaydedildi: {timestamp_dir}")
            print(f"  - Tam resim: {full_image.shape[1]}x{full_image.shape[0]} ({full_image_size} bytes)")
            print(f"  - Yüz kırpma: {face_img.shape[1]}x{face_img.shape[0]} ({face_crop_size} bytes)")

            # Fotoğraf detaylarını döndür
            return {
                'timestamp': timestamp,
                'timestamp_dir': timestamp_dir,
                'full_image_path': full_image_path,
                'face_crop_path': face_image_path,
                'full_image_size': full_image.shape,
                'face_crop_size': face_img.shape,
                'full_base64': full_base64,
                'face_base64': face_base64,
                'recognition_info_path': info_path,
                'file_sizes': {
                    'full_image_bytes': full_image_size,
                    'face_crop_bytes': face_crop_size
                },
                'image_info': {
                    'full_image_dimensions': f"{full_image.shape[1]}x{full_image.shape[0]}",
                    'face_crop_dimensions': f"{face_img.shape[1]}x{face_img.shape[0]}"
                }
            }

        except Exception as e:
            print(f"Fotoğraf kaydetme hatası: {str(e)}")
            return None

    def delete_user(self, user_id):
        try:
            print(f"🚀 delete_user başlatıldı: user_id={user_id}")
            print(f"📂 KNOWN_FACES_DIR: {KNOWN_FACES_DIR}")
            
            if not os.path.exists(USERS_DB_FILE):
                return False, "Kullanıcı bulunamadı"
            
            with open(USERS_DB_FILE, 'r', encoding='utf-8') as f:
                users_data = json.load(f)
            
            # Kullanıcıyı bul (id veya id_no ile)
            user_index = None
            user_name = None
            user_id_no = None
            for i, user in enumerate(users_data):
                if user.get('id') == user_id or user.get('id_no') == user_id:
                    user_index = i
                    user_name = user.get('name', 'Bilinmeyen')
                    user_id_no = user.get('id_no') or user.get('id')
                    break
            
            if user_index is None:
                return False, "Kullanıcı bulunamadı"
            
            print(f"🔍 Kullanıcı bulundu: {user_name} (ID: {user_id}, ID_NO: {user_id_no})")
            
            # Kullanıcıyı JSON'dan sil
            users_data.pop(user_index)
            with open(USERS_DB_FILE, 'w', encoding='utf-8') as f:
                json.dump(users_data, f, ensure_ascii=False, indent=2)

            # 1. Kullanıcının profil fotoğrafları klasörünü sil (kimlik numarası ile oluşturulan klasör)
            user_dirs_to_delete = []
            
            # Kimlik numarası ile oluşturulan klasör (add_user'da user_id = str(id_no) olarak ayarlanıyor)
            if user_id_no:
                user_dir_by_id_no = os.path.join(KNOWN_FACES_DIR, str(user_id_no))
                print(f"🔍 Kontrol ediliyor (id_no): {user_dir_by_id_no}")
                if os.path.exists(user_dir_by_id_no):
                    user_dirs_to_delete.append(user_dir_by_id_no)
                    print(f"📁 Bulunan klasör (id_no): {user_dir_by_id_no}")
                else:
                    print(f"❌ Klasör bulunamadı (id_no): {user_dir_by_id_no}")
            
            # user_id ile oluşturulan klasör (eğer farklıysa)
            user_dir_by_id = os.path.join(KNOWN_FACES_DIR, str(user_id))
            print(f"🔍 Kontrol ediliyor (user_id): {user_dir_by_id}")
            if os.path.exists(user_dir_by_id):
                if user_dir_by_id not in user_dirs_to_delete:  # Tekrar ekleme
                    user_dirs_to_delete.append(user_dir_by_id)
                print(f"📁 Bulunan klasör (user_id): {user_dir_by_id}")
            else:
                print(f"❌ Klasör bulunamadı (user_id): {user_dir_by_id}")
            
            # Tüm klasörleri sil
            deleted_count = 0
            for user_dir in user_dirs_to_delete:
                try:
                    print(f"🗑️ Siliniyor: {user_dir}")
                    # Klasörün içeriğini listele
                    if os.path.exists(user_dir):
                        print(f"📋 Klasör içeriği:")
                        for root, dirs, files in os.walk(user_dir):
                            for file in files:
                                print(f"   📄 {os.path.join(root, file)}")
                    
                    # Windows için güçlü silme mekanizması
                    def remove_readonly(func, path, _):
                        """Dosya izinlerini değiştir ve sil"""
                        try:
                            os.chmod(path, 0o777)
                            func(path)
                        except Exception as e:
                            print(f"⚠️ Dosya izni değiştirme hatası: {e}")
                    
                    # Önce dosyaları tek tek silmeyi dene
                    try:
                        for root, dirs, files in os.walk(user_dir, topdown=False):
                            for file in files:
                                file_path = os.path.join(root, file)
                                try:
                                    os.chmod(file_path, 0o777)
                                    os.remove(file_path)
                                    print(f"   🗑️ Dosya silindi: {file_path}")
                                except Exception as e:
                                    print(f"   ⚠️ Dosya silme hatası ({file_path}): {e}")
                            
                            for dir_name in dirs:
                                dir_path = os.path.join(root, dir_name)
                                try:
                                    os.chmod(dir_path, 0o777)
                                    os.rmdir(dir_path)
                                    print(f"   🗑️ Alt klasör silindi: {dir_path}")
                                except Exception as e:
                                    print(f"   ⚠️ Alt klasör silme hatası ({dir_path}): {e}")
                        
                        # Ana klasörü sil
                        os.chmod(user_dir, 0o777)
                        os.rmdir(user_dir)
                        print(f"✅ Kullanıcı klasörü silindi: {user_dir}")
                        deleted_count += 1
                        
                    except Exception as e:
                        print(f"⚠️ Tek tek silme başarısız, shutil.rmtree deneniyor: {e}")
                        # Eğer tek tek silme başarısız olursa, shutil.rmtree ile dene
                        shutil.rmtree(user_dir, onerror=remove_readonly)
                        print(f"✅ Kullanıcı klasörü silindi (shutil.rmtree): {user_dir}")
                        deleted_count += 1
                        
                except Exception as e:
                    print(f"⚠️ Klasör silme hatası ({user_dir}): {e}")
                    # Son çare: force silme
                    try:
                        import subprocess
                        if os.name == 'nt':  # Windows
                            subprocess.run(['rmdir', '/s', '/q', user_dir], shell=True, check=True)
                            print(f"✅ Kullanıcı klasörü silindi (force): {user_dir}")
                            deleted_count += 1
                        else:  # Unix/Linux
                            subprocess.run(['rm', '-rf', user_dir], check=True)
                            print(f"✅ Kullanıcı klasörü silindi (force): {user_dir}")
                            deleted_count += 1
                    except Exception as force_error:
                        print(f"❌ Force silme de başarısız ({user_dir}): {force_error}")
            
            if deleted_count == 0:
                print(f"⚠️ Hiçbir klasör silinmedi!")
            else:
                print(f"✅ Toplam {deleted_count} klasör silindi")

            # 2. Tanıma loglarından kullanıcıya ait kayıtları sil
            if os.path.exists(RECOGNITION_LOG_FILE):
                try:
                    with open(RECOGNITION_LOG_FILE, 'r', encoding='utf-8') as f:
                        logs = json.load(f)
                    
                    # Kullanıcıya ait logları filtrele (hem user_id hem de id_no ile)
                    filtered_logs = []
                    deleted_logs = 0
                    for log in logs:
                        log_user_id = log.get('user_id')
                        if log_user_id != user_id and log_user_id != user_id_no:
                            filtered_logs.append(log)
                        else:
                            deleted_logs += 1
                    
                    with open(RECOGNITION_LOG_FILE, 'w', encoding='utf-8') as f:
                        json.dump(filtered_logs, f, ensure_ascii=False, indent=2)
                    print(f"📝 Tanıma logları temizlendi: {deleted_logs} kayıt silindi")
                except Exception as e:
                    print(f"⚠️ Tanıma logları temizlenirken hata: {e}")

            # 3. Threshold loglarından kullanıcıya ait kayıtları sil
            threshold_log_file = 'threshold_logs.json'
            if os.path.exists(threshold_log_file):
                try:
                    with open(threshold_log_file, 'r', encoding='utf-8') as f:
                        threshold_logs = json.load(f)
                    
                    # Kullanıcıya ait threshold loglarını filtrele (hem user_id hem de id_no ile)
                    filtered_threshold_logs = []
                    deleted_threshold_logs = 0
                    for log in threshold_logs:
                        log_user_id = log.get('user_id')
                        if log_user_id != user_id and log_user_id != user_id_no:
                            filtered_threshold_logs.append(log)
                        else:
                            deleted_threshold_logs += 1
                    
                    with open(threshold_log_file, 'w', encoding='utf-8') as f:
                        json.dump(filtered_threshold_logs, f, ensure_ascii=False, indent=2)
                    print(f"📊 Threshold logları temizlendi: {deleted_threshold_logs} kayıt silindi")
                except Exception as e:
                    print(f"⚠️ Threshold logları temizlenirken hata: {e}")

            # 4. Cache'i güncelle
            self.load_known_faces()
            
            print(f"✅ Kullanıcı {user_name} (ID: {user_id}, ID_NO: {user_id_no}) başarıyla silindi")
            return True, f"Kullanıcı {user_name} başarıyla silindi"
            
        except Exception as e:
            print(f"❌ Kullanıcı silme hatası: {e}")
            return False, f"Hata: {str(e)}"

    def get_users(self):
        if os.path.exists(USERS_DB_FILE):
            with open(USERS_DB_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

face_api = FaceRecognitionAPI()

def log_recognition(user_id, name, result, image_b64=None, action=None):
    log_entry = {
        'user_id': user_id,
        'name': name,
        'date': datetime.now().isoformat(),
        'result': result,
        'action': action,
        'image': image_b64
    }
    logs = []
    if os.path.exists(RECOGNITION_LOG_FILE):
        with open(RECOGNITION_LOG_FILE, 'r', encoding='utf-8') as f:
            logs = json.load(f)
    logs.append(log_entry)
    with open(RECOGNITION_LOG_FILE, 'w', encoding='utf-8') as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

def log_threshold_event(user_id, name, threshold_used, distance, confidence, success, image_b64=None, threshold_details=None):
    """Threshold olaylarını loglar"""
    # float32 değerlerini float'a çevir
    threshold_used = float(threshold_used) if threshold_used is not None else 0.0
    distance = float(distance) if distance is not None else 0.0
    confidence = float(confidence) if confidence is not None else 0.0

    # Threshold detaylarını genişlet
    if threshold_details is None:
        threshold_details = {}
    
    # Threshold hesaplama detaylarını ekle
    threshold_details.update({
        'calculation_timestamp': datetime.now().isoformat(),
        'threshold_type': 'auto' if threshold_details.get('auto_threshold_enabled', False) else 'manual',
        'test_embeddings_used': threshold_details.get('test_embeddings_count', 1),
        'rotations_tried': threshold_details.get('rotations_tried', []),
        'calculation_method': threshold_details.get('calculation_method', 'adaptive'),
        'optimal_threshold_value': threshold_details.get('optimal_threshold', threshold_used)
    })

    log_entry = {
        'user_id': user_id,
        'name': name,
        'threshold_used': threshold_used,
        'distance': distance,
        'confidence': confidence,
        'success': success,
        'timestamp': datetime.now().isoformat(),
        'image': image_b64,
        'threshold_details': threshold_details,
        'recognition_log': {
            'threshold_calculation': True,
            'threshold_details_captured': True,
            'calculation_method': threshold_details.get('calculation_method', 'adaptive'),
            'test_embeddings_count': threshold_details.get('test_embeddings_count', 1),
            'rotations_tried': threshold_details.get('rotations_tried', []),
            'log_type': 'threshold_event'
        }
    }

    # Threshold log dosyası
    threshold_log_file = 'threshold_logs.json'
    logs = []

    try:
        if os.path.exists(threshold_log_file):
            with open(threshold_log_file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
    except Exception as e:
        print(f"Threshold log okuma hatası: {e}")
        logs = []

    logs.append(log_entry)

    try:
        with open(threshold_log_file, 'w', encoding='utf-8') as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)
        
        # Detaylı log mesajları
        print(f"🔍 Threshold log kaydedildi: {threshold_used:.4f} -> {success}")
        if threshold_details:
            print(f"  - Hesaplama yöntemi: {threshold_details.get('calculation_method', 'adaptive')}")
            print(f"  - Test embedding sayısı: {threshold_details.get('test_embeddings_count', 1)}")
            print(f"  - Denenen rotasyonlar: {threshold_details.get('rotations_tried', [])}")
            print(f"  - Optimal threshold: {threshold_details.get('optimal_threshold', threshold_used):.4f}")
    except Exception as e:
        print(f"Threshold log yazma hatası: {e}")

def analyze_lighting_conditions(image_array):
    """Işık durumunu analiz et"""
    try:
        # Görüntüyü gri tonlamaya çevir
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array
        
        # Histogram analizi
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        
        # Ortalama parlaklık
        mean_brightness = np.mean(gray)
        
        # Standart sapma (kontrast)
        std_brightness = np.std(gray)
        
        # Histogram analizi
        total_pixels = gray.shape[0] * gray.shape[1]
        
        # Karanlık pikseller (0-50)
        dark_pixels = np.sum(hist[0:50])
        dark_ratio = dark_pixels / total_pixels
        
        # Parlak pikseller (200-255)
        bright_pixels = np.sum(hist[200:256])
        bright_ratio = bright_pixels / total_pixels
        
        # Orta ton pikseller (50-200)
        mid_pixels = np.sum(hist[50:200])
        mid_ratio = mid_pixels / total_pixels
        
        # Işık durumunu belirle
        condition = 'good'
        message = 'Işık durumu uygun'
        suggestion = 'Yüzünüzü odaklanma alanına yerleştirin'
        
        # Çok karanlık
        if mean_brightness < 50 or dark_ratio > 0.6:
            condition = 'too_dark'
            message = 'Çok karanlık'
            suggestion = 'Daha aydınlatılmış bir alana geçin veya ışığı artırın'
        
        # Çok parlak
        elif mean_brightness > 200 or bright_ratio > 0.4:
            condition = 'too_bright'
            message = 'Çok fazla ışık var'
            suggestion = 'Daha az aydınlatılmış bir alana geçin veya ışığı azaltın'
        
        # Dengesiz ışık
        elif std_brightness > 80 or (dark_ratio > 0.3 and bright_ratio > 0.2):
            condition = 'uneven'
            message = 'Dengesiz ışık'
            suggestion = 'Yüzünüzü daha eşit aydınlatılmış bir konuma getirin'
        
        # Düşük kontrast
        elif std_brightness < 30:
            condition = 'uneven'
            message = 'Düşük kontrast'
            suggestion = 'Daha iyi aydınlatılmış bir alana geçin'
        
        return {
            'condition': condition,
            'message': message,
            'suggestion': suggestion,
            'metrics': {
                'mean_brightness': float(mean_brightness),
                'std_brightness': float(std_brightness),
                'dark_ratio': float(dark_ratio),
                'bright_ratio': float(bright_ratio),
                'mid_ratio': float(mid_ratio)
            }
        }
        
    except Exception as e:
        print(f"Işık analizi hatası: {e}")
        return {
            'condition': 'unknown',
            'message': 'Işık durumu belirlenemedi',
            'suggestion': 'Kamerayı yeniden konumlandırın',
            'metrics': {}
        }

@app.route('/fix_json', methods=['POST'])
def fix_json_endpoint():
    """JSON dosyasını düzeltmek için endpoint"""
    try:
        success = fix_corrupted_json()
        if success:
            return jsonify({'success': True, 'message': 'JSON dosyası başarıyla düzeltildi'})
        else:
            return jsonify({'success': False, 'message': 'JSON dosyası düzeltilemedi'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Hata: {str(e)}'})

@app.route('/health', methods=['GET'])
def health_check():
    import os
    return jsonify({
        'status': 'OK',
        'message': 'API çalışıyor',
        'timestamp': datetime.now().isoformat(),
        'current_directory': os.getcwd(),
        'known_faces_path': os.path.abspath(KNOWN_FACES_DIR)
    })

@app.route('/reset_recognition_session', methods=['POST'])
def reset_recognition_session():
    """Yeni tanıma oturumu başlatmak için endpoint"""
    try:
        success, message = face_api.reset_recognition_session()

        # Threshold'u her kamera açıldığında güncelle
        new_threshold = face_api.calculate_threshold_via_roc()

        return jsonify({
            'success': success,
            'message': message,
            'new_threshold': new_threshold,
            'threshold_method': 'roc_curve'
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Hata: {str(e)}'}), 500

@app.route('/realtime_recognition_logs', methods=['GET'])
def get_realtime_recognition_logs():
    """Gerçek zamanlı tanıma kayıtlarını döndüren endpoint"""
    try:
        logs = face_api.get_realtime_recognition_logs()
        return jsonify({'success': True, 'logs': logs})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Hata: {str(e)}'}), 500

@app.route('/test_recognition_logs/<id_no>', methods=['GET'])
def test_recognition_logs(id_no):
    """Test endpoint - tanıma loglarını kontrol et"""
    try:
        import os
        current_dir = os.getcwd()
        recognition_logs_dir = os.path.join(KNOWN_FACES_DIR, id_no, 'recognition_logs')
        abs_path = os.path.abspath(recognition_logs_dir)
        exists = os.path.exists(recognition_logs_dir)
        files = []
        subdirs = []
        if exists:
            files = os.listdir(recognition_logs_dir)
            for item in files:
                item_path = os.path.join(recognition_logs_dir, item)
                if os.path.isdir(item_path):
                    subdirs.append({
                        'name': item,
                        'files': os.listdir(item_path)
                    })
        return jsonify({
            'success': True,
            'id_no': id_no,
            'current_directory': current_dir,
            'directory_exists': exists,
            'directory_path': recognition_logs_dir,
            'absolute_path': abs_path,
            'files': files,
            'subdirectories': subdirs
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Hata: {str(e)}'}), 500

@app.route('/recognize', methods=['POST'])
def recognize_face():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'message': 'Resim verisi gerekli'}), 400
        face_image_base64 = data['image']
        success, result = face_api.recognize_face(face_image_base64)
        if success:
            return jsonify({
                'success': True,
                'recognized': True,
                'name': result.get('name'),
                'id_no': result.get('id_no'),
                'birth_date': result.get('birth_date'),
                'message': f"Kişi tanındı: {result.get('name')}"
            })
        else:
            # Eğer result bir dict ise (zaten tanınan kişi durumu)
            if isinstance(result, dict):
                return jsonify({
                    'success': True,
                    'recognized': False,
                    'name': result.get('name'),
                    'id_no': result.get('id_no'),
                    'birth_date': result.get('birth_date'),
                    'message': result
                })
            else:
                return jsonify({'success': False, 'recognized': False, 'message': result})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Hata: {str(e)}'}), 500

@app.route('/add_user', methods=['POST'])
def add_user():
    try:
        data = request.get_json()
        name = data['name']
        images_base64 = data['images']
        id_no = data['id_no']
        birth_date = data['birth_date']
        success, message = face_api.add_user(name, images_base64, id_no, birth_date)
        return jsonify({'success': success, 'message': message})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Hata: {str(e)}'}), 500

@app.route('/delete_user/<user_id>', methods=['DELETE'])
def delete_user(user_id):
    try:
        success, message = face_api.delete_user(user_id)
        return jsonify({'success': success, 'message': message})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Hata: {str(e)}'}), 500

@app.route('/update_user_name/<user_id>', methods=['PUT'])
def update_user_name(user_id):
    try:
        print(f"🔄 update_user_name çağrıldı: user_id={user_id}")
        data = request.get_json()
        print(f"📄 Gelen veri: {data}")
        new_name = data.get('name')

        if not new_name:
            print("❌ Yeni isim belirtilmedi")
            return jsonify({'success': False, 'message': 'Yeni isim belirtilmedi'}), 400

        print(f"📝 Yeni isim: {new_name}")

        # Users.json dosyasını oku
        if not os.path.exists(USERS_DB_FILE):
            print("❌ Users.json dosyası bulunamadı")
            return jsonify({'success': False, 'message': 'Kullanıcı veritabanı bulunamadı'}), 404

        with open(USERS_DB_FILE, 'r', encoding='utf-8') as f:
            users_data = json.load(f)

        print(f"👥 Toplam kullanıcı sayısı: {len(users_data)}")

        # Kullanıcıyı bul ve güncelle
        user_found = False
        for user in users_data:
            print(f"🔍 Kullanıcı kontrol ediliyor: {user.get('id')} veya {user.get('id_no')} == {user_id}")
            if user.get('id') == user_id or user.get('id_no') == user_id:
                print(f"✅ Kullanıcı bulundu: {user}")
                user['name'] = new_name
                user_found = True
                break

        if not user_found:
            print(f"❌ Kullanıcı bulunamadı: {user_id}")
            return jsonify({'success': False, 'message': 'Kullanıcı bulunamadı'}), 404

        # Güncellenmiş veriyi kaydet
        with open(USERS_DB_FILE, 'w', encoding='utf-8') as f:
            json.dump(users_data, f, ensure_ascii=False, indent=2)

        print("💾 Veri kaydedildi")

        # Cache'i güncelle
        face_api.update_cache()

        print("✅ Kullanıcı adı başarıyla güncellendi")
        return jsonify({'success': True, 'message': 'Kullanıcı adı başarıyla güncellendi'})

    except Exception as e:
        print(f"❌ update_user_name hatası: {e}")
        return jsonify({'success': False, 'message': f'Hata: {str(e)}'}), 500

@app.route('/users', methods=['GET'])
def get_users():
    try:
        users = face_api.get_users()
        return jsonify({'success': True, 'users': users})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Hata: {str(e)}'}), 500

@app.route('/user_logs/<user_id>', methods=['GET'])
def get_user_logs(user_id):
    try:
        logs = []
        if os.path.exists(RECOGNITION_LOG_FILE):
            with open(RECOGNITION_LOG_FILE, 'r', encoding='utf-8') as f:
                all_logs = json.load(f)
            logs = [log for log in all_logs if log.get('user_id') == user_id]
        return jsonify({'success': True, 'logs': logs})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Hata: {str(e)}'}), 500

@app.route('/user_photo/<id_no>/<filename>')
def user_photo(id_no, filename):
    user_dir = os.path.join(KNOWN_FACES_DIR, id_no)
    return send_from_directory(user_dir, filename)

@app.route('/user_photos/<id_no>', methods=['GET'])
def get_user_photos(id_no):
    """Kullanıcının profil fotoğraflarını listeler"""
    try:
        user_dir = os.path.join(KNOWN_FACES_DIR, id_no)
        if not os.path.exists(user_dir):
            return jsonify({'success': True, 'photos': []})

        photos = []
        for filename in os.listdir(user_dir):
            if filename.endswith('.jpg') and filename != 'encodings.pkl':
                photos.append(filename)

        # Dosya adlarına göre sırala (1.jpg, 2.jpg, 3.jpg...)
        photos.sort(key=lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else 0)

        return jsonify({'success': True, 'photos': photos})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Hata: {str(e)}'}), 500


@app.route('/user_photos/<id_no>/delete', methods=['POST'])
def delete_user_photos(id_no):
    """Kullanıcının belirli fotoğraflarını siler"""
    try:
        data = request.get_json()
        photo_names = data.get('photo_names', [])

        if not photo_names:
            return jsonify({'success': False, 'message': 'Silinecek fotoğraf belirtilmedi'}), 400

        user_dir = os.path.join(KNOWN_FACES_DIR, id_no)
        if not os.path.exists(user_dir):
            return jsonify({'success': False, 'message': 'Kullanıcı bulunamadı'}), 404

        deleted_count = 0
        errors = []

        for photo_name in photo_names:
            try:
                photo_path = os.path.join(user_dir, photo_name)
                if os.path.exists(photo_path):
                    os.remove(photo_path)
                    deleted_count += 1
                    print(f"Fotoğraf silindi: {photo_path}")
                else:
                    errors.append(f"Fotoğraf bulunamadı: {photo_name}")
            except Exception as e:
                errors.append(f"Silme hatası ({photo_name}): {str(e)}")

        # Embeddings'i yeniden yükle
        face_api.load_known_faces()

        return jsonify({
            'success': True,
            'deleted_count': deleted_count,
            'errors': errors,
            'message': f'{deleted_count} fotoğraf silindi'
        })

    except Exception as e:
        return jsonify({'success': False, 'message': f'Hata: {str(e)}'}), 500

@app.route('/test_user_photos/<id_no>', methods=['GET'])
def test_user_photos(id_no):
    """Test endpoint - kullanıcı fotoğrafları klasörünü kontrol et"""
    try:
        import os
        current_dir = os.getcwd()
        user_dir = os.path.join(KNOWN_FACES_DIR, id_no)
        abs_path = os.path.abspath(user_dir)
        exists = os.path.exists(user_dir)
        files = []
        if exists:
            files = os.listdir(user_dir)
        return jsonify({
            'success': True,
            'id_no': id_no,
            'current_directory': current_dir,
            'directory_exists': exists,
            'directory_path': user_dir,
            'absolute_path': abs_path,
            'files': files
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Hata: {str(e)}'}), 500

@app.route('/recognition_photos/<id_no>/<timestamp>/<filename>')
def recognition_photo(id_no, timestamp, filename):
    """Tanıma fotoğraflarını görüntülemek için endpoint"""
    recognition_dir = os.path.join(KNOWN_FACES_DIR, id_no, 'recognition_logs', timestamp)
    return send_from_directory(recognition_dir, filename)

@app.route('/recognition_logs/<id_no>', methods=['GET'])
def get_recognition_photos(id_no):
    """Bir kullanıcının tüm tanıma fotoğraflarını listeler"""
    try:
        recognition_logs_dir = os.path.join(KNOWN_FACES_DIR, id_no, 'recognition_logs')
        if not os.path.exists(recognition_logs_dir):
            return jsonify({'success': True, 'logs': []})

        logs = []
        for timestamp_dir in os.listdir(recognition_logs_dir):
            timestamp_path = os.path.join(recognition_logs_dir, timestamp_dir)
            if os.path.isdir(timestamp_path):
                info_file = os.path.join(timestamp_path, 'recognition_info.json')
                if os.path.exists(info_file):
                    with open(info_file, 'r', encoding='utf-8') as f:
                        info = json.load(f)
                        
                        # Threshold loglarını da al
                        threshold_logs = get_threshold_logs_for_user(id_no, timestamp_dir)
                        
                        logs.append({
                            'timestamp': timestamp_dir,
                            'datetime': info.get('timestamp'),
                            'files': info.get('files', {}),
                            'photo_details': info.get('photo_details', {}),
                            'threshold_logs': threshold_logs,
                            'photo_urls': {
                                'full_image': f'/recognition_photos/{id_no}/{timestamp_dir}/full_image.jpg',
                                'face_crop': f'/recognition_photos/{id_no}/{timestamp_dir}/face_crop.jpg'
                            },
                            'full_base64': info.get('files', {}).get('full_base64', ''),  # Orijinal fotoğraf
                            'face_base64': info.get('files', {}).get('face_base64', '')   # Yüz kısmı
                        })

        # Tarihe göre sırala (en yeni en üstte)
        logs.sort(key=lambda x: x['datetime'], reverse=True)

        return jsonify({'success': True, 'logs': logs})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Hata: {str(e)}'}), 500

@app.route('/augmentation_stats', methods=['GET'])
def augmentation_stats():
    """Augmentation istatistiklerini döndüren endpoint"""
    try:
        stats = get_augmentation_stats()
        return jsonify({'success': True, 'stats': stats})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Hata: {str(e)}'}), 500

@app.route('/test_augmentation', methods=['POST'])
def test_augmentation():
    """Augmentation test endpoint'i"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'message': 'Resim verisi gerekli'}), 400

        # Base64'ten PIL Image'e çevir
        image_data = base64.b64decode(data['image'].split(',')[1] if ',' in data['image'] else data['image'])
        image = Image.open(io.BytesIO(image_data))

        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Augmented versiyonlar oluştur
        num_versions = data.get('num_versions', 4)
        augmented_versions = create_augmented_versions(image, num_versions)

        # Base64'e çevir
        results = []
        for i, aug_img in enumerate(augmented_versions):
            img_buffer = io.BytesIO()
            aug_img.save(img_buffer, format='JPEG', quality=95)
            img_str = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            results.append({
                'index': i + 1,
                'image': f'data:image/jpeg;base64,{img_str}'
            })

        return jsonify({
            'success': True,
            'original_image': data['image'],
            'augmented_versions': results,
            'total_versions': len(results)
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Hata: {str(e)}'}), 500

@app.route('/apply_single_augmentation', methods=['POST'])
def apply_single_augmentation():
    """Tek bir augmentation türü uygulama"""
    try:
        data = request.get_json()
        if not data or 'image' not in data or 'augmentation_type' not in data:
            return jsonify({'success': False, 'message': 'Resim ve augmentation türü gerekli'}), 400

        # Base64'ten PIL Image'e çevir
        image_data = base64.b64decode(data['image'].split(',')[1] if ',' in data['image'] else data['image'])
        image = Image.open(io.BytesIO(image_data))

        if image.mode != 'RGB':
            image = image.convert('RGB')

        augmentation_type = data['augmentation_type']
        params = data.get('params', {})

        # Augmentation uygula
        if augmentation_type == 'brightness_contrast':
            brightness = params.get('brightness', 1.2)
            contrast = params.get('contrast', 1.1)
            result = apply_brightness_contrast(image, brightness, contrast)
        elif augmentation_type == 'noise':
            noise_factor = params.get('noise_factor', 0.02)
            result = apply_noise(image, noise_factor)
        elif augmentation_type == 'blur':
            blur_factor = params.get('blur_factor', 1.0)
            result = apply_blur(image, blur_factor)
        elif augmentation_type == 'sharpening':
            sharpness = params.get('sharpness', 1.5)
            result = apply_sharpening(image, sharpness)
        elif augmentation_type == 'geometric_transform':
            rotation_range = params.get('rotation_range', 10)
            scale_range = params.get('scale_range', 0.1)
            result = apply_geometric_transform(image, rotation_range, scale_range)
        elif augmentation_type == 'lighting_simulation':
            lighting_factor = params.get('lighting_factor', 0.3)
            result = apply_lighting_simulation(image, lighting_factor)
        else:
            return jsonify({'success': False, 'message': 'Geçersiz augmentation türü'}), 400

        # Sonucu base64'e çevir
        img_buffer = io.BytesIO()
        result.save(img_buffer, format='JPEG', quality=95)
        img_str = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

        return jsonify({
            'success': True,
            'original_image': data['image'],
            'augmented_image': f'data:image/jpeg;base64,{img_str}',
            'augmentation_type': augmentation_type,
            'params': params
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Hata: {str(e)}'}), 500

@app.route('/test_image_resolution', methods=['POST'])
def test_image_resolution():
    """Görüntü çözünürlüğünü test etmek için endpoint"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'message': 'Resim verisi gerekli'}), 400

        face_image_base64 = data['image']
        image_array = face_api.process_image_with_rotation(face_image_base64)

        if image_array is None:
            return jsonify({'success': False, 'message': 'Görüntü işlenemedi'}), 400

        height, width = image_array.shape[:2]
        faces_with_landmarks = face_api.detect_faces_with_landmarks(image_array)

        # Çözünürlük analizi
        resolution_quality = "İyi"
        if width < 640 or height < 640:
            resolution_quality = "Düşük"
        elif width < 1280 or height < 1280:
            resolution_quality = "Orta"

        # Yüz tespit analizi - sadece orijinal görüntüde
        face_detection_quality = "Başarılı"
        if len(faces_with_landmarks) == 0:
            face_detection_quality = "Yüz tespit edilemedi (sadece orijinal yönde arandı)"
        elif len(faces_with_landmarks) > 1:
            face_detection_quality = f"{len(faces_with_landmarks)} yüz tespit edildi"

        # Yüz boyutu analizi
        face_size_quality = "Uygun"
        if len(faces_with_landmarks) > 0:
            x, y, w, h = faces_with_landmarks[0]['box']
            if w < 80 or h < 80:
                face_size_quality = "Çok küçük"
            elif w < 120 or h < 120:
                face_size_quality = "Küçük"

        return jsonify({
            'success': True,
            'image_info': {
                'width': width,
                'height': height,
                'resolution_quality': resolution_quality
            },
            'face_detection': {
                'faces_found': len(faces_with_landmarks),
                'detection_quality': face_detection_quality,
                'face_size_quality': face_size_quality
            },
            'recommendations': {
                'resolution': "1280x1280 veya daha yüksek önerilir" if resolution_quality == "Düşük" else "Çözünürlük uygun",
                'face_size': "Yüz daha büyük olmalı" if face_size_quality in ["Çok küçük", "Küçük"] else "Yüz boyutu uygun",
                'lighting': "İyi aydınlatma önerilir" if len(faces_with_landmarks) == 0 else "Aydınlatma uygun"
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Hata: {str(e)}'}), 500

@app.route('/detect_faces', methods=['POST'])
def detect_faces():
    """Yüz tespit koordinatlarını döndüren endpoint"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'message': 'Resim verisi gerekli'}), 400

        face_image_base64 = data['image']
        image_array = face_api.process_image_with_rotation(face_image_base64)

        if image_array is None:
            return jsonify({'success': False, 'message': 'Görüntü işlenemedi'}), 400

        height, width = image_array.shape[:2]
        faces_with_landmarks = face_api.detect_faces_with_landmarks(image_array)
        rotation_angle = 0

        # Sadece orijinal görüntüde yüz tespiti yap, başka rotasyonlar deneme
        if len(faces_with_landmarks) == 0:
            print("Orijinal görüntüde yüz bulunamadı")
            return jsonify({
                'success': True,
                'faces': [],
                'total_faces': 0,
                'image_info': {
                    'width': width,
                    'height': height,
                    'rotation_angle': rotation_angle
                },
                'message': 'Yüz tespit edilemedi'
            })

        # Yüz koordinatlarını hazırla
        faces_data = []
        for face in faces_with_landmarks:
            x, y, w, h = face['box']
            confidence = face['confidence']

            # Margin ekle (daha geniş margin)
            margin = int(min(w, h) * 0.3)  # %30 margin
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(width, x + w + margin)
            y2 = min(height, y + h + margin)

            # Kare şeklinde yap (en büyük boyutu kullan)
            size = max(x2 - x1, y2 - y1)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Kare koordinatları
            square_x = max(0, center_x - size // 2)
            square_y = max(0, center_y - size // 2)
            square_size = min(size, width - square_x, height - square_y)

            faces_data.append({
                'x': square_x,
                'y': square_y,
                'width': square_size,
                'height': square_size,
                'confidence': confidence,
                'rotation_angle': rotation_angle,
                'original_box': {
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h
                }
            })

        return jsonify({
            'success': True,
            'faces': faces_data,
            'total_faces': len(faces_data),
            'image_info': {
                'width': width,
                'height': height,
                'rotation_angle': rotation_angle
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Hata: {str(e)}'}), 500

@app.route('/threshold/status', methods=['GET'])
def threshold_status():
    """Otomatik threshold durumunu kontrol eder"""
    try:
        last_method = None
        last_auc = None
        last_tpr = None
        last_fpr = None

        if face_api.threshold_history:
            last_record = face_api.threshold_history[-1]
            last_method = last_record.get('method', 'unknown')
            last_auc = last_record.get('auc')
            last_tpr = last_record.get('tpr')
            last_fpr = last_record.get('fpr')

        return jsonify({
            'success': True,
            'auto_threshold_enabled': face_api.auto_threshold_enabled,
            'current_threshold': face_api.optimal_threshold,
            'adaptive_threshold': face_api.get_adaptive_threshold(),
            'threshold_history_count': len(face_api.threshold_history),
            'last_threshold_update': face_api.threshold_history[-1]['timestamp'] if face_api.threshold_history else None,
            'last_method': last_method,
            'last_auc': last_auc,
            'last_tpr': last_tpr,
            'last_fpr': last_fpr
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Hata: {str(e)}'}), 500

@app.route('/threshold/toggle', methods=['POST'])
def toggle_auto_threshold():
    """Otomatik threshold'u açıp kapatır"""
    try:
        data = request.get_json()
        enabled = data.get('enabled', True)
        face_api.auto_threshold_enabled = enabled

        return jsonify({
            'success': True,
            'auto_threshold_enabled': face_api.auto_threshold_enabled,
            'message': f'Otomatik threshold {"açıldı" if enabled else "kapatıldı"}'
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Hata: {str(e)}'}), 500

@app.route('/threshold/history', methods=['GET'])
def get_threshold_history():
    """Threshold geçmişini döndürür"""
    try:
        return jsonify({
            'success': True,
            'history': face_api.threshold_history[-20:]  # Son 20 kayıt
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Hata: {str(e)}'}), 500

@app.route('/threshold/set', methods=['POST'])
def set_manual_threshold():
    """Manuel threshold değeri ayarlar"""
    try:
        data = request.get_json()
        threshold = data.get('threshold', 0.7270)

        # Threshold'u sınırla
        threshold = max(0.3, min(0.8, threshold))
        face_api.optimal_threshold = threshold

        return jsonify({
            'success': True,
            'threshold': face_api.optimal_threshold,
            'message': f'Threshold {threshold:.4f} olarak ayarlandı'
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Hata: {str(e)}'}), 500

@app.route('/threshold/calculate_roc', methods=['POST'])
def calculate_roc_threshold():
    """ROC eğrisi kullanarak threshold hesaplar"""
    try:
        new_threshold = face_api.calculate_threshold_via_roc()

        return jsonify({
            'success': True,
            'threshold': new_threshold,
            'method': 'roc_curve',
            'message': f'ROC tabanlı threshold hesaplandı: {new_threshold:.4f}'
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Hata: {str(e)}'}), 500

@app.route('/optimize_threshold', methods=['POST'])
def optimize_threshold():
    """Threshold optimizasyonu endpoint'i"""
    try:
        print("🔧 Threshold optimizasyonu başlatılıyor...")

        # Mevcut threshold'u al
        current_threshold = face_api.get_adaptive_threshold()
        print(f"📊 Mevcut threshold: {current_threshold}")

        # Embeddings'lerin mevcut olup olmadığını kontrol et
        if not hasattr(face_api, 'known_face_encodings') or not face_api.known_face_encodings:
            print("❌ Embeddings bulunamadı - yeniden yükleniyor...")
            face_api.load_known_faces()

            if not face_api.known_face_encodings:
                print("❌ Embeddings hala bulunamadı")
                return jsonify({
                    'success': False,
                    'message': 'Embeddings bulunamadı. Lütfen önce kullanıcı ekleyin.'
                }), 400

        if not hasattr(face_api, 'known_face_names') or not face_api.known_face_names:
            print("❌ Face names bulunamadı")
            return jsonify({
                'success': False,
                'message': 'Face names bulunamadı. Lütfen önce kullanıcı ekleyin.'
            }), 400

        if not hasattr(face_api, 'known_face_ids') or not face_api.known_face_ids:
            print("❌ Face IDs bulunamadı")
            return jsonify({
                'success': False,
                'message': 'Face IDs bulunamadı. Lütfen önce kullanıcı ekleyin.'
            }), 400

        print(f"📈 Toplam embeddings: {len(face_api.known_face_encodings)}")
        print(f"📈 Toplam names: {len(face_api.known_face_names)}")
        print(f"📈 Toplam IDs: {len(face_api.known_face_ids)}")

        # Embeddings'leri kişilere göre grupla
        embeddings_dict = defaultdict(list)
        for name, id_no, emb in zip(face_api.known_face_names, face_api.known_face_ids, face_api.known_face_encodings):
            embeddings_dict[id_no].append(emb)

        if len(embeddings_dict) < 2:
            print(f"❌ Yeterli kullanıcı yok: {len(embeddings_dict)}")
            return jsonify({
                'success': False,
                'message': f'ROC için yeterli kullanıcı yok (en az 2 kullanıcı gerekli, mevcut: {len(embeddings_dict)})'
            }), 400

        y_true = []
        distances = []

        # Pozitif çiftler (aynı kişi)
        print("Pozitif çiftler hesaplanıyor...")
        positive_pairs = 0
        for emb_list in embeddings_dict.values():
            if len(emb_list) >= 2:  # En az 2 embedding gerekli
                for emb1, emb2 in combinations(emb_list, 2):
                    try:
                        dist = np.linalg.norm(emb1 - emb2)
                        distances.append(dist)
                        y_true.append(1)
                        positive_pairs += 1
                    except Exception as e:
                        print(f"❌ Pozitif pair hesaplama hatası: {e}")
                        continue

        # Negatif çiftler (farklı kişiler)
        print("Negatif çiftler hesaplanıyor...")
        negative_pairs = 0
        person_ids = list(embeddings_dict.keys())
        for i in range(len(person_ids)):
            for j in range(i + 1, len(person_ids)):
                for emb1 in embeddings_dict[person_ids[i]]:
                    for emb2 in embeddings_dict[person_ids[j]]:
                        try:
                            dist = np.linalg.norm(emb1 - emb2)
                            distances.append(dist)
                            y_true.append(0)
                            negative_pairs += 1
                        except Exception as e:
                            print(f"❌ Negatif pair hesaplama hatası: {e}")
                            continue

        if not distances or not y_true:
            print("❌ Yeterli veri yok")
            return jsonify({
                'success': False,
                'message': 'ROC için yeterli veri yok'
            }), 400

        print(f"✅ Toplam {len(distances)} çift hesaplandı ({positive_pairs} pozitif, {negative_pairs} negatif)")

        # ROC eğrisi hesapla
        try:
            fpr, tpr, thresholds = roc_curve(y_true, [-d for d in distances])
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = abs(thresholds[optimal_idx])

            # AUC hesapla
            roc_auc = auc(fpr, tpr)
        except Exception as e:
            print(f"❌ ROC hesaplama hatası: {e}")
            return jsonify({
                'success': False,
                'message': f'ROC hesaplama hatası: {str(e)}'
            }), 500

        # Threshold değişikliği hesapla
        threshold_change = optimal_threshold - current_threshold

        print(f"✅ Threshold optimizasyonu tamamlandı: {optimal_threshold:.4f}")
        print(f"📊 Değişiklik: {current_threshold:.4f} → {optimal_threshold:.4f} ({threshold_change:+.4f})")
        print(f"🔍 Pairs analizi: {positive_pairs} pozitif, {negative_pairs} negatif")
        print(f"📈 ROC analizi: AUC={roc_auc:.4f}, TPR={tpr[optimal_idx]:.4f}, FPR={fpr[optimal_idx]:.4f}")

        return jsonify({
            'success': True,
            'optimal_threshold': float(optimal_threshold),
            'current_threshold': float(current_threshold),
            'threshold_change': float(threshold_change),
            'total_pairs': len(distances),
            'positive_pairs': positive_pairs,
            'negative_pairs': negative_pairs,
            'message': f'Threshold optimize edildi: {current_threshold:.4f} → {optimal_threshold:.4f}'
        })

    except Exception as e:
        print(f"❌ Threshold optimizasyon hatası: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'Threshold optimizasyon hatası: {str(e)}'
        }), 500

@app.route('/test_face_detection', methods=['POST'])
def test_face_detection():
    """Yüz tespit testi endpoint'i"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'message': 'Resim verisi gerekli'}), 400

        # Base64'ten resmi çöz
        image_data = data['image']
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]

        image_bytes = base64.b64decode(image_data)
        image_array = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

        if image_array is None:
            return jsonify({'success': False, 'message': 'Resim yüklenemedi'}), 400

        # Yüz tespiti test et
        faces_with_landmarks = face_api.detect_faces_with_landmarks(image_array)

        # Sonuçları hazırla
        results = {
            'total_faces_detected': len(faces_with_landmarks),
            'faces': []
        }

        for i, face in enumerate(faces_with_landmarks):
            face_info = {
                'face_id': i + 1,
                'box': face.get('box', []),
                'confidence': face.get('confidence', 0.0),
                'landmarks': face.get('keypoints', {})
            }
            results['faces'].append(face_info)

        return jsonify({
            'success': True,
            'message': f'{len(faces_with_landmarks)} yüz tespit edildi',
            'results': results
        })

    except Exception as e:
        return jsonify({'success': False, 'message': f'Hata: {str(e)}'}), 500

@app.route('/threshold/logs', methods=['GET'])
def get_threshold_logs():
    """Threshold loglarını döndürür"""
    try:
        threshold_log_file = 'threshold_logs.json'
        if not os.path.exists(threshold_log_file):
            return jsonify({'success': True, 'logs': []})

        with open(threshold_log_file, 'r', encoding='utf-8') as f:
            logs = json.load(f)

        # Son 100 logu döndür
        recent_logs = logs[-100:] if len(logs) > 100 else logs

        # Threshold detaylarını da dahil et
        for log in recent_logs:
            if 'threshold_details' not in log:
                log['threshold_details'] = {}

        return jsonify({
            'success': True,
            'total_logs': len(logs),
            'recent_logs': recent_logs
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Hata: {str(e)}'}), 500

@app.route('/threshold/logs/<user_id>', methods=['GET'])
def get_user_threshold_logs(user_id):
    """Belirli bir kullanıcının threshold loglarını döndürür"""
    try:
        threshold_log_file = 'threshold_logs.json'
        if not os.path.exists(threshold_log_file):
            return jsonify({'success': True, 'logs': []})

        with open(threshold_log_file, 'r', encoding='utf-8') as f:
            all_logs = json.load(f)

        # Kullanıcıya ait logları filtrele
        user_logs = [log for log in all_logs if log.get('user_id') == user_id]

        # Threshold detaylarını da dahil et
        for log in user_logs:
            if 'threshold_details' not in log:
                log['threshold_details'] = {}

        return jsonify({
            'success': True,
            'user_id': user_id,
            'total_logs': len(user_logs),
            'logs': user_logs
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Hata: {str(e)}'}), 500

@app.route('/threshold/analytics', methods=['GET'])
def get_threshold_analytics():
    """Threshold analitiklerini döndürür"""
    try:
        threshold_log_file = 'threshold_logs.json'
        if not os.path.exists(threshold_log_file):
            return jsonify({'success': True, 'analytics': {}})

        with open(threshold_log_file, 'r', encoding='utf-8') as f:
            logs = json.load(f)

        if not logs:
            return jsonify({'success': True, 'analytics': {}})

        # Başarılı ve başarısız tanımaları ayır
        successful_logs = [log for log in logs if log.get('success', False)]
        failed_logs = [log for log in logs if not log.get('success', False)]

        # Threshold değerlerini analiz et
        thresholds = [log.get('threshold_used', 0) for log in logs]
        distances = [log.get('distance', 0) for log in logs if log.get('distance') is not None]
        confidences = [log.get('confidence', 0) for log in logs if log.get('confidence') is not None]

        analytics = {
            'total_events': len(logs),
            'successful_recognitions': len(successful_logs),
            'failed_recognitions': len(failed_logs),
            'success_rate': len(successful_logs) / len(logs) if logs else 0,
            'threshold_stats': {
                'min': min(thresholds) if thresholds else 0,
                'max': max(thresholds) if thresholds else 0,
                'avg': sum(thresholds) / len(thresholds) if thresholds else 0,
                'current': thresholds[-1] if thresholds else 0
            },
            'distance_stats': {
                'min': min(distances) if distances else 0,
                'max': max(distances) if distances else 0,
                'avg': sum(distances) / len(distances) if distances else 0
            },
            'confidence_stats': {
                'min': min(confidences) if confidences else 0,
                'max': max(confidences) if confidences else 0,
                'avg': sum(confidences) / len(confidences) if confidences else 0
            },
            'recent_events': logs[-10:]  # Son 10 olay
        }

        return jsonify({
            'success': True,
            'analytics': analytics
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Hata: {str(e)}'}), 500

@app.route('/threshold/logs/clear', methods=['POST'])
def clear_threshold_logs():
    """Threshold loglarını temizle"""
    try:
        if os.path.exists(THRESHOLD_LOG_FILE):
            os.remove(THRESHOLD_LOG_FILE)
        return jsonify({'success': True, 'message': 'Threshold logları temizlendi'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Log temizleme hatası: {e}'}), 500

@app.route('/analyze_lighting', methods=['POST'])
def analyze_lighting():
    """Işık durumunu analiz et"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'Görüntü verisi bulunamadı'}), 400
        
        # Base64 görüntüyü decode et
        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data))
        
        # PIL Image'i numpy array'e çevir
        image_array = np.array(image)
        
        # Işık analizi yap
        lighting_result = analyze_lighting_conditions(image_array)
        
        return jsonify(lighting_result)
        
    except Exception as e:
        print(f"Işık analizi endpoint hatası: {e}")
        return jsonify({
            'condition': 'unknown',
            'message': 'Işık durumu belirlenemedi',
            'suggestion': 'Kamerayı yeniden konumlandırın',
            'error': str(e)
        }), 500

@app.route('/recognition_details/<id_no>', methods=['GET'])
def get_recognition_details(id_no):
    """Bir kullanıcının tanıma detaylarını (fotoğraf + threshold) döndürür"""
    try:
        recognition_logs_dir = os.path.join(KNOWN_FACES_DIR, id_no, 'recognition_logs')
        if not os.path.exists(recognition_logs_dir):
            return jsonify({'success': True, 'details': []})

        details = []
        for timestamp_dir in os.listdir(recognition_logs_dir):
            timestamp_path = os.path.join(recognition_logs_dir, timestamp_dir)
            if os.path.isdir(timestamp_path):
                info_file = os.path.join(timestamp_path, 'recognition_info.json')
                if os.path.exists(info_file):
                    with open(info_file, 'r', encoding='utf-8') as f:
                        info = json.load(f)
                        
                        # Threshold loglarını da al
                        threshold_logs = get_threshold_logs_for_user(id_no, timestamp_dir)
                        
                        details.append({
                            'timestamp': timestamp_dir,
                            'datetime': info.get('timestamp'),
                            'files': info.get('files', {}),
                            'photo_details': info.get('photo_details', {}),
                            'threshold_logs': threshold_logs,
                            'photo_urls': {
                                'full_image': f'/recognition_photos/{id_no}/{timestamp_dir}/full_image.jpg',
                                'face_crop': f'/recognition_photos/{id_no}/{timestamp_dir}/face_crop.jpg'
                            }
                        })

        # Tarihe göre sırala (en yeni en üstte)
        details.sort(key=lambda x: x['datetime'], reverse=True)

        return jsonify({'success': True, 'details': details})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Hata: {str(e)}'}), 500

def get_threshold_logs_for_user(user_id, timestamp_dir=None):
    """Belirli bir kullanıcının threshold loglarını döndürür"""
    try:
        threshold_log_file = 'threshold_logs.json'
        if not os.path.exists(threshold_log_file):
            return []
        
        with open(threshold_log_file, 'r', encoding='utf-8') as f:
            logs = json.load(f)
        
        # Kullanıcının loglarını filtrele
        user_logs = [log for log in logs if log.get('user_id') == user_id]
        
        # Eğer timestamp_dir belirtilmişse, o zaman dilimindeki logları filtrele
        if timestamp_dir:
            # timestamp_dir formatı: YYYYMMDD_HHMMSS
            # ISO timestamp formatına çevir
            try:
                dt = datetime.strptime(timestamp_dir, "%Y%m%d_%H%M%S")
                start_time = dt.replace(second=0, microsecond=0)
                end_time = start_time.replace(second=59, microsecond=999999)
                
                filtered_logs = []
                for log in user_logs:
                    log_time = datetime.fromisoformat(log.get('timestamp', '').replace('Z', '+00:00'))
                    if start_time <= log_time <= end_time:
                        filtered_logs.append(log)
                return filtered_logs
            except:
                return user_logs
        
        return user_logs
    except Exception as e:
        print(f"Threshold log okuma hatası: {e}")
        return []

@app.route('/threshold/details/<user_id>', methods=['GET'])
def get_user_threshold_details(user_id):
    """Belirli bir kullanıcının threshold hesaplama detaylarını döndürür"""
    try:
        threshold_logs = get_threshold_logs_for_user(user_id)
        
        if not threshold_logs:
            return jsonify({'success': True, 'threshold_details': []})
        
        # Threshold detaylarını işle
        threshold_details = []
        for log in threshold_logs:
            detail = {
                'timestamp': log.get('timestamp'),
                'threshold_used': log.get('threshold_used'),
                'distance': log.get('distance'),
                'confidence': log.get('confidence'),
                'success': log.get('success'),
                'threshold_details': log.get('threshold_details', {})
            }
            threshold_details.append(detail)
        
        # Tarihe göre sırala
        threshold_details.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return jsonify({'success': True, 'threshold_details': threshold_details})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Hata: {str(e)}'}), 500

@app.route('/recognition_photo_details/<id_no>/<timestamp>', methods=['GET'])
def get_recognition_photo_details(id_no, timestamp):
    """Belirli bir tanıma fotoğrafının detaylarını döndürür"""
    try:
        recognition_logs_dir = os.path.join(KNOWN_FACES_DIR, id_no, 'recognition_logs', timestamp)
        if not os.path.exists(recognition_logs_dir):
            return jsonify({'success': False, 'message': 'Tanıma kaydı bulunamadı'}), 404

        info_file = os.path.join(recognition_logs_dir, 'recognition_info.json')
        if not os.path.exists(info_file):
            return jsonify({'success': False, 'message': 'Tanıma bilgisi bulunamadı'}), 404

        with open(info_file, 'r', encoding='utf-8') as f:
            info = json.load(f)

        # Threshold loglarını da al
        threshold_logs = get_threshold_logs_for_user(id_no, timestamp)

        # Dosya boyutlarını kontrol et
        full_image_path = os.path.join(recognition_logs_dir, 'full_image.jpg')
        face_crop_path = os.path.join(recognition_logs_dir, 'face_crop.jpg')
        
        full_image_size = os.path.getsize(full_image_path) if os.path.exists(full_image_path) else 0
        face_crop_size = os.path.getsize(face_crop_path) if os.path.exists(face_crop_path) else 0

        details = {
            'timestamp': timestamp,
            'datetime': info.get('timestamp'),
            'user_id': info.get('user_id'),
            'files': info.get('files', {}),
            'photo_details': info.get('photo_details', {}),
            'threshold_logs': threshold_logs,
            'photo_urls': {
                'full_image': f'/recognition_photos/{id_no}/{timestamp}/full_image.jpg',
                'face_crop': f'/recognition_photos/{id_no}/{timestamp}/face_crop.jpg'
            },
            'recognition_log': info.get('recognition_log', {}),
            'file_sizes': {
                'full_image_bytes': full_image_size,
                'face_crop_bytes': face_crop_size
            },
            'image_info': {
                'full_image_dimensions': info.get('photo_details', {}).get('image_info', {}).get('full_image_dimensions', 'N/A'),
                'face_crop_dimensions': info.get('photo_details', {}).get('image_info', {}).get('face_crop_dimensions', 'N/A'),
                'full_image_channels': info.get('photo_details', {}).get('image_info', {}).get('full_image_channels', 'N/A'),
                'face_crop_channels': info.get('photo_details', {}).get('image_info', {}).get('face_crop_channels', 'N/A')
            },
            'captured_images': {
                'full_image': {
                    'captured': True,
                    'description': 'Orijinal çekilen fotoğrafın tamamı',
                    'file_path': full_image_path,
                    'file_size_bytes': full_image_size,
                    'dimensions': info.get('photo_details', {}).get('image_info', {}).get('full_image_dimensions', 'N/A')
                },
                'face_crop': {
                    'captured': True,
                    'description': 'Sadece yüz kısmı, hizalanmış ve kırpılmış',
                    'file_path': face_crop_path,
                    'file_size_bytes': face_crop_size,
                    'dimensions': info.get('photo_details', {}).get('image_info', {}).get('face_crop_dimensions', 'N/A')
                }
            }
        }

        return jsonify({'success': True, 'details': details})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Hata: {str(e)}'}), 500

if __name__ == '__main__':
    print("Yüz Tanıma API'si başlatılıyor...")
    app.run(host='0.0.0.0', port=5000, debug=True)