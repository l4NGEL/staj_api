# Yüz Tanıma API (Face Recognition API)

Bu proje, gelişmiş yüz tanıma özelliklerine sahip bir Flask tabanlı REST API'sidir. MobilNet modeli kullanarak yüz tanıma, kullanıcı yönetimi, görüntü işleme ve analitik özellikleri sunar.

## 🚀 Özellikler

### Temel Özellikler
- **Yüz Tanıma**: MobilNet modeli ile yüksek doğruluklu yüz tanıma
- **Kullanıcı Yönetimi**: Kullanıcı ekleme, silme, güncelleme
- **Görüntü İşleme**: Otomatik yüz hizalama, kırpma ve iyileştirme
- **Çoklu Yüz Algılama**: MTCNN ve Haar Cascade ile yüz algılama
- **Görüntü Artırma**: Parlaklık, kontrast, gürültü, bulanıklık, keskinleştirme

### Gelişmiş Özellikler
- **Adaptif Threshold**: ROC eğrisi tabanlı otomatik threshold optimizasyonu
- **Gerçek Zamanlı Loglama**: Tanıma işlemlerinin detaylı loglanması
- **Analitik Dashboard**: Threshold performans analizi
- **Aydınlatma Analizi**: Görüntü aydınlatma koşullarının analizi
- **Otomatik Rotasyon**: Görüntü otomatik döndürme
- **Batch İşleme**: Toplu yüz karşılaştırma

## 📁 Proje Yapısı

```
staj_api/
├── ff.py                          # Ana Flask uygulaması
├── users.json                     # Kullanıcı veritabanı
├── logs.json                      # Tanıma logları
├── threshold_logs.json            # Threshold logları
├── recognition_logs.json          # Detaylı tanıma logları
├── faces/                         # Kullanıcı yüz görüntüleri
│   ├── [user_id]/                # Her kullanıcı için ayrı klasör
│   │   ├── photos/               # Kullanıcı fotoğrafları
│   │   └── recognition_logs/     # Tanıma kayıtları
└── README.md                      # Bu dosya
```

## 🛠️ Kurulum

### Gereksinimler

```bash
pip install -r requirements.txt
```

### Gerekli Kütüphaneler

```python
flask
flask-cors
opencv-python
numpy
pillow
tensorflow
mtcnn
scikit-learn
scipy
```

### Çalıştırma

```bash
python ff.py
```

API varsayılan olarak `http://localhost:5000` adresinde çalışacaktır.

## 📚 API Endpoints

### Kullanıcı Yönetimi

#### Kullanıcı Ekleme
```http
POST /add_user
Content-Type: application/json

{
  "name": "John Doe",
  "id_no": "12345678901",
  "birth_date": "1990-01-01",
  "images": ["base64_encoded_image1", "base64_encoded_image2"]
}
```

#### Kullanıcı Listesi
```http
GET /users
```

#### Kullanıcı Silme
```http
DELETE /delete_user/{user_id}
```

#### Kullanıcı Adı Güncelleme
```http
PUT /update_user_name/{user_id}
Content-Type: application/json

{
  "name": "New Name"
}
```

### Yüz Tanıma

#### Yüz Tanıma
```http
POST /recognize
Content-Type: application/json

{
  "image": "base64_encoded_image"
}
```

#### Yüz Algılama
```http
POST /detect_faces
Content-Type: application/json

{
  "image": "base64_encoded_image"
}
```

### Threshold Yönetimi

#### Threshold Durumu
```http
GET /threshold/status
```

#### Threshold Toggle
```http
POST /threshold/toggle
```

#### Threshold Ayarlama
```http
POST /threshold/set
Content-Type: application/json

{
  "threshold": 0.6
}
```

#### ROC Threshold Hesaplama
```http
POST /threshold/calculate_roc
```

### Loglar ve Analitik

#### Gerçek Zamanlı Loglar
```http
GET /realtime_recognition_logs
```

#### Kullanıcı Logları
```http
GET /user_logs/{user_id}
```

#### Threshold Logları
```http
GET /threshold/logs
```

#### Threshold Analitik
```http
GET /threshold/analytics
```

### Görüntü İşleme

#### Görüntü Artırma Testi
```http
POST /test_augmentation
Content-Type: application/json

{
  "image": "base64_encoded_image"
}
```

#### Aydınlatma Analizi
```http
POST /analyze_lighting
Content-Type: application/json

{
  "image": "base64_encoded_image"
}
```

### Sistem

#### Sağlık Kontrolü
```http
GET /health
```

#### JSON Düzeltme
```http
POST /fix_json
```

## 🔧 Konfigürasyon

### Önemli Değişkenler

```python
USERS_DB_FILE = 'users.json'                    # Kullanıcı veritabanı dosyası
KNOWN_FACES_DIR = 'faces'                       # Yüz görüntüleri klasörü
RECOGNITION_LOG_FILE = 'logs.json'              # Tanıma log dosyası
MOBILFACENET_MODEL_PATH = 'facenet.tflite'      # MobilNet model dosyası
THRESHOLD_LOG_FILE = 'threshold_logs.json'      # Threshold log dosyası
```

### Threshold Ayarları

- **Varsayılan Threshold**: 0.6
- **Adaptif Threshold**: ROC eğrisi tabanlı otomatik hesaplama
- **Manuel Threshold**: Kullanıcı tarafından ayarlanabilir

## 📊 Veri Yapıları

### Kullanıcı Verisi
```json
{
  "id": "12345678901",
  "name": "John Doe",
  "id_no": "12345678901",
  "birth_date": "1990-01-01",
  "created_at": "2025-01-01T00:00:00.000000"
}
```

### Tanıma Sonucu
```json
{
  "success": true,
  "user_id": "12345678901",
  "name": "John Doe",
  "confidence": 0.85,
  "distance": 0.15,
  "threshold_used": 0.6
}
```

## 🎯 Kullanım Örnekleri

### Python ile API Kullanımı

```python
import requests
import base64

# Görüntüyü base64'e çevir
with open("face.jpg", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode()

# Yüz tanıma
response = requests.post('http://localhost:5000/recognize', 
                        json={'image': encoded_string})
result = response.json()

if result['success']:
    print(f"Tanınan kişi: {result['name']}")
    print(f"Güven: {result['confidence']}")
else:
    print("Kişi tanınamadı")
```

### cURL ile API Kullanımı

```bash
# Kullanıcı listesi
curl -X GET http://localhost:5000/users

# Yüz tanıma
curl -X POST http://localhost:5000/recognize \
  -H "Content-Type: application/json" \
  -d '{"image": "base64_encoded_image"}'
```

## 🔍 Hata Ayıklama

### Yaygın Sorunlar

1. **Model Dosyası Bulunamadı**: `facenet.tflite` dosyasının mevcut olduğundan emin olun
2. **Görüntü Formatı**: Sadece JPEG ve PNG formatları desteklenir
3. **Bellek Sorunları**: Büyük görüntüler için bellek yetersizliği olabilir

### Log Dosyaları

- `logs.json`: Genel tanıma logları
- `threshold_logs.json`: Threshold hesaplama logları
- `recognition_logs.json`: Detaylı tanıma logları

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/AmazingFeature`)
3. Commit yapın (`git commit -m 'Add some AmazingFeature'`)
4. Push yapın (`git push origin feature/AmazingFeature`)
5. Pull Request oluşturun


## 👥 Geliştirici

Bu proje staj projesi olarak geliştirilmiştir.

## 🔄 Güncellemeler

### v1.0.0
- Temel yüz tanıma özellikleri
- Kullanıcı yönetimi
- Threshold optimizasyonu
- Gerçek zamanlı loglama
- Analitik dashboard

---

**Not**: Bu API production ortamında kullanılmadan önce güvenlik önlemleri alınmalı ve performans testleri yapılmalıdır.
