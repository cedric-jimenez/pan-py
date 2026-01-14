# Exemples d'Appels API - Salamander Detection

Ce fichier contient des exemples d'appels curl pour tester tous les endpoints de l'API.

## Prérequis

1. **Démarrer le serveur**:
```bash
make run
# ou
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

2. **Préparer une image de test**:
```bash
# Télécharger une image de test ou utiliser une de vos images
# Par exemple:
ls imagestest/
```

---

## 🏥 Health Check Endpoints

### 1. Root Endpoint (/)
```bash
curl http://localhost:8000/
```

**Réponse attendue**:
```json
{
  "status": "healthy",
  "yolo_loaded": true,
  "segment_loaded": true,
  "version": "0.1.0"
}
```

### 2. Health Check (/health)
```bash
curl http://localhost:8000/health
```

**Réponse attendue**:
```json
{
  "status": "healthy",
  "yolo_loaded": true,
  "segment_loaded": true,
  "version": "0.1.0"
}
```

---

## 🔍 Detection Endpoint

### 3. Crop Salamander (POST /crop-salamander)

#### Exemple Simple (avec paramètres par défaut)
```bash
curl -X POST http://localhost:8000/crop-salamander \
  -F "file=@imagestest/salamander.jpg"
```

#### Exemple avec Tous les Paramètres
```bash
curl -X POST "http://localhost:8000/crop-salamander?confidence=0.3&return_base64=true&image_format=JPEG&image_quality=85&max_size=1280" \
  -F "file=@imagestest/salamander.jpg" \
  -H "accept: application/json"
```

#### Exemple sans Base64 (plus rapide)
```bash
curl -X POST "http://localhost:8000/crop-salamander?return_base64=false&confidence=0.25" \
  -F "file=@imagestest/salamander.jpg"
```

#### Exemple avec Format PNG (plus lent mais sans perte)
```bash
curl -X POST "http://localhost:8000/crop-salamander?image_format=PNG&return_base64=true" \
  -F "file=@imagestest/salamander.jpg"
```

#### Exemple avec Haute Résolution
```bash
curl -X POST "http://localhost:8000/crop-salamander?max_size=4096&confidence=0.25" \
  -F "file=@imagestest/salamander.jpg"
```

#### Sauvegarder la Réponse dans un Fichier
```bash
curl -X POST "http://localhost:8000/crop-salamander?return_base64=true" \
  -F "file=@imagestest/salamander.jpg" \
  -o result.json

# Pour extraire l'image base64 et la décoder:
jq -r '.cropped_image_base64' result.json | base64 -d > cropped_salamander.jpg
```

**Paramètres disponibles**:
| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `file` | File | Required | Image à analyser (JPEG, PNG, etc.) |
| `confidence` | float | 0.25 | Seuil de confiance (0.0-1.0) |
| `return_base64` | bool | true | Retourner l'image croppée en base64 |
| `image_format` | string | "JPEG" | Format de sortie (JPEG ou PNG) |
| `image_quality` | int | 85 | Qualité JPEG (1-95) |
| `max_size` | int | 1280 | Dimension max pour détection (320-4096) |

**Réponse attendue**:
```json
{
  "detected": true,
  "num_detections": 1,
  "bounding_box": {
    "x": 150,
    "y": 200,
    "width": 300,
    "height": 250,
    "confidence": 0.92
  },
  "cropped_image_base64": "base64_encoded_image_string...",
  "original_width": 1920,
  "original_height": 1080,
  "cropped_width": 300,
  "cropped_height": 250,
  "processing_time_ms": 245.3
}
```

---

## 🎨 Segmentation Endpoint

### 4. Segment Salamander (POST /segment-salamander)

#### Exemple Simple
```bash
curl -X POST http://localhost:8000/segment-salamander \
  -F "file=@imagestest/salamander.jpg"
```

#### Exemple avec Tous les Paramètres
```bash
curl -X POST "http://localhost:8000/segment-salamander?confidence=0.3&return_base64=true&image_format=PNG&max_size=1280" \
  -F "file=@imagestest/salamander.jpg"
```

#### Exemple Rapide (JPEG, pas de base64)
```bash
curl -X POST "http://localhost:8000/segment-salamander?return_base64=false&image_format=JPEG" \
  -F "file=@imagestest/salamander.jpg"
```

#### Sauvegarder le Masque de Segmentation
```bash
curl -X POST "http://localhost:8000/segment-salamander?return_base64=true" \
  -F "file=@imagestest/salamander.jpg" \
  -o segmentation_result.json

# Extraire le masque:
jq -r '.mask_base64' segmentation_result.json | base64 -d > mask.png
```

**Paramètres disponibles**:
| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `file` | File | Required | Image à analyser |
| `confidence` | float | 0.25 | Seuil de confiance (0.0-1.0) |
| `return_base64` | bool | true | Retourner le masque en base64 |
| `image_format` | string | "PNG" | Format de sortie (JPEG ou PNG) |
| `max_size` | int | 1280 | Dimension max pour segmentation |

**Réponse attendue**:
```json
{
  "detected": true,
  "num_detections": 1,
  "mask_base64": "base64_encoded_mask_string...",
  "bounding_box": {
    "x": 150,
    "y": 200,
    "width": 300,
    "height": 250,
    "confidence": 0.92
  },
  "original_width": 1920,
  "original_height": 1080,
  "processing_time_ms": 312.7
}
```

---

## 📊 Model Info Endpoint

### 5. Model Information (GET /model-info)
```bash
curl http://localhost:8000/model-info
```

**Réponse attendue**:
```json
{
  "detection_model": {
    "loaded": true,
    "model_path": "models/detect.pt",
    "type": "YOLOv8 Detection"
  },
  "segmentation_model": {
    "loaded": true,
    "model_path": "models/segment.pt",
    "type": "YOLOv8 Segmentation"
  }
}
```

---

## 🧪 Tests Complets

### Script de Test Rapide
```bash
#!/bin/bash

echo "🧪 Testing Salamander Detection API"
echo "===================================="

# Test 1: Health check
echo -e "\n1️⃣  Testing /health endpoint..."
curl -s http://localhost:8000/health | jq '.'

# Test 2: Root endpoint
echo -e "\n2️⃣  Testing / endpoint..."
curl -s http://localhost:8000/ | jq '.'

# Test 3: Model info
echo -e "\n3️⃣  Testing /model-info endpoint..."
curl -s http://localhost:8000/model-info | jq '.'

# Test 4: Detection (if image exists)
if [ -f "imagestest/salamander.jpg" ]; then
    echo -e "\n4️⃣  Testing /crop-salamander endpoint..."
    curl -s -X POST "http://localhost:8000/crop-salamander?return_base64=false" \
      -F "file=@imagestest/salamander.jpg" | jq '.'
else
    echo -e "\n⚠️  No test image found at imagestest/salamander.jpg"
fi

# Test 5: Segmentation (if image exists)
if [ -f "imagestest/salamander.jpg" ]; then
    echo -e "\n5️⃣  Testing /segment-salamander endpoint..."
    curl -s -X POST "http://localhost:8000/segment-salamander?return_base64=false" \
      -F "file=@imagestest/salamander.jpg" | jq '.'
fi

echo -e "\n✅ Tests completed!"
```

**Sauvegarder comme `test_api.sh`**:
```bash
chmod +x test_api.sh
./test_api.sh
```

---

## 🔧 Tests de Performance

### Test avec Temps de Réponse
```bash
# Detection avec timing
time curl -X POST "http://localhost:8000/crop-salamander?return_base64=false" \
  -F "file=@imagestest/salamander.jpg"

# Segmentation avec timing
time curl -X POST "http://localhost:8000/segment-salamander?return_base64=false" \
  -F "file=@imagestest/salamander.jpg"
```

### Test avec Headers Verbose
```bash
curl -v -X POST "http://localhost:8000/crop-salamander" \
  -F "file=@imagestest/salamander.jpg"
```

### Test de Charge (avec Apache Bench)
```bash
# Installer ab si nécessaire: sudo apt install apache2-utils

# Test avec 100 requêtes, 10 concurrentes
ab -n 100 -c 10 http://localhost:8000/health
```

---

## ❌ Tests d'Erreurs

### Test avec Fichier Invalide
```bash
# Test avec un fichier texte (devrait échouer)
echo "not an image" > test.txt
curl -X POST http://localhost:8000/crop-salamander \
  -F "file=@test.txt"

# Réponse attendue: 400 Bad Request
```

### Test sans Fichier
```bash
curl -X POST http://localhost:8000/crop-salamander

# Réponse attendue: 422 Unprocessable Entity
```

### Test avec Paramètres Invalides
```bash
# Confidence hors limite
curl -X POST "http://localhost:8000/crop-salamander?confidence=2.0" \
  -F "file=@imagestest/salamander.jpg"

# Réponse attendue: 422 Unprocessable Entity
```

---

## 📖 Documentation Interactive

### Accéder à Swagger UI
```bash
# Ouvrir dans le navigateur:
http://localhost:8000/docs
```

### Accéder à ReDoc
```bash
# Ouvrir dans le navigateur:
http://localhost:8000/redoc
```

### Télécharger le Schéma OpenAPI
```bash
curl http://localhost:8000/openapi.json > openapi.json
```

---

## 💡 Conseils d'Utilisation

### Optimisation des Performances

1. **Pour la rapidité** (recommandé pour production):
   ```bash
   ?return_base64=false&image_format=JPEG&image_quality=85&max_size=640
   ```

2. **Pour la qualité** (segmentation précise):
   ```bash
   ?image_format=PNG&max_size=1920&confidence=0.3
   ```

3. **Pour le débogage** (verbose):
   ```bash
   ?return_base64=true&confidence=0.1
   ```

### Tailles d'Image Recommandées

| Use Case | max_size | Temps | Précision |
|----------|----------|-------|-----------|
| Temps réel | 640 | ~100ms | Bonne |
| Standard | 1280 | ~250ms | Très bonne |
| Haute qualité | 1920 | ~500ms | Excellente |
| Maximum | 4096 | ~2s | Maximum |

### Formats d'Image

- **JPEG**: 10-20x plus rapide, idéal pour preview/production
- **PNG**: Sans perte, idéal pour analyse/archivage
