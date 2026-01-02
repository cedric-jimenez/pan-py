# 🦎 Salamander Detection API

API REST Python avec FastAPI pour détecter et cropper des salamandres dans des images en utilisant un modèle YOLO custom.

## 🚀 Fonctionnalités

- **Détection automatique** de salamandres avec YOLO
- **Cropping intelligent** centré sur la salamandre détectée
- **API REST** simple et rapide avec FastAPI
- **Support CORS** pour intégration avec Next.js/Vercel
- **Déploiement facile** sur Railway
- **Response en base64** pour faciliter l'utilisation côté frontend

## 📋 Prérequis

- Python 3.11+
- Un modèle YOLO entraîné (fichier `.pt`)
- Docker (pour le déploiement)

## 🛠️ Installation locale

### 1. Cloner le repo

```bash
git clone https://github.com/cedric-jimenez/pan-py.git
cd pan-py
```

### 2. Créer un environnement virtuel

```bash
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Ajouter votre modèle YOLO

Placez votre fichier de modèle YOLO (`.pt`) dans le dossier `models/` :

```bash
cp /chemin/vers/votre/modele.pt models/best.pt
```

### 5. Lancer l'API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

L'API sera accessible sur `http://localhost:8000`

## 📡 Utilisation de l'API

### Endpoints disponibles

#### `GET /` ou `GET /health`
Vérifier l'état de l'API et du modèle

```bash
curl http://localhost:8000/health
```

Response :
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "0.1.0"
}
```

#### `POST /crop-salamander`
Détecter et cropper une salamandre

**Paramètres :**
- `file` (required) : Image file (multipart/form-data)
- `confidence` (optional) : Seuil de confiance (0.0 à 1.0, défaut : 0.25)
- `return_base64` (optional) : Retourner l'image en base64 (défaut : true)

**Exemple avec curl :**

```bash
curl -X POST "http://localhost:8000/crop-salamander?confidence=0.3" \
  -F "file=@salamander.jpg"
```

**Exemple avec Python :**

```python
import requests

url = "http://localhost:8000/crop-salamander"
files = {"file": open("salamander.jpg", "rb")}
params = {"confidence": 0.3, "return_base64": True}

response = requests.post(url, files=files, params=params)
result = response.json()

if result["detected"]:
    print(f"Salamandre détectée avec {result['bounding_box']['confidence']:.2%} de confiance")
    # L'image croppée est dans result["cropped_image"] en base64
else:
    print("Aucune salamandre détectée")
```

**Exemple avec JavaScript/TypeScript (Next.js) :**

```typescript
async function detectSalamander(file: File) {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch('https://your-api.railway.app/crop-salamander?confidence=0.3', {
    method: 'POST',
    body: formData,
  });

  const result = await response.json();

  if (result.detected) {
    // Afficher l'image croppée
    const imgSrc = `data:image/png;base64,${result.cropped_image}`;
    // Utiliser imgSrc dans un <img> tag
  }

  return result;
}
```

**Response :**

```json
{
  "success": true,
  "message": "Salamander detected and cropped successfully",
  "detected": true,
  "bounding_box": {
    "x1": 145.2,
    "y1": 203.7,
    "x2": 456.8,
    "y2": 512.3,
    "confidence": 0.87
  },
  "cropped_image": "iVBORw0KGgoAAAANSUhEUgAA...",
  "original_width": 1920,
  "original_height": 1080
}
```

## 🐳 Déploiement sur Railway

### 1. Préparer votre modèle

Assurez-vous que votre fichier modèle YOLO est bien dans `models/best.pt` et est commit dans le repo (ou ajoutez-le via une autre méthode de storage pour les gros fichiers).

### 2. Déployer sur Railway

1. Créez un compte sur [Railway](https://railway.app/)
2. Créez un nouveau projet
3. Connectez votre repo GitHub
4. Railway détectera automatiquement le `Dockerfile` et le `railway.toml`
5. Déployez !

### 3. Configuration des variables d'environnement (optionnel)

Dans Railway, vous pouvez configurer :

- `YOLO_MODEL_PATH` : Chemin vers le fichier modèle (défaut : `models/best.pt`)
- `ALLOWED_ORIGINS` : Origins autorisées pour CORS (défaut : `*`)

### 4. Obtenir l'URL de votre API

Railway vous donnera une URL publique comme : `https://your-api.railway.app`

## 🔗 Intégration avec Next.js/Vercel

Créez un service dans votre app Next.js :

```typescript
// lib/salamanderApi.ts
const API_URL = process.env.NEXT_PUBLIC_SALAMANDER_API_URL || 'http://localhost:8000';

export async function cropSalamander(file: File, confidence: number = 0.25) {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_URL}/crop-salamander?confidence=${confidence}`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error('Failed to process image');
  }

  return response.json();
}
```

Dans votre `.env.local` (Next.js) :

```bash
NEXT_PUBLIC_SALAMANDER_API_URL=https://your-api.railway.app
```

## 📁 Structure du projet

```
pan-py/
├── app/
│   ├── __init__.py          # Package initialization
│   ├── main.py              # FastAPI application
│   ├── models.py            # Pydantic models
│   ├── detection.py         # YOLO detection logic
│   └── utils.py             # Utility functions
├── models/
│   └── best.pt              # YOLO model (à ajouter)
├── tests/                   # Tests unitaires (à venir)
├── requirements.txt         # Python dependencies
├── Dockerfile               # Docker configuration
├── railway.toml             # Railway configuration
├── .dockerignore           # Docker ignore rules
├── .gitignore              # Git ignore rules
└── README.md               # Ce fichier
```

## 🧪 Tests

Pour tester l'API localement :

```bash
# Tester le health check
curl http://localhost:8000/health

# Tester la détection avec une image
curl -X POST "http://localhost:8000/crop-salamander" \
  -F "file=@test_image.jpg" \
  -o response.json

# Voir la documentation interactive
# Ouvrir http://localhost:8000/docs dans votre navigateur
```

## 📊 Documentation API interactive

FastAPI génère automatiquement une documentation interactive :

- **Swagger UI** : `http://localhost:8000/docs`
- **ReDoc** : `http://localhost:8000/redoc`

## 🐛 Dépannage

### Le modèle ne se charge pas

- Vérifiez que le fichier `models/best.pt` existe
- Vérifiez les logs : le chemin du modèle doit être affiché au démarrage
- Testez l'endpoint `/model-info` pour voir l'état du modèle

### Erreur de mémoire sur Railway

- Railway free tier a des limitations de RAM
- Considérez optimiser votre modèle YOLO ou passer à un plan payant

### CORS errors depuis Next.js

- Vérifiez que `ALLOWED_ORIGINS` inclut votre domaine Vercel
- En développement, utilisez `ALLOWED_ORIGINS=*`

## 🚀 Améliorations futures

- [ ] Support de batch processing (plusieurs images)
- [ ] Cache des résultats
- [ ] Support de différents formats de sortie (JPEG, WebP)
- [ ] Webhooks pour processing asynchrone
- [ ] Métriques et monitoring
- [ ] Tests unitaires et d'intégration

## 📝 License

MIT

## 👤 Auteur

Cédric Jimenez

## 🤝 Contributing

Les contributions sont les bienvenues ! N'hésitez pas à ouvrir une issue ou un PR.
