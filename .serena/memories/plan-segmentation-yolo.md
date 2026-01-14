# Plan : Segmentation YOLO pour crop précis des salamandres

## Contexte

**Situation actuelle** :
- Modèle YOLO de détection (`crop.pt`) entraîné avec des bounding boxes
- Crop rectangulaire via `image.crop((x1, y1, x2, y2))`
- Le fond (feuilles, terre, pierres) reste dans l'image croppée

**Objectif** :
- Passer de la **détection** (bbox) à la **segmentation d'instance** (contour précis)
- Obtenir un masque pixel-parfait de la salamandre
- Éliminer complètement le bruit de fond

## Architecture API

**Routes distinctes** :
- `POST /crop-salamander` → Détection bbox + crop rectangulaire (existant, inchangé)
- `POST /segment-salamander` → Segmentation + masque précis (nouveau)

**Modèles** :
- `models/crop.pt` → Détection (existant, conservé)
- `models/segment.pt` → Segmentation (à créer)

## Phase 1 : Roboflow - Création du dataset de segmentation

1. Créer un nouveau projet Roboflow de type **Instance Segmentation**
2. Importer les images existantes (50-200 images)
3. Annoter avec **Smart Polygon** (clic sur salamandre → contour auto)
4. Export format **YOLOv12 Segmentation (format compatible YOLO11/v8)**

## Phase 2 : Entraînement sur Kaggle

### Préparation
1. **Roboflow** : Exporter le dataset → "Download zip to computer" → format YOLOv8
2. **Kaggle** : Créer un nouveau Dataset et uploader le zip extrait
3. **Notebook** : Ajouter le dataset comme input

### Code notebook

```python
# Notebook Kaggle
!pip install ultralytics

from ultralytics import YOLO

# Le dataset est dans /kaggle/input/votre-dataset-name/
# Vérifier la structure
!ls /kaggle/input/salamander-segmentation/

# Charger le modèle pré-entraîné segmentation
model = YOLO('yolo11s-seg.pt')

# Entraîner sur le dataset local
model.train(
    data='/kaggle/input/salamander-segmentation/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='salamander_seg'
)

# Le modèle entraîné sera dans:
# /kaggle/working/runs/segment/salamander_seg/weights/best.pt
```

### Récupérer le modèle
- Dans Kaggle : Output → Download `best.pt`
- Renommer en `segment.pt`
- Placer dans `models/segment.pt`

## Phase 3 : Code Python

### app/detection.py - Nouvelle classe SalamanderSegmenter

```python
class SalamanderSegmenter:
    """Salamander segmenter using YOLO-seg."""

    def __init__(self, model_path: str = "models/segment.pt"):
        self.model_path = Path(model_path)
        self.model = None
        self.load_model()

    def segment(self, image: Image.Image, conf_threshold: float = 0.25,
                bg_color: tuple = (150, 150, 150)):
        """Segment salamander and return masked image."""
        results = self.model(image, conf=conf_threshold, verbose=False)

        if len(results) == 0 or results[0].masks is None:
            return False, None

        # Meilleur masque
        masks = results[0].masks
        boxes = results[0].boxes
        best_idx = np.argmax(boxes.conf.cpu().numpy())

        mask = masks.data[best_idx].cpu().numpy()
        bbox = boxes.xyxy[best_idx].cpu().numpy()

        # Appliquer masque
        cropped = self._apply_mask(image, mask, bbox, bg_color)

        return True, {
            "mask": mask,
            "bbox": bbox,
            "confidence": float(boxes.conf[best_idx]),
            "segmented_image": cropped
        }

    def _apply_mask(self, image, mask, bbox, bg_color):
        img_array = np.array(image)
        mask_resized = cv2.resize(mask, (image.width, image.height))
        mask_bool = mask_resized > 0.5

        result = np.ones_like(img_array) * np.array(bg_color, dtype=np.uint8)
        result[mask_bool] = img_array[mask_bool]

        # Crop sur bbox
        x1, y1, x2, y2 = map(int, bbox)
        cropped = Image.fromarray(result).crop((x1, y1, x2, y2))
        return cropped
```

### app/main.py - Nouvelle route /segment-salamander

```python
@app.post("/segment-salamander")
async def segment_salamander(
    file: UploadFile = File(...),
    confidence: float = Query(0.25),
    background: str = Query("gray", regex="^(gray|white|black)$"),
):
    """Segment salamander with precise mask."""
    bg_colors = {
        "gray": (150, 150, 150),
        "white": (255, 255, 255),
        "black": (0, 0, 0)
    }
    # ... implementation
```

## Fichiers à modifier

| Fichier | Action |
|---------|--------|
| `models/segment.pt` | Nouveau modèle segmentation |
| `app/detection.py` | Ajouter classe `SalamanderSegmenter` |
| `app/main.py` | Ajouter route `/segment-salamander` |
| `app/models.py` | Ajouter `SegmentationResponse` |

## Décisions utilisateur

- **Roboflow** : Nouveau projet de segmentation (pas conversion)
- **Dataset** : 50-200 images avec Smart Polygon
- **Fond** : Gris neutre (150, 150, 150) par défaut - validé par tests (+0.95%)
- **Routes API** : Séparées - conserver `/crop-salamander` et ajouter `/segment-salamander`

## Tests de validation

```bash
# Test crop (existant)
curl -X POST "http://localhost:8000/crop-salamander" -F "file=@image.jpg"

# Test segmentation (nouveau)
curl -X POST "http://localhost:8000/segment-salamander" -F "file=@image.jpg"

# Comparer scores similarité embeddings
# Baseline original ViT-DINO : 0.7739
# Objectif avec segmentation : > 0.78
```

## Contexte tests précédents (imagestest/)

Tests de masques couleur HSV réalisés :
- Meilleure config trouvée : fond gris (150,150,150) → score 0.7812 (+0.95% vs original)
- La segmentation ML devrait donner de meilleurs résultats que le masque couleur
