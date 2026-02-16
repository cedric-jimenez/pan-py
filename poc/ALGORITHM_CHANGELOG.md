# Verification Algorithm Changelog

Suivi des modifications de l'algorithme de vérification DINOv2.

## Métriques de référence

- **Dataset**: 8 images, 28 paires
- **Ground truth**: titine-1 ↔ titine-2 = SAME, reste = DIFFERENT
- **Objectif**: Maximiser accuracy, gap positif entre same/different

---

## v0 - Baseline (Mutual Nearest Neighbor)

**Date**: 2025-02-15

**Algorithme**:
- Mutual nearest neighbor matching sur patch tokens DINOv2
- Score = (n_mutual_matches / n_patches) × mean_similarity

**Code** (`_patch_match_score`):
```python
sim_matrix = patches1 @ patches2.T
nn_1to2 = sim_matrix.argmax(axis=1)
nn_2to1 = sim_matrix.argmax(axis=0)

mutual_sims = []
for i, j in enumerate(nn_1to2):
    if nn_2to1[j] == i:
        mutual_sims.append(sim_matrix[i, j])

match_ratio = len(mutual_sims) / n_patches
mean_sim = np.mean(mutual_sims)
return match_ratio * mean_sim
```

**Seuils**: high=0.25, medium=0.15, low=0.10

**Résultats**:
| Métrique | Valeur |
|----------|--------|
| Accuracy | 21.4% |
| Precision | 4.3% |
| Recall | 100.0% |
| F1 | 0.08 |
| TP / TN / FP / FN | 1 / 5 / 22 / 0 |

**Distribution des scores**:
| Type | Min | Max | Mean |
|------|-----|-----|------|
| Same | 0.417 | 0.417 | 0.417 |
| Different | 0.118 | 0.743 | 0.393 |
| **Gap** | **-0.326** (OVERLAP) |

**Problèmes identifiés**:
1. Gap inversé - les différents individus scorent plus haut
2. Paire titine (même individu) score seulement 0.417
3. Paire IMG_195956 vs IMG_200016 score 0.743 (probablement même individu non labelé)

---

## v1 - Tentatives d'amélioration (2025-02-15)

### Tentative 1: Lowe's Ratio Test

**Hypothèse**: Filtrer les matches ambigus améliorerait la discrimination.

**Résultat**: **ÉCHEC** - Ratio test inapplicable aux embeddings DINOv2.

**Analyse**:
- Les embeddings DINOv2 ont des similarités très élevées (0.7-0.98) pour tous les patches
- Le ratio 2nd_best/best est toujours > 0.9 (contre ~0.7 pour SIFT)
- Avec ratio_thresh=0.8, **0% des patches** passent le test
- Les patches DINOv2 capturent des features sémantiques (salamandre), pas locales (spots)

### Tentative 2: P10 (10ème percentile)

**Hypothèse**: Le minimum (ou p10) des matches discrimine mieux que la moyenne.

**Résultat**: Amélioration modeste (75% accuracy) mais overlap persiste.

**Distribution p10**:
| Paire | p10 | Type |
|-------|-----|------|
| titine-1 vs titine-2 | 0.860 | SAME |
| IMG_195956 vs IMG_200016 | 0.889 | "DIFF" (suspect) |
| titine-1 vs IMG_200058 | 0.709 | DIFF |

### Tentative 3: Distinctiveness Score

**Hypothèse**: Mesurer l'écart entre meilleur match et moyenne améliorerait.

**Résultat**: Accuracy dégradée à 64.3%.

---

## Analyse du Dataset

### Problème de Ground Truth

L'analyse révèle que le ground truth est probablement **incomplet**:

**Evidence**: Les images IMG_195931, IMG_195941, IMG_195956, IMG_200016 forment un cluster avec des scores élevés (0.5-0.74), suggérant qu'elles sont du **même individu** (photos prises dans la même minute: 19:59:31 à 20:00:16).

**Matrice de similarité (v0)**:
```
             195851  195931  195941  195956  200016  200058  titine1 titine2
195851        1.000   0.407   0.331   0.435   0.481   0.168   0.449   0.403
195931        0.407   1.000   0.555   0.541   0.531   0.145   0.594   0.372
195941        0.331   0.555   1.000   0.498   0.421   0.149   0.471   0.329
195956        0.435   0.541   0.498   1.000   0.743   0.127   0.628   0.418
200016        0.481   0.531   0.421   0.743   1.000   0.147   0.589   0.382
200058        0.168   0.145   0.149   0.127   0.147   1.000   0.118   0.221
titine1       0.449   0.594   0.471   0.628   0.589   0.118   1.000   0.417
titine2       0.403   0.372   0.329   0.418   0.382   0.221   0.417   1.000
```

### Observations clés

1. **IMG_200058** est clairement différent (scores 0.12-0.22 avec tous)
2. **IMG_195956 vs IMG_200016**: Score max 0.743 → très probablement même individu
3. **titine-1 vs titine-2**: Score 0.417 → même individu mais poses différentes
4. **titine-1 vs IMG_195956**: Score 0.628 → différents mais l'algo ne discrimine pas

### Conclusion

Le dataset est **correct**. L'algorithme v0 ne discrimine pas les individus :
1. La différence de pose entre titine-1 et titine-2 impacte le score
2. DINOv2 capture des features "salamandre" génériques, pas les spots individuels
3. Le score `match_ratio × mean_similarity` ne pénalise pas les matches sémantiques non-discriminants

---

## v2 - Proposition d'amélioration

**Date**: 2026-02-16

### Diagnostic du problème

Le problème fondamental de v0 est que **tous les patches de salamandre se ressemblent** dans l'espace DINOv2. Un patch "patte" de l'individu A matche un patch "patte" de l'individu B avec une similarité cosinus de 0.7-0.98. Le MNN trouve donc de nombreux matches mutuels même entre individus différents, et le score `match_ratio × mean_similarity` est élevé pour tous.

Ce qui distingue deux images du **même** individu vs deux individus **différents**, c'est :
1. **La cohérence spatiale** : si le patch (3,5) matche le patch (3,6) et le patch (7,2) matche le patch (7,3), les vecteurs de déplacement sont cohérents → même individu vu sous un angle légèrement différent. Pour des individus différents, les matches MNN seront spatialement aléatoires.
2. **Les "hub" patches** : certains patches génériques (corps noir, fond) sont les plus proches voisins de beaucoup d'autres patches. Ils créent des faux matches qui gonflent les scores.
3. **Le padding** : les images sont redimensionnées et paddées à 224×224 avec du noir. Les patches de padding ajoutent du bruit au matching.

### Améliorations proposées (par ordre de priorité)

#### A1 - Filtrage des patches de padding

**Problème** : `_ResizePad` ajoute du padding noir (0,0,0) pour obtenir un carré 224×224. Les patches couvrant du padding sont des vecteurs quasi-identiques qui se matchent trivialement entre toutes les images, gonflant artificiellement `match_ratio`.

**Solution** : Avant le matching, retirer les patches dont la norme (avant L2-normalisation) est inférieure à un seuil. Les patches de padding ont une norme très faible car ils représentent des régions uniformément noires après normalisation ImageNet.

**Impact attendu** : Faible mais nécessaire. Élimine du bruit, réduit le nombre de patches inutiles (potentiellement 30-50% selon le ratio d'aspect de l'image).

**Implémentation** : Modifier `extract_features()` dans `embedder.py` pour retourner aussi un masque de validité, ou filtrer les patches dans `_patch_match_score()`.

#### A2 - CSLS (Cross-domain Similarity Local Scaling)

**Problème** : Certains patches sont des "hubs" — ils sont le plus proche voisin de beaucoup d'autres patches. Un patch générique "corps noir de salamandre" match bien avec tout. Le MNN filtre partiellement ce problème mais pas assez.

**Solution** : Remplacer la similarité cosinus brute par CSLS :

```
CSLS(x, y) = 2·cos(x, y) - mean_kNN(x) - mean_kNN(y)
```

Où `mean_kNN(x)` = moyenne des similarités des k plus proches voisins de x dans l'autre ensemble de patches.

**Principe** : Si un patch x a une similarité moyenne élevée avec beaucoup de patches (c'est un hub), sa similarité CSLS avec n'importe quel patch y sera pénalisée. Seuls les matches **spécifiques** (haute similarité entre x et y, mais x et y ne matchent pas bien avec le reste) obtiennent un score CSLS élevé.

**Paramètre** : k=10 (standard dans la littérature, à valider empiriquement).

**Impact attendu** : Modéré. Devrait réduire les scores des paires "different" sans trop affecter les paires "same" car les vrais matches ont une spécificité que les hubs n'ont pas.

#### A3 - Score de cohérence spatiale

**Problème** : C'est le problème central. Le MNN matche un patch "patte avant gauche" de l'image 1 avec un patch "patte arrière droite" de l'image 2. Le score de similarité est élevé (c'est une patte dans les deux cas), mais la position spatiale est incohérente.

Pour le **même** individu : les matches MNN devraient former une transformation spatiale cohérente (translation, légère rotation, léger changement d'échelle). Le patch en haut-à-gauche de l'image 1 devrait matcher un patch en haut-à-gauche de l'image 2 (à une transformation près).

Pour des individus **différents** : les matches MNN seront spatialement dispersés, les vecteurs de déplacement pointant dans toutes les directions.

**Solution** : Après le MNN, calculer un score de cohérence spatiale :

1. Pour chaque match mutuel (i, j), calculer le vecteur de déplacement `d = pos(j) - pos(i)` où `pos()` retourne les coordonnées (row, col) du patch dans la grille 16×16.
2. Calculer le vecteur de déplacement médian `d_med`.
3. Pour chaque match, calculer la distance au déplacement médian : `||d - d_med||`.
4. Compter le nombre d'**inliers** : matches dont la distance au médian est < seuil (ex: 2 patches).
5. **Score spatial** = `n_inliers / n_mutual_matches`.

**Score final combiné** :
```
score = spatial_consistency × mean_similarity_of_inliers
```

Ce score remplace le `match_ratio × mean_similarity` de v0. Il ne récompense plus le nombre brut de matches, mais la **qualité géométrique** des matches.

**Analogie** : C'est l'équivalent de RANSAC pour les keypoints SIFT, mais adapté à la grille régulière de patches DINOv2. La grille régulière simplifie le problème car on n'a pas besoin d'estimer une homographie complète — un simple modèle de translation suffit.

**Impact attendu** : Élevé. C'est l'amélioration la plus prometteuse. Pour la paire titine-1 vs titine-2 (même individu), les matches devraient être spatialement cohérents. Pour IMG_195956 vs IMG_200016 (différents individus mais même espèce), les matches devraient être spatialement aléatoires même si les similarités patch-à-patch sont élevées.

#### A4 - Pondération par distinctivité des patches (optionnel, après validation A1-A3)

**Problème** : Tous les patches sont traités de la même façon. Un patch "tache jaune unique" devrait compter plus qu'un patch "corps noir uniforme".

**Solution** : Pondérer chaque patch par sa distance au centroïde moyen de tous les patches de l'image. Les patches distinctifs (taches jaunes, motifs particuliers) seront plus éloignés du centroïde que les patches génériques (corps noir).

**Impact attendu** : Complémentaire aux autres améliorations. Permet de ne pas diluer le signal des patches discriminants dans la masse de patches non-informatifs.

### Plan d'implémentation

L'ordre est important — chaque étape se base sur la précédente :

| Étape | Amélioration | Fichier modifié | Complexité |
|-------|-------------|-----------------|------------|
| 1 | A1 - Filtrage padding | `embedder.py` (retourner normes) + `verifier.py` (filtrer) | Faible |
| 2 | A2 - CSLS | `verifier.py` (`_patch_match_score`) | Faible |
| 3 | A3 - Cohérence spatiale | `verifier.py` (`_patch_match_score`) | Moyenne |
| 4 | Recalibrer les seuils | `verifier.py` (constantes) | Faible |
| 5 | A4 - Pondération distinctivité | `verifier.py` | Faible |

Chaque étape doit être évaluée sur le dataset de 28 paires avant de passer à la suivante.

### Résultats attendus

| Métrique | v0 (actuel) | v2 (cible) |
|----------|-------------|------------|
| Accuracy | 21.4% | > 85% |
| Gap same/different | -0.326 (inversé) | > 0 (positif) |
| Score same (titine) | 0.417 | Le plus haut ou parmi les plus hauts |
| Score different max | 0.743 | < score same |

### Risques et limites

1. **Pose extrême** : Si deux photos du même individu sont prises de côtés opposés (ventre vs dos), la cohérence spatiale sera faible. Mitigation : la v2 sera meilleure que v0 dans tous les cas, et la normalisation de pose reste une piste future.
2. **Grille fixe 16×16** : La résolution de la grille de patches limite la précision spatiale. Un décalage d'un patch = 14 pixels. Pour des petites salamandres dans l'image, ça peut être significatif.
3. **Seuils k et distance** : Les paramètres de CSLS (k) et de cohérence spatiale (seuil inlier) devront être ajustés empiriquement sur le dataset.

---

## v2 - Résultats d'implémentation

**Date** : 2026-02-16

### Implémentation effective

Les trois améliorations A1, A2, A3 ont été implémentées. L'implémentation de A1 a dû être adaptée par rapport au plan initial :

**A1 — Filtrage padding (adapté)** : Le plan prévoyait d'utiliser les normes L2 des patch tokens avant normalisation pour distinguer padding/contenu. En pratique, le `LayerNorm` interne de DINOv2 égalise toutes les normes (~45-56), rendant cette approche inopérante. Remplacé par un **masque géométrique** calculé depuis les dimensions de l'image et le padding déterministe de `_ResizePad` (ratio de contenu par patch, seuil à 0.5).

**A2 — CSLS** : Implémenté tel que prévu. `CSLS(x,y) = 2·cos(x,y) - mean_kNN(x) - mean_kNN(y)` avec k=10 et `np.partition` pour les k-NN en O(n).

**A3 — Cohérence spatiale** : Implémenté tel que prévu. Modèle de déplacement médian, seuil d'inlier à 2.0 patches.

### Résultats

| Métrique | v0 | v2 |
|----------|-----|-----|
| Accuracy | 21.4% | 85.7% |
| Spécificité | 18.5% | 88.9% |
| Recall | 100% | 0% |
| TP / TN / FP / FN | 1 / 5 / 22 / 0 | 0 / 24 / 3 / 1 |

**Distribution des scores v2** :
| Type | Min | Max | Mean |
|------|-----|-----|------|
| Same (titine) | 0.101 | 0.101 | 0.101 |
| Different | 0.036 | 0.206 | 0.095 |
| **Gap** | **-0.105** (OVERLAP) |

**Matrice de similarité v2** :
```
             195851  195931  195941  195956  200016  200058  titine1 titine2
195851        1.000   0.097   0.051   0.050   0.072   0.036   0.074   0.078
195931        0.097   1.000   0.151   0.139   0.142   0.040   0.149   0.084
195941        0.051   0.151   1.000   0.133   0.125   0.064   0.144   0.088
195956        0.050   0.139   0.133   1.000   0.206   0.038   0.129   0.076
200016        0.072   0.142   0.125   0.206   1.000   0.049   0.158   0.079
200058        0.036   0.040   0.064   0.038   0.049   1.000   0.049   0.067
titine1       0.074   0.149   0.144   0.129   0.158   0.049   1.000   0.101
titine2       0.078   0.084   0.088   0.076   0.079   0.067   0.101   1.000
```

**Top résultats détaillés** :
| # | Image 1 | Image 2 | Score | Matches | Inliers | Préd | Réel |
|---|---------|---------|-------|---------|---------|------|------|
| 1 | 195956 | 200016 | 0.206 | 37 | 36 | SAME | DIFF |
| 2 | 200016 | titine-1 | 0.158 | 29 | 23 | SAME | DIFF |
| 3 | 195931 | 195941 | 0.151 | 33 | 23 | SAME | DIFF |
| … | | | | | | | |
| 11 | titine-1 | titine-2 | 0.101 | 37 | 22 | DIFF | SAME |

### Analyse

**Progrès** :
- L'accuracy passe de 21.4% à 85.7% (+64 points)
- Les scores sont beaucoup plus bas et différenciés (0.03-0.21 vs 0.12-0.74 en v0)
- IMG_200058 est correctement isolé (scores 0.036-0.067)
- Le filtrage padding réduit de 256 à ~64 patches utiles pour les images étroites

**Limitations persistantes** :
1. **Images trop étroites** : Les images segmentées ont un ratio d'aspect ~0.25, ne laissant que **64 patches de contenu sur 256** (25%). Avec si peu de patches, la discrimination inter-individus est faible.
2. **Cohérence spatiale trompeuse** : Les paires d'images de même forme (salamandres étroites verticales) obtiennent une forte cohérence spatiale même entre individus différents (36/37 inliers pour 195956↔200016), car les patches "tête", "corps", "queue" se retrouvent aux mêmes positions relatives.
3. **Poses différentes pour titine** : titine-1 (110×400) et titine-2 (218×400) ont des cadrages très différents. Le modèle de déplacement médian (translation rigide) n'est pas adapté → seulement 22/37 inliers (59%).
4. **Gap négatif** : La paire same (0.101) ne se sépare pas des paires different (max 0.206).

### Pistes d'amélioration future

1. **Images de meilleure qualité** : Des crops plus larges (ratio > 0.5) augmenteraient significativement le nombre de patches utiles.
2. **Modèle géométrique flexible** : Remplacer le déplacement médian par une transformation affine ou similitude pour mieux gérer les changements de pose/échelle.
3. **A4 — Pondération par distinctivité** : Donner plus de poids aux patches discriminants (taches jaunes) vs génériques (corps noir).
4. **Fine-tuning DINOv2** : Entraîner les dernières couches sur des paires salamandre pour des features plus discriminantes au niveau individuel.

---

## Statut actuel

- **Algorithme en production** : v2 (CSLS + cohérence spatiale + filtrage padding géométrique)
- **Performance** : 85.7% accuracy — amélioration significative mais gap négatif persiste
- **Limitation principale** : Images segmentées trop étroites (64 patches utiles) et poses trop différentes entre photos du même individu
- **Prochaine étape** : Améliorer la qualité des images (crops plus larges) ou implémenter un modèle géométrique plus flexible

---
