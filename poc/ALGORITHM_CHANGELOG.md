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

L'algorithme de base (v0) détecte correctement les similarités visuelles, mais:
1. Le dataset de test a des labels incomplets (cluster IMG probable même individu)
2. La différence de pose entre titine-1 et titine-2 impacte le score
3. DINOv2 capture des features "salamandre" génériques, pas les spots individuels

### Pistes d'amélioration futures

1. **Augmenter le dataset** avec plus de paires same/different labelées
2. **Fine-tuner DINOv2** spécifiquement pour les patterns de salamandres
3. **Combiner avec détection de spots** explicite (segmentation des taches jaunes)
4. **Normaliser la pose** avant comparaison

---

## Statut actuel

- **Algorithme en production**: v0 (mutual nearest neighbor)
- **Performance sur dataset actuel**: 21.4% accuracy (mais dataset suspect)
- **Performance estimée sur données propres**: À déterminer avec meilleur ground truth

---
