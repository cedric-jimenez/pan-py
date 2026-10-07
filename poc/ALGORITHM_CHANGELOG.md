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

## v2 - Masquage du premier plan + normalisation symétrique (2026-06-21)

**Contexte**: Trop de faux positifs en production. Nouveau jeu labellisé propre
dans `docs/images/<idN>/` (14 individus, 49 paires same / 897 paires different
après dédoublonnage). Harnais d'évaluation: `poc/eval_identification.py`.

**Diagnostic**:
1. **Seuils déréglés** (cause principale): l'algo v0 classait `is_same` dès
   score ≥ 0.15, alors que les individus *différents* scorent ~0.30 en moyenne
   → 98 % des paires différentes franchissaient le seuil.
2. **Le fond polluait le matching**: patchs de fond achromatiques (blanc, gris
   150 du segmenter, padding noir) créaient de fausses correspondances entre
   individus différents.

**Changements** (`embedder.py`, `verifier.py`):
- `extract_features()` retourne désormais un masque de premier plan par patch.
  Un patch est *fond* s'il est achromatique (saturation < 28) ET clair
  (luminosité > 110, gris/blanc) ou quasi-noir (< 18, padding). Background-
  agnostique → robuste au gris-150 produit en production par le segmenter.
- `_patch_match_score()` ignore les patchs de fond et normalise le match_ratio
  par le **plus petit** ensemble de patchs premier plan (symétrique) au lieu de
  `len(patches1)`. Repli automatique sur tous les patchs si un masque est vide.
- Seuils recalibrés: high=0.55, medium=0.50 (frontière `is_same`), low=0.40.

**Résultats** (jeu propre, 14 individus):
| Métrique | v0 | v2 (fg_sym) |
|----------|-----|-------------|
| AUC-ROC | 0.794 | **0.882** |
| Gap moyen same-diff | +0.154 | **+0.174** |
| Précision @ seuil 0.50 | — | **80 %** (rappel 33 %) |
| Précision @ seuil 0.55 | — | **~100 %** (rappel 25 %) |

**Limites connues**: rappel modeste (~30 % au seuil retenu) — la frontière
0.40–0.50 est exposée en confiance "low" pour revue humaine. Petit échantillon
(49 paires same): re-calibrer les seuils via le harnais quand le jeu grossit.

**Piste suivante**: fine-tuner DINOv2 sur les individus (gain majeur attendu).
Optionnel: appliquer le masque aussi au pooling GeM de `/embed` pour améliorer
le rappel grossier pgvector (⚠️ nécessite de ré-embedder les photos stockées).

---

## v3 - SIFT + RANSAC (changement de paradigme) (2026-06-21)

**Contexte**: Même après v2 + correction du timeout (warmup DINOv2), un cas réel
restait faux : le vrai individu scorait 0.489 et un autre 0.505 → mauvais
classement. Aucun seuil ne corrige ça (le score ne classe pas le bon en tête).
Recul demandé sur l'algo.

**Diagnostic de fond**: la ré-ID d'individus tachetés est un problème de
**géométrie** (l'agencement des taches = empreinte). DINOv2 encode l'apparence
générique ("une salamandre"), et le matching mutual-NN **ignorait la position**
des patchs → deux individus différents partagent des patchs "tache jaune sur
noir" qui matchent. Indice : les champs `matches`/`inliers` du verifier, prévus
pour une vérif géométrique, n'avaient jamais été implémentés.

**Benchmark** (`poc/benchmark_methods.py`, 14 individus, top-1 retrieval + AUC):

| méthode | top-1 | AUC |
|---------|-------|-----|
| cosine (retrieval) | 63.6 % | 0.786 |
| fg_sym (verifier v2) | 70.5 % | 0.882 |
| combined cos+patch | 81.8 % | 0.872 |
| patch + RANSAC | 84.1 % | 0.907 |
| **SIFT + RANSAC** | **97.7 %** | **0.914** |

**Distribution SIFT** (inlier ratio): même individu médiane 0.31 / max 0.60 ;
individus différents médiane 0.02 / **max 0.06**. → seuil 0.08 = **100 %
précision, ~80 % rappel** (vs 80 %/33 % pour fg_sym). Validé sur le cas de prod :
vrai match 0.40 (60 inliers, is_same/high) vs faux positif 0.03 (5 inliers).

**Changements** (`verifier.py`):
- `SalamanderVerifier` réécrit en **SIFT + Lowe ratio (0.75) + RANSAC homography**.
  Score = `inliers / min(#keypoints)`. Plus aucune dépendance à DINOv2 → `/verify`
  rapide et insensible au warmup.
- Seuils: high=0.15, medium=0.08 (frontière is_same), low=0.05.
- Contrat d'API inchangé ; `matches`/`inliers` enfin renseignés ; `cosine_similarity`
  devient vestigial (0.0) ; param `cosine_threshold` ignoré.

**Limites**: petit jeu, photos d'un individu possiblement de la même session
(re-captures éloignées plus dures → le rappel baissera). Nouveau goulot = le
retrieval pgvector (top-1 63.6 %) : augmenter le top-K côté Pan. Destination
long terme inchangée : metric learning / fine-tuning.

---

## v4 - Seuil is_same 0.08 → 0.06

**Date**: 2026-10-07

**Contexte**: éval bout-en-bout de la chaîne complète (segment → embed → verify)
via `poc/eval_identification_e2e.py`, sur **deux** jeux : `docs/images` (14
individus, croppés) et le corpus terrain brut `Echantillons/.../par-individu`
(5 individus, multi-session). Objectif : chiffrer pourquoi l'identification
paraît approximative sur photos terrain.

**Constat clé**: la frontière `is_same` *opérante* est **medium**, pas high —
`/verify` renvoie `is_same=True` dès medium, et côté Pan `bestExistingMatch`
(`lib/identify-suggestion.ts`) ne filtre que sur `is_same`. Le label high (0.15)
est purement cosmétique. Le vérifieur lui-même est excellent (AUC 0.91–0.96,
**0 faux positif** sur 964 paires d'individus différents, qui plafonnent à 0.06).

**Balayage du seuil is_same** (recall / faux positifs / identification e2e) :

| seuil | docs/images | corpus terrain |
|-------|-------------|----------------|
| 0.05 | 82 % / **3 FP** / 86 % | 64 % / **1 FP** / 85 % |
| **0.06** | 82 % / 0 FP / 86 % | 64 % / 0 FP / **77 %** |
| 0.08 (ancien) | 80 % / 0 FP / 86 % | 55 % / 0 FP / 69 % |
| 0.15 | 67 % / 0 FP / 82 % | 27 % / 0 FP / 46 % |

**Changement** (`verifier.py`): `DEFAULT_MEDIUM_THRESHOLD` 0.08 → **0.06**.
0.06 reste juste au-dessus du plafond des paires-différentes (0.06) → précision
toujours 100 % sur 964 paires, tout en remontant le rappel terrain (identification
e2e 69 % → 77 %). 0.05 est écarté (fait apparaître des faux positifs).

**Limites**: petits jeux (5 et 14 individus) → le rappel a de larges barres
d'erreur ; la marge au-dessus du max diff (0.06) est fine et montera avec un
catalogue plus gros → re-calibrer à mesure que le set labellisé grandit.

---

## Statut actuel

- **Algorithme en production**: v4 (SIFT + RANSAC, seuil is_same 0.06)
- **Performance jeu propre (14 individus)**: top-1 ~97 %, AUC 0.914, 100 % précision @ 0.06
- **Performance terrain (corpus, 5 individus)**: identification e2e 77 %, 0 faux positif
- **Benchmark des alternatives**: `./venv/bin/python poc/benchmark_methods.py`
- **Éval bout-en-bout + balayage de seuil**: `./venv/bin/python poc/eval_identification_e2e.py --dataset <dir> [--segment]`
- **Calibration des seuils (vérifieur seul)**: `./venv/bin/python poc/eval_identification.py`

---
