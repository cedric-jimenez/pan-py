# Migration Guide: `/embed` & `/verify` API Changes

## Summary

The `/embed` and `/verify` endpoints have been updated to improve individual salamander identification accuracy. The previous approach used the DINOv2 CLS token (which only captured species-level similarity — all fire salamanders scored ~0.92) and SIFT keypoint matching (grayscale-only, assumed rigid objects).

The new approach uses **DINOv2 patch tokens** which capture local pattern details (spot arrangement, color boundaries) that distinguish individual salamanders.

Both endpoints maintain **backward compatibility** — no breaking changes to the request format or response structure.

---

## What changed (algorithm)

### Before

| Step | Method | Problem |
|------|--------|---------|
| Embedding | DINOv2 CLS token + CenterCrop | CLS token captures "fire salamander" → all individuals score ~0.92. CenterCrop cuts off pattern edges. |
| Verification | SIFT keypoints + RANSAC homography | Grayscale only (loses color), assumes rigid object (salamanders bend), ad-hoc scoring formula. |

### After

| Step | Method | Improvement |
|------|--------|-------------|
| Embedding | **GeM-pooled DINOv2 patch tokens** + resize/pad | Patch tokens encode local pattern details. GeM pooling emphasizes distinctive patches. Resize+pad preserves full pattern. |
| Verification | **Patch-level mutual nearest neighbor matching** | Compares ~256 local patches between images. Only mutual best matches count, filtering out ambiguous correspondences. Color-aware. |

---

## `/embed` Endpoint

### Request format

**No change.**

```http
POST /embed
Content-Type: multipart/form-data

file: <image file>
```

### Response format

**No change.** Same structure and dimensions:

```json
{
  "success": true,
  "message": "Embedding extracted successfully",
  "embedding": [0.0123, -0.0456, ...],
  "embedding_dim": 384,
  "model": "dinov2_vits14"
}
```

### Action required

**Re-index all stored embeddings.** The new GeM-pooled patch token embeddings are fundamentally different from the old CLS token embeddings. You **must** re-compute all existing embeddings via the updated `/embed` endpoint. Old and new embeddings are not comparable.

---

## `/verify` Endpoint

### Request format

**No change.**

```http
POST /verify
Content-Type: multipart/form-data

query: <image file>
candidates: <image file 1>
candidates: <image file 2>
...
```

### Response format

The response structure is **backward compatible**:

```json
{
  "success": true,
  "message": "Verified against 3 candidates",
  "results": [
    {
      "candidate_index": 0,
      "is_same": true,
      "score": 0.32,
      "confidence": "high",
      "cosine_similarity": 0.78,
      "matches": 0,
      "inliers": 0
    },
    {
      "candidate_index": 2,
      "is_same": false,
      "score": 0.08,
      "confidence": "high",
      "cosine_similarity": 0.61,
      "matches": 0,
      "inliers": 0
    }
  ]
}
```

### Field changes

| Field | Status | Notes |
|-------|--------|-------|
| `candidate_index` | Unchanged | Index of the candidate image |
| `is_same` | Unchanged | Whether likely the same individual |
| `score` | **Changed scale** | Now based on patch matching (0-1). Typical same-individual: 0.25-0.45. Typical different: 0.05-0.15. |
| `confidence` | Unchanged | `"low"`, `"medium"`, or `"high"` |
| `cosine_similarity` | Unchanged | Global embedding cosine similarity (for reference) |
| `matches` | Deprecated | Always `0` — SIFT-specific, kept for backward compat |
| `inliers` | Deprecated | Always `0` — SIFT-specific, kept for backward compat |

### New score scale

**Important:** The `score` field now uses a completely different scale. Previously scores were clustered around 0.91-0.94 for all same-species individuals. Now scores are much more spread out:

| Patch Score | `is_same` | `confidence` | Interpretation |
|-------------|-----------|--------------|----------------|
| >= 0.25 | `true` | `"high"` | Strong pattern match — very likely same individual |
| >= 0.15 | `true` | `"medium"` | Moderate match — possible same individual, different pose |
| >= 0.10 | `false` | `"low"` | Weak match — uncertain, review manually |
| < 0.10 | `false` | `"high"` | No match — confidently different individuals |

### What `score` means now

The score is computed as: `(mutual_match_ratio) × (mean_similarity_of_matches)`

- **Mutual match ratio**: proportion of patches (out of ~256) that are mutual best matches between the two images. Same individual → many mutual matches. Different individual → few.
- **Mean similarity**: average cosine similarity of those mutual matches. Same individual → high similarity. Different → lower.

---

## Action required for frontend

### Must do

1. **Re-compute all stored embeddings** via `/embed` — old embeddings are incompatible
2. **Update any hardcoded score thresholds** — the score scale changed dramatically:
   - Old: all same-species individuals scored 0.91-0.94
   - New: same individual ~0.25-0.45, different individual ~0.05-0.15
3. **Update score display** — if you show the score as a percentage, the raw values are now lower but more meaningful

### Should do

4. **Use `score` (patch matching) as the primary sorting/display metric** — it's the most discriminative
5. **Use `cosine_similarity` as a secondary metric** — useful for quick pre-filtering but less precise
6. **Stop using `matches` and `inliers`** — always 0, will be removed in a future version

### Optional

7. **Show `cosine_similarity` alongside `score`** for debugging — helps understand edge cases

---

## Quick checklist

- [ ] Re-compute all stored embeddings via `/embed`
- [ ] Update hardcoded score thresholds to new scale (0.25/0.15/0.10)
- [ ] Update score display formatting (no longer 0.90+ range)
- [ ] Verify UI displays work correctly with the new score range
- [ ] Remove dependencies on `matches` / `inliers` fields
