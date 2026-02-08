# Migration Guide: `/embed` & `/verify` API Changes

## Summary

The `/embed` and `/verify` endpoints have been updated to improve salamander identification accuracy. The main changes are:

1. **`/embed`**: Better image preprocessing that preserves the full salamander pattern
2. **`/verify`**: Replaced SIFT keypoint matching with DINOv2 cosine similarity

Both endpoints maintain **backward compatibility** — no breaking changes to the request format or response structure.

---

## `/embed` Endpoint

### What changed

The internal image preprocessing now uses **resize + pad** instead of center-crop. This preserves the full salamander pattern regardless of image aspect ratio, producing more accurate embeddings.

### Request format

**No change.** Same as before:

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

**Re-index existing embeddings.** The new preprocessing produces slightly different embedding values for the same image. If you have embeddings stored in a vector database (pgvector), you should re-compute all existing embeddings using the updated `/embed` endpoint to ensure consistent similarity search results.

---

## `/verify` Endpoint

### What changed

The verification engine now uses **DINOv2 cosine similarity** instead of SIFT keypoint matching. This provides:

- Better accuracy on deformable objects (salamanders change posture)
- Color/pattern-aware matching (SIFT was grayscale-only)
- More consistent and interpretable scores

### Request format

**No change.** Same as before:

```http
POST /verify
Content-Type: multipart/form-data

query: <image file>
candidates: <image file 1>
candidates: <image file 2>
...
```

### Response format

The response structure is **backward compatible**. Existing fields are preserved, one new field is added:

```json
{
  "success": true,
  "message": "Verified against 3 candidates",
  "results": [
    {
      "candidate_index": 0,
      "is_same": true,
      "score": 0.82,
      "confidence": "high",
      "cosine_similarity": 0.82,
      "matches": 0,
      "inliers": 0
    }
  ]
}
```

#### Field changes

| Field | Status | Notes |
|-------|--------|-------|
| `candidate_index` | Unchanged | Index of the candidate image |
| `is_same` | Unchanged | Whether likely the same individual |
| `score` | Unchanged | Similarity score (0-1), now based on cosine similarity |
| `confidence` | Unchanged | `"low"`, `"medium"`, or `"high"` |
| `cosine_similarity` | **New** | Raw cosine similarity between DINOv2 embeddings (-1 to 1) |
| `matches` | Deprecated | Always `0` — was SIFT-specific, kept for backward compat |
| `inliers` | Deprecated | Always `0` — was SIFT-specific, kept for backward compat |

### New thresholds

The decision thresholds are now based on cosine similarity:

| Cosine Similarity | `is_same` | `confidence` |
|-------------------|-----------|--------------|
| >= 0.70 | `true` | `"high"` |
| >= 0.50 | `true` | `"medium"` |
| >= 0.40 | `false` | `"low"` |
| < 0.40 | `false` | `"high"` |

### Action required for frontend

1. **Start using `cosine_similarity`** instead of `matches`/`inliers` for any display or logic. These two fields will be removed in a future version.
2. **`score` values will differ** from previous versions — the scale and distribution have changed. If you have hardcoded thresholds on the frontend (e.g., showing a warning below 0.05), update them to match the new scale (0-1 range based on cosine similarity).
3. **`is_same` and `confidence` work the same way** — you can continue using them as-is for UI decisions.

---

## Quick checklist

- [ ] Re-compute all stored embeddings via `/embed`
- [ ] Replace any usage of `matches` / `inliers` with `cosine_similarity`
- [ ] Update any hardcoded score thresholds to the new cosine similarity scale
- [ ] Verify UI displays work correctly with the new score range (0-1)
- [ ] Remove references to SIFT in any user-facing text
