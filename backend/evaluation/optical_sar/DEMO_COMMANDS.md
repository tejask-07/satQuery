# Optical–SAR Team Demo & Verification Commands

This cheat sheet provides practical, copy-pasteable commands for running, testing, and evaluating the Optical–SAR backend.

---

## 1. Backend Startup

From the repository root or `backend/` directory:

```powershell
# In PowerShell / terminal:
cd backend
python -m uvicorn app.main:app --reload --port 8000
```

The API docs will be available at: `http://127.0.0.1:8000/docs`.

---

## 2. Automatic Optical–SAR API Test (Live S2 + S1 Acquisition)

Test the complete end-to-end flow where the agent automatically discovers, downloads, aligns, and interprets real Sentinel-2 and Sentinel-1 imagery from Microsoft Planetary Computer.

### Option A: PowerShell (`Invoke-RestMethod`)

```powershell
$body = @{
    query = "Use the optical and SAR images together to identify built-up areas."
    aoi = @(13.0, 48.0, 13.02, 48.02)
    time_start = "2021-06-25"
    time_end = "2021-06-28"
} | ConvertTo-Json

$response = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/api/query" `
    -ContentType "application/json" `
    -Body $body

# Inspect output
Write-Host "Status:" $response.status
Write-Host "Temporal Delta (days):" $response.statistics.optical_sar_pair.temporal_delta_days
Write-Host "Optical Scene:" $response.statistics.optical_sar_pair.optical_item_id
Write-Host "SAR Scene:" $response.statistics.optical_sar_pair.sar_item_id
Write-Host "`nExecution Trace:"
$response.execution_trace | ForEach-Object { Write-Host " - $_" }
Write-Host "`nAnswer Preview:"
Write-Host $response.answer.Substring(0, [Math]::Min(300, $response.answer.Length)) "..."
```

### Option B: cURL

```bash
curl -X POST "http://127.0.0.1:8000/api/query" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "Use the optical and SAR images together to identify built-up areas.",
       "aoi": [13.0, 48.0, 13.02, 48.02],
       "time_start": "2021-06-25",
       "time_end": "2021-06-28"
     }'
```

### Option C: Python (`requests`)

```python
import requests

payload = {
    "query": "Use the optical and SAR images together to identify built-up areas.",
    "aoi": [13.0, 48.0, 13.02, 48.02],
    "time_start": "2021-06-25",
    "time_end": "2021-06-28",
}

res = requests.post("http://127.0.0.1:8000/api/query", json=payload)
data = res.json()

print("Status:", data.get("status"))
print("Pair Provenance:", data.get("statistics", {}).get("optical_sar_pair"))
print("Trace:", data.get("execution_trace"))
```

---

## 3. Direct Multipart File Upload Test (Mode A)

To test user-provided Optical + SAR GeoTIFF uploads:

```powershell
# From repository root:
$optFile = "backend/evaluation/optical_sar/cases/austria_s2_optical.tif"
$sarFile = "backend/evaluation/optical_sar/cases/austria_s1_sar.tif"

curl -X POST "http://127.0.0.1:8000/api/upload/optical-sar" `
     -F "optical_image=@$optFile;type=image/tiff" `
     -F "sar_image=@$sarFile;type=image/tiff" `
     -F "query=Analyze built-up structures and vegetation using optical and SAR."
```

---

## 4. Run Benchmark Evaluation & Modality Comparison

To execute the standardized evaluation suite on all 7 real remote-sensing cases:

```powershell
# Run full evaluation with controlled modality ablation comparison
python backend/evaluation/optical_sar_eval.py `
    --manifest backend/evaluation/optical_sar/manifest.json `
    --output-dir backend/evaluation/optical_sar/ `
    --run-comparison
```

Outputs generated:
- `results.jsonl`: Structured per-case alignment and reasoning metrics.
- `ablation_comparison.jsonl`: Comparative outputs between Optical-only and Optical+SAR.
- `human_review.csv`: Reviewer template for manual qualitative assessment.

---

## 5. Full Person 2 Automated Test Regression

To run all automated test suites covering Optical-SAR, Sentinel-1, Pairing, VLM, and Remote Sensing:

```powershell
python -m pytest `
  backend/tests/test_api_optical_sar_auto.py `
  backend/tests/test_optical_sar_pairing.py `
  backend/tests/test_sentinel1_provider.py `
  backend/app/remote_sensing/multimodal/test_optical_sar.py `
  backend/app/vlm/test_optical_sar.py `
  backend/app/agent/test_optical_sar.py `
  backend/tests/test_api_optical_sar.py `
  backend/tests/test_modality_integrity.py `
  backend/tests/test_optical_sar_evaluation.py `
  backend/tests/test_remote_sensing.py `
  -v
```
