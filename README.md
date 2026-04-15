git# Aegis Earth - NEO Defense System

## Local Development
```bash
pip install -r requirements.txt
start.bat  # Backend @ localhost:8000
# Open Frontend/predicton.html
```

## Vercel Deployment
1. `npm i -g vercel`
2. `vercel login`
3. `vercel --prod`
4. Vercel Dashboard → Project Settings → Environment Variables:
   ```
   NASA_API_KEY = QpCinFsJT2fYXQLkkrTNwCCsOlBgW1v66T1OqFqt
   ```

**API:** `/api/predict/full` (POST JSON)
**Frontend:** `/predicton.html` (spacecraft recs!)

## Features
- ML Impact Prediction (crater/seismic)
- Spacecraft Matching (DART, ORION, HAMMER, AEGIS-X)
- NASA NEO Integration
- 3D Earth Viz

![Demo](Frontend/predicton.html)
