# Traffic Flow Prediction

This project is ready to deploy on Render for free as a public web app with:

- a public backend service that also serves the frontend
- a public prediction service
- a free Render Key Value cache

## Render Deployment

1. Push this repository to GitHub.
2. In Render, create a new Blueprint and point it to this repository.
3. Render will detect [`render.yaml`](./render.yaml) and propose:
   - `traffic-flow-backend`
   - `traffic-flow-prediction-service`
   - `traffic-flow-cache`
4. Approve the infrastructure creation and deploy.
5. Open the public backend URL after deployment finishes.

## Notes

- The backend service is public.
- The prediction service is also public so the backend can reach it on Render's free tier.
- The backend reads the prediction service URL from `ML_URL`, which Render injects from the prediction service's public URL.
- Render injects `REDIS_URL` from the Key Value instance's internal connection string.
- The backend still starts without Redis if `REDIS_URL` is not set or the cache is unavailable.
- Free Render Key Value is in-memory only, so cached data can be lost on restarts.
- On Render's free tier, services spin down after inactivity, so the first request can take a little longer.

## Local Development

Run the full stack locally with:

```bash
docker compose up --build
```

Then open:

```text
http://localhost:3000
```

## Retraining The Model

If `traffic.csv` is present in the project root, retrain the model with:

```bash
python prediction-service/train_model.py
```

This rewrites:

- `prediction-service/traffic_model.pkl`
- `prediction-service/model_metadata.json`
