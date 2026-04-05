# Traffic Flow Prediction

This project is ready to deploy on Render as a public web app with:

- a public backend service that also serves the frontend
- a private prediction service
- a managed Redis-compatible cache

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
- The prediction service is private and is reached over Render's private network.
- Redis is private and is not exposed publicly.
- The private prediction service uses the `starter` plan because Render private services are not available on the free plan.

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
