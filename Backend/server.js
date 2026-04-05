const express = require("express");
const fs = require("fs");
const axios = require("axios");
const cors = require("cors");
const path = require("path");
const { createClient } = require("redis");

const PORT = Number(process.env.PORT || 3000);
const REDIS_URL = process.env.REDIS_URL || "redis://127.0.0.1:6379";
const ML_HOSTPORT = process.env.ML_HOSTPORT;
const ML_URL = process.env.ML_URL || (ML_HOSTPORT ? `http://${ML_HOSTPORT}/predict` : "http://127.0.0.1:8000/predict");
const FRONTEND_DIR = process.env.FRONTEND_DIR || path.join(__dirname, "..", "Frontend");
const CACHE_TTL_SECONDS = 3600;
const FEATURE_COUNT = 13;
const CORS_ORIGIN = process.env.CORS_ORIGIN;

const app = express();
app.use(express.json());
app.use(cors(CORS_ORIGIN ? { origin: CORS_ORIGIN } : undefined));

app.use((error, req, res, next) => {
  if (error instanceof SyntaxError && "body" in error) {
    return res.status(400).json({ error: "Request body must be valid JSON" });
  }

  return next(error);
});

const redis = createClient({
  url: REDIS_URL,
  socket: {
    reconnectStrategy: false
  }
});

let cacheAvailable = false;

redis.on("ready", () => {
  cacheAvailable = true;
  console.log("Redis cache connected");
});

redis.on("end", () => {
  cacheAvailable = false;
  console.warn("Redis connection closed");
});

redis.on("error", (error) => {
  cacheAvailable = false;
  console.warn(`Redis unavailable: ${error.message}`);
});

function isFiniteNumber(value) {
  return typeof value === "number" && Number.isFinite(value);
}

function validatePredictionInput(payload) {
  const numericFields = ["hour", "day", "month", "weekday", "junction", "lag1", "lag2"];
  const errors = [];

  for (const field of numericFields) {
    if (!isFiniteNumber(payload[field])) {
      errors.push(`${field} must be a valid number`);
    }
  }

  if (errors.length > 0) {
    return errors;
  }

  if (!Number.isInteger(payload.hour) || payload.hour < 0 || payload.hour > 23) {
    errors.push("hour must be an integer between 0 and 23");
  }

  if (!Number.isInteger(payload.day) || payload.day < 1 || payload.day > 31) {
    errors.push("day must be an integer between 1 and 31");
  }

  if (!Number.isInteger(payload.month) || payload.month < 1 || payload.month > 12) {
    errors.push("month must be an integer between 1 and 12");
  }

  if (!Number.isInteger(payload.weekday) || payload.weekday < 0 || payload.weekday > 6) {
    errors.push("weekday must be an integer between 0 and 6");
  }

  if (!Number.isInteger(payload.junction) || payload.junction < 1 || payload.junction > 4) {
    errors.push("junction must be an integer between 1 and 4");
  }

  if (payload.year !== undefined && (!Number.isInteger(payload.year) || payload.year < 2000 || payload.year > 2100)) {
    errors.push("year must be an integer between 2000 and 2100");
  }

  return errors;
}

function normalizePredictionInput(payload) {
  return {
    ...payload,
    year: Number.isInteger(payload.year) ? payload.year : new Date().getUTCFullYear()
  };
}

function generateFeatures({ year, hour, day, month, weekday, junction, lag1, lag2 }) {
  const hourSin = Math.sin((2 * Math.PI * hour) / 24);
  const hourCos = Math.cos((2 * Math.PI * hour) / 24);
  const rollingMean = (lag1 + lag2) / 2;

  return [
    year,
    hour,
    day,
    month,
    weekday,
    hourSin,
    hourCos,
    lag1,
    lag2,
    rollingMean,
    junction === 2 ? 1 : 0,
    junction === 3 ? 1 : 0,
    junction === 4 ? 1 : 0
  ];
}

function buildTimestamp({ year, month, day, hour }) {
  return new Date(Date.UTC(year, month - 1, day, hour));
}

function getTimeParts(date) {
  return {
    year: date.getUTCFullYear(),
    month: date.getUTCMonth() + 1,
    day: date.getUTCDate(),
    hour: date.getUTCHours(),
    weekday: (date.getUTCDay() + 6) % 7
  };
}

async function getCachedPrediction(key) {
  if (!cacheAvailable) {
    return null;
  }

  try {
    return await redis.get(key);
  } catch (error) {
    cacheAvailable = false;
    console.warn(`Cache read failed: ${error.message}`);
    return null;
  }
}

async function setCachedPrediction(key, value) {
  if (!cacheAvailable) {
    return;
  }

  try {
    await redis.set(key, JSON.stringify(value), { EX: CACHE_TTL_SECONDS });
  } catch (error) {
    cacheAvailable = false;
    console.warn(`Cache write failed: ${error.message}`);
  }
}

async function requestPrediction(features) {
  if (!Array.isArray(features) || features.length !== FEATURE_COUNT) {
    throw new Error("Invalid feature vector");
  }

  const response = await axios.post(ML_URL, { features }, { timeout: 10000 });
  return response.data;
}

function sendHealth(req, res) {
  res.status(200).json({
    success: true,
    message: "Backend running",
    cacheAvailable
  });
}

app.use("/api/health", sendHealth);
app.get("/health", sendHealth);

async function handlePredict(req, res) {
  const payload = normalizePredictionInput(req.body || {});
  const errors = validatePredictionInput(payload);
  if (errors.length > 0) {
    return res.status(400).json({ errors });
  }

  try {
    const features = generateFeatures(payload);
    const key = features.join(",");
    const cached = await getCachedPrediction(key);

    if (cached) {
      return res.json({ source: "cache", prediction: JSON.parse(cached) });
    }

    const prediction = await requestPrediction(features);
    await setCachedPrediction(key, prediction);

    return res.json({ source: "model", prediction });
  } catch (error) {
    const status = error.response?.status || 500;
    const message = error.response?.data?.detail || error.message || "Prediction failed";
    return res.status(status).json({ error: message });
  }
}

async function handlePredict24h(req, res) {
  const payload = normalizePredictionInput(req.body || {});
  const errors = validatePredictionInput(payload);
  if (errors.length > 0) {
    return res.status(400).json({ errors });
  }

  try {
    let { year, month, day, hour, weekday, junction, lag1, lag2 } = payload;
    let cursor = buildTimestamp({ year, month, day, hour });
    const predictions = [];
    const labels = [];

    for (let i = 0; i < 24; i += 1) {
      const parts = getTimeParts(cursor);
      const features = generateFeatures({
        year: parts.year,
        hour: parts.hour,
        day: parts.day,
        month: parts.month,
        weekday: parts.weekday,
        junction,
        lag1,
        lag2
      });

      const predictionResponse = await requestPrediction(features);
      const prediction = predictionResponse.prediction;

      predictions.push(prediction);
      labels.push(`${parts.hour.toString().padStart(2, "0")}:00 ${parts.day}/${parts.month}`);

      lag2 = lag1;
      lag1 = prediction;
      cursor.setUTCHours(cursor.getUTCHours() + 1);

      const nextParts = getTimeParts(cursor);
      year = nextParts.year;
      month = nextParts.month;
      day = nextParts.day;
      hour = nextParts.hour;
      weekday = nextParts.weekday;
    }

    return res.json({ predictions, labels });
  } catch (error) {
    const status = error.response?.status || 500;
    const message = error.response?.data?.detail || error.message || "Prediction failed";
    return res.status(status).json({ error: message });
  }
}

app.post("/api/predict", handlePredict);
app.post("/predict", handlePredict);
app.post("/api/predict-24h", handlePredict24h);
app.post("/predict-24h", handlePredict24h);

if (fs.existsSync(FRONTEND_DIR)) {
  app.use(express.static(FRONTEND_DIR));

  app.get("/", (req, res) => {
    res.sendFile(path.join(FRONTEND_DIR, "index.html"));
  });
}

async function start() {
  try {
    await redis.connect();
    cacheAvailable = true;
  } catch (error) {
    cacheAvailable = false;
    console.warn(`Starting without Redis cache: ${error.message}`);
  }

  app.listen(PORT, () => {
    console.log(`Backend running on port ${PORT}`);
  });
}

start();
