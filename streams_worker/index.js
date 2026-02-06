import express from "express";
import Redis from "ioredis";

const app = express();
app.use(express.json());

app.get("/", (_req, res) => {
  res.send("ok");
});

app.post("/events", async (req, res) => {
  res.json({ ok: true });

  try {
    const redis = new Redis(process.env.REDIS_URL, {
      lazyConnect: true,
      enableOfflineQueue: false,
    });

    await redis.connect();
    await redis.xadd(
      "events",
      "*",
      "payload",
      JSON.stringify(req.body)
    );
    redis.disconnect();
  } catch (err) {
    console.error("Redis error:", err);
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`streams_worker listening on ${PORT}`);
});