import Redis from "ioredis";

const redis = new Redis(process.env.REDIS_URL || "redis://localhost:6379");

const STREAM = "events";
const GROUP = "main";
const CONSUMER = process.env.CONSUMER_NAME || "worker-1";

async function ensureGroup() {
  try {
    await redis.xgroup(
      "CREATE",
      STREAM,
      GROUP,
      "$",
      "MKSTREAM"
    );
  } catch (e) {
    if (!e.message.includes("BUSYGROUP")) throw e;
  }
}

async function run() {
  await ensureGroup();
  console.log("Worker listening…");

  while (true) {
    const res = await redis.xreadgroup(
      "GROUP",
      GROUP,
      CONSUMER,
      "COUNT",
      10,
      "BLOCK",
      5000,
      "STREAMS",
      STREAM,
      ">"
    );

    if (!res) continue;

    for (const [, messages] of res) {
      for (const [id, fields] of messages) {
        const message = {};

        for (let i = 0; i < fields.length; i += 2) {
          message[fields[i]] = fields[i + 1];
        }

        // 🔧 do work here
        console.log("Event:", message);

        await redis.xack(STREAM, GROUP, id);
      }
    }
  }
}

run().catch(err => {
  console.error(err);
  process.exit(1);
});