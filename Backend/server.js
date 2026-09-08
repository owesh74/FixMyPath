const express    = require("express");
const bodyParser = require("body-parser");
const fs         = require("fs");
const nodemailer = require("nodemailer");
const cors       = require("cors");
require("dotenv").config();

const app = express();
app.use(bodyParser.json());

// FIX: explicit CORS config — allows dashboard on any port/origin during dev
app.use(cors({
    origin: "*",
    methods: ["GET", "POST", "OPTIONS"],
    allowedHeaders: ["Content-Type"]
}));

const PORT = process.env.PORT || 5000; // Render will assign a port automatically
const FILE = "reports.json";

let lastDetectionTime = 0;
let mode = "AUTO"; // "AUTO" | "MANUAL"

// ── Load existing reports ──────────────────────────────────────
let reports = [];
if (fs.existsSync(FILE)) {
    try {
        reports = JSON.parse(fs.readFileSync(FILE, "utf8"));
    } catch (e) {
        console.error("⚠️  Failed to parse reports.json — starting fresh:", e.message);
        reports = [];
    }
}

// ── Helper: next safe ID (FIX: length+1 breaks after deletes) ──
function nextId() {
    if (reports.length === 0) return 1;
    return Math.max(...reports.map(r => r.id)) + 1;
}

// ── Helper: save to disk ───────────────────────────────────────
function saveReports() {
    try {
        fs.writeFileSync(FILE, JSON.stringify(reports, null, 2)); // FIX: try/catch
    } catch (e) {
        console.error("⚠️  Failed to write reports.json:", e.message);
    }
}

// ── Haversine distance (metres) ────────────────────────────────
function getDistance(lat1, lon1, lat2, lon2) {
    const R     = 6371e3;
    const toRad = x => x * Math.PI / 180;
    const dLat  = toRad(lat2 - lat1);
    const dLon  = toRad(lon2 - lon1);
    const a     = Math.sin(dLat/2) ** 2 +
                  Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon/2) ** 2;
    return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

// ── Email transporter ──────────────────────────────────────────
const transporter = nodemailer.createTransport({
    service: "gmail",
    host: "smtp.gmail.com",
    port: 587,
    secure: false,
    auth: {
        user: process.env.EMAIL_USER,
        pass: process.env.EMAIL_PASS   // use Gmail App Password, not your real password
    }
});

function buildMailOptions(pothole, subject) {
    const mapLink = `https://maps.google.com/?q=${pothole.latitude},${pothole.longitude}`;
    return {
        from:    process.env.EMAIL_USER,
        to:      process.env.NOTIFY_EMAIL || "owesh9@gmail.com",
        subject,
        text: `
Pothole Report — FixMyPath

📍 Location : ${mapLink}
🔢 Count     : ${pothole.count}
⚠️  Status   : ${pothole.status.toUpperCase()}
🕐 Detected  : ${pothole.lastDetected}

--
Sent by FixMyPath automated system
        `.trim()
    };
}

// ════════════════════════════════════════════════════════════════
// POST /report  — receive detection from Android app
// ════════════════════════════════════════════════════════════════
app.post("/report", async (req, res) => {
    const { latitude, longitude, magnitude, probability, timestamp } = req.body;

    // FIX: validate inputs before processing
    if (latitude == null || longitude == null || magnitude == null || probability == null) {
        return res.status(400).json({ message: "Missing required fields" });
    }

    // FILTER 1: Confidence threshold
    if (!(probability > 0.4 && magnitude > 11.5)) {
        return res.json({ message: "Ignored (low confidence)" });
    }

    // FILTER 2: Time spam guard (3 seconds global cooldown)
    const currentTime = Date.now();
    if (currentTime - lastDetectionTime < 3000) {
        return res.json({ message: "Ignored (spam detection)" });
    }
    lastDetectionTime = currentTime;

    // ── Clustering: find pothole within 20m ───────────────────
    let pothole = reports.find(r =>
        getDistance(latitude, longitude, r.latitude, r.longitude) < 20
    );

    if (pothole) {
        pothole.count++;
        pothole.lastDetected = timestamp;
    } else {
        pothole = {
            id:           nextId(),   // FIX: safe ID generation
            latitude,
            longitude,
            count:        1,
            status:       "normal",
            lastDetected: timestamp   // FIX: stored as readable string from app
        };
        reports.push(pothole);
    }

    // ── Danger classification ─────────────────────────────────
    let shouldSendEmail = false;
    if (pothole.count >= 2 && pothole.status !== "dangerous") {
        pothole.status = "dangerous";
        shouldSendEmail = true;
    } else if (pothole.count < 2) {
        pothole.status = "normal";
    }

    saveReports();

    // ── Auto email ────────────────────────────────────────────
    if (mode === "AUTO" && shouldSendEmail) {
        try {
            await transporter.sendMail(
                buildMailOptions(pothole, "⚠️ Pothole Detected – Road Maintenance Required")
            );
            console.log(`📧 Auto email sent for pothole #${pothole.id}`);
        } catch (err) {
            console.error("❌ Auto email error:", err.message);
        }
    }

    res.json({ message: "Cluster updated", pothole });
});

// ════════════════════════════════════════════════════════════════
// GET /reports  — all potholes (used by dashboard)
// ════════════════════════════════════════════════════════════════
app.get("/reports", (req, res) => {
    res.json(reports);
});

// ════════════════════════════════════════════════════════════════
// GET /mode  — FIX: dashboard can now READ current mode
// ════════════════════════════════════════════════════════════════
app.get("/mode", (req, res) => {
    res.json({ mode });
});

// ════════════════════════════════════════════════════════════════
// POST /toggle-mode  — switch AUTO ↔ MANUAL
// ════════════════════════════════════════════════════════════════
app.post("/toggle-mode", (req, res) => {
    mode = mode === "AUTO" ? "MANUAL" : "AUTO";
    console.log(`⚙️  Mode switched to ${mode}`);
    res.send(`Mode: ${mode}`);  // dashboard parses this string
});

// ════════════════════════════════════════════════════════════════
// POST /send-email/:id  — manual email from dashboard
// ════════════════════════════════════════════════════════════════
app.post("/send-email/:id", async (req, res) => {
    const id      = parseInt(req.params.id);
    const pothole = reports.find(p => p.id === id);

    if (!pothole) {
        return res.status(404).json({ message: "Pothole not found" }); // FIX: JSON response
    }

    try {
        await transporter.sendMail(
            buildMailOptions(pothole, "📋 Pothole Report – Manual Alert")
        );
        console.log(`📧 Manual email sent for pothole #${id}`);
        res.json({ message: "Email sent", id });  // FIX: JSON response
    } catch (err) {
        console.error("❌ Manual email error:", err.message);
        res.status(500).json({ message: "Email failed", error: err.message });
    }
});

// ════════════════════════════════════════════════════════════════
// GET /stats  — FIX: new endpoint, pre-computed stats for dashboard
// ════════════════════════════════════════════════════════════════
app.get("/stats", (req, res) => {
    const total    = reports.length;
    const dangerous = reports.filter(r => r.status === "dangerous").length;
    const highest  = reports.reduce((m, r) => Math.max(m, r.count), 0);
    res.json({ total, dangerous, normal: total - dangerous, highest, mode });
});

// ── Start ─────────────────────────────────────────────────────
app.listen(PORT, "0.0.0.0", () => {
    console.log(`🚀 Server running on port ${PORT}`);
});