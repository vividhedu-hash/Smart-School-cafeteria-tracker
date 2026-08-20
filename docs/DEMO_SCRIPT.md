# Showcase Demo Script (5–7 minutes)

A minute-by-minute script for demo day. Practice it at least twice the day
before. The demo needs: the laptop, its webcam, one clean plate, one plate
with visible leftover food, one enrolled person (you), and one volunteer who
is NOT enrolled.

---

## Pre-demo checklist — 30 minutes before

Do these in order. Total time: ~10 minutes, leaving buffer to fix problems.

1. **Power and position.** Laptop plugged in (training/inference drains
   battery). Camera aimed the same way as when you took the training photos —
   same height, angle, and distance. Same lighting if at all possible.
2. **Camera permission.** macOS: System Settings → Privacy & Security →
   Camera → make sure Terminal (or whatever launches the app) is allowed.
   Test by starting the engine once and confirming the Live Monitor shows
   frames.
3. **Start everything.** Double-click `Start_Cafeteria_Tracker.command` (or
   run `./run.sh` / `python run.py` from the project folder). The engine and
   dashboard start together and the browser opens at http://localhost:8501.
   This does NOT need wifi — everything is localhost.
4. **Verify System Readiness.** On the dashboard home page, the
   "🩺 System Readiness" panel must show READY for: Plate model, Waste model,
   Face engine, Enrolled people, and Camera. If Plate or Waste shows MISSING,
   the pipeline is disabled — you must activate a trained model on
   **Training → 📋 Model Versions** before the demo can work.
5. **Two or more people enrolled.** Check the roster on
   **Training → 👤 Face Enrollment** — you need at least yourself showing the
   "✅ READY" badge. Enroll a second teammate if you haven't (takes ~3 min).
6. **Sidebar sanity check.** Engine badge says "● ENGINE RUNNING", Camera
   "● ONLINE", Active Models shows real versions for Plate and Waste,
   Enrolled shows your count.
7. **Empty the Review Queue** (resolve or reject anything left over) and note
   the current Total Transactions count on the home page — you'll point at it
   going up by exactly one during the demo.
8. **Keep the launcher terminal open.** `./run.sh` holds the engine up until
   you press Ctrl+C. If you started the engine from the website instead, it
   auto-shuts down about a minute after you leave Live Monitor (that releases
   the camera). During the demo, never close the browser.
9. **Do one full dry run.** Show the food plate, confirm one transaction
   appears. If it works once, it will work on stage.
10. **Backup plan.** If the camera or machine fails on stage: you have the
    dry-run transaction in the Transactions page and its evidence photo —
    walk the judges through that recorded event, the Training page, and
    Analytics instead. The pitch survives without live video; know this path.

---

## The script

### 0:00–0:45 — Setup line and the live view

**Screen: 📹 Live Monitor page.**

> "This is a live feed from that webcam. On the right, the system is already
> scanning for faces — it says 'Scanning for faces…' because nobody's in
> frame. Below, this state bar says IDLE: the pipeline is a state machine and
> right now there's nothing to do. Every number you'll see — FPS, latency —
> is measured live, not mocked."

Step into frame briefly: the "Who's Here?" panel flips to your name with a
match percentage.

> "That's me — recognized from a 512-number face embedding, not a stored
> photo comparison. Watch what it does with plates."

### 0:45–1:45 — Empty plate: prove it doesn't fake

**Screen: 📹 Live Monitor.** Hold up / place the CLEAN plate in the camera's
plate zone (lower part of the frame).

> "First, an empty plate. Watch the state bar: PLATE_DETECTED …
> FOOD_ANALYSIS … and back to IDLE. No waste event, no transaction. This
> matters: the system classifies this plate as EMPTY and deliberately records
> nothing. It cannot be impressed into inventing waste — if we hadn't trained
> real models, this whole pipeline would be disabled and the home page would
> say so."

(If judges look skeptical, this is the moment to show the home page's System
Readiness panel for 5 seconds, then return to Live Monitor.)

### 1:45–3:00 — Plate with food + enrolled person: one transaction

**Screen: 📹 Live Monitor.** You (enrolled) hold the plate WITH leftover food
in the plate zone, with your face visible in the upper part of the frame.

> "Now a plate with leftovers. State bar: PLATE_DETECTED → FOOD_ANALYSIS →
> WASTE_EVENT — it's classified the waste level — → FACE_CAPTURE →
> FACE_RECOGNITION — it's matching my face against enrolled embeddings, voting
> across frames — → TRANSACTION_COMMIT → COOLDOWN. Done."

Point at the "Last Transaction" panel on the right (person, waste class,
status, latency in milliseconds).

**Switch to 📋 Transactions page.**

> "Here's the record: timestamp, my name, the waste category, the model's
> actual confidence scores, processing latency, and an evidence photo. One
> plate, exactly one transaction — the cooldown state prevents
> double-counting."

(Optional: paste the transaction ID into the "View Transaction Image" box to
show the evidence photo full-size.)

### 3:00–4:30 — Unknown person: the review queue

**Screen: 📹 Live Monitor.** Your NON-enrolled volunteer holds the food plate,
face in frame.

> "Same plate, but this person never enrolled. The face panel says
> 'Unknown — below threshold'. Watch the state machine: instead of
> TRANSACTION_COMMIT it goes to REVIEW_REQUIRED. The system does not guess."

**Switch to 🔍 Review Queue page.**

> "The event landed here: evidence photo, waste category, why it needs
> review, and the closest candidate matches with their similarity scores. A
> human makes the call."

Select the correct person from the "Person" dropdown (or leave a demo person),
click **✅ Confirm**.

> "Confirmed — the transaction is updated to MANUALLY_CONFIRMED. And this is
> more than bookkeeping: every human correction is a labeled example we can
> feed back into training. The system improves from its own mistakes."

### 4:30–5:30 — The training story

**Screen: 🧠 Training page — 📦 Waste Dataset tab.**

> "Nothing here came pretrained for our cafeteria. We photographed our own
> plates — here's the dataset, sorted into EMPTY, LOW, MEDIUM, HIGH waste —
> and the automatic 70/20/10 train/validation/test split."

**Click over to 🚀 Train Model tab, then 📋 Model Versions tab.**

> "One button fine-tunes a YOLOv8 classifier on those photos — transfer
> learning, minutes on this laptop. Every training run is versioned with its
> real accuracy metrics, and activating a new version hot-swaps it into the
> running engine within seconds, no restart. Enrollment works the same way:
> a guided wizard captures five poses and computes the face embedding."

### 5:30–6:30 — Analytics: why anyone should care

**Screen: 📊 Analytics page.**

> "And this is the payoff for the cafeteria: waste distribution, events over
> time, waste by person, and real processing latency. Today it's demo data —
> deployed for a term, this answers: which dishes get thrown away? Which days
> are worst? Is the new menu working? That's data nobody has today, from one
> webcam."

### 6:30–7:00 — Close

> "Everything runs locally — no cloud, no data leaves this laptop, and only
> people who opt in are ever recognized. It's honest AI: when it doesn't
> know, it asks a human, and it learns from the answer. Happy to take
> questions."

---

## Failure recovery lines

Things go wrong in live demos. Every failure here maps to a real feature —
use it.

**Camera feed lags or freezes:**
> "The feed's catching up — the dashboard polls a state file the engine
> writes; the engine itself is still processing at the FPS you see here."
Do: stay calm, wait 3 seconds. If truly stuck: sidebar → ⏹ Stop Engine →
▶ Start Engine (takes ~10 seconds; narrate the readiness panel while waiting).

**Plate not detected:**
> "The detector wants the plate in its region of interest and above a 55%
> confidence threshold — precision over recall, because a missed detection
> costs nothing but a false one pollutes the data."
Do: move the plate lower/center in frame, hold it still for a full second
(minimum presence time is deliberate debouncing).

**Waste class comes out wrong (e.g. LOW instead of MEDIUM):**
> "That's a real model making a real mistake — it was trained on about a
> hundred photos. The fix isn't code, it's data: this event's photo becomes a
> training example, we retrain from the dashboard, and version two replaces
> version one with a click. That correction loop is the actual design."

**Face not recognized (you show as Unknown):**
> "Below the similarity threshold — probably the lighting difference from my
> enrollment photos. And note what it did: it didn't misidentify me, it
> refused to guess and queued the event for review. That's the failure mode
> you want."
Do: proceed to the Review Queue and resolve it — it's the same demo, reordered.

**Everything dies (power, OS, camera hardware):**
Switch to the backup path: Transactions page from the dry run (or your phone
photos of it), Training page, Analytics. The pitch narrative in
`docs/PITCH.md` sections 3–7 works without live video.

---

## Screen-by-screen summary

| Time | Dashboard page | What happens |
|------|----------------|--------------|
| 0:00 | 📹 Live Monitor | Live feed, face panel recognizes you, state IDLE |
| 0:45 | 📹 Live Monitor | Empty plate → back to IDLE, no transaction |
| 1:45 | 📹 Live Monitor → 📋 Transactions | Food plate + you → full state walk → 1 transaction |
| 3:00 | 📹 Live Monitor → 🔍 Review Queue | Unknown person → REVIEW_REQUIRED → manual Confirm |
| 4:30 | 🧠 Training (Waste Dataset, Train Model, Model Versions tabs) | Dataset, training, versioning story |
| 5:30 | 📊 Analytics | Charts and the "why it matters" close |
| 6:30 | (any) | Privacy close + questions |
