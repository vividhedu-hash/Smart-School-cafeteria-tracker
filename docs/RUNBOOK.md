# Operator Runbook — Smart Cafeteria Waste Tracker

This manual is for the person running the system day-to-day (e.g. the
cafeteria supervisor). No programming knowledge required. Where you see a
word in a box like **▶ Start Engine**, it's a button you'll find on screen.

---

## 1. Daily startup

1. Make sure the computer is on and the webcam is connected and pointing at
   the tray-return spot (same position every day — the AI was trained on
   that exact view).
2. Double-click **`Start_Cafeteria_Tracker.command`** in the project folder.
   (Alternative for a terminal user: `./run.sh` or `python run.py` from the
   project folder.)
3. A terminal window opens and, after a few seconds, your web browser opens
   the dashboard at **http://localhost:8501**. This is a local page — it works
   without internet.
4. On the dashboard home page, check two things:
   - The sidebar shows **● ENGINE RUNNING** (green). If it shows
     **● ENGINE STOPPED**, click **▶ Start Engine** in the sidebar.
   - The **System Readiness** row shows READY for Plate model, Waste model,
     Face engine, Enrolled people, and Camera.
5. Open the **Live Monitor** page (left sidebar) and confirm you can see the
   camera picture.

**Important:** leave the terminal that launched `./run.sh` open. That process
keeps the engine alive. Ctrl+C stops the engine and the dashboard together
and turns the camera light off. If you started the engine from the website
instead (▶ Start Engine) and then leave Live Monitor, it still releases the
camera after about a minute on its own — click **▶ Start Engine** when you
come back.

## 2. Daily shutdown

Either of these works:

- Click **⏹ Stop Engine** in the dashboard sidebar, then close the browser
  tab and the terminal window.
- Or simply go to the terminal window that opened at startup and press
  **Ctrl+C** — it stops the engine and the dashboard together and prints
  "All services stopped."

## 3. Enrolling a new student

Enrollment is opt-in: only enrolled people are ever recognized by name.
Get consent first.

1. Go to the **Training** page → **👤 Face Enrollment** tab.
2. Under "➕ New Person", type a **Person ID** (short, no spaces — e.g.
   `priya_07`) and a **Display Name** (e.g. `Priya Sharma`), then click
   **Create & Start Enrollment →**.
3. The browser will ask permission to use the camera — click Allow.
4. Follow the on-screen guide. It asks for 5 poses in order: face forward,
   turn left, turn right, look up, look down. Line the face up with the oval;
   the tool auto-captures when the face is steady and takes **3 photos per
   pose** (15 total), advancing automatically. Blurry or dark photos are
   rejected — just retake.
   - You can click **➡️ Next Pose** early or **⏭ Skip pose** if a pose won't
     cooperate; at least 5 good photos overall is the recommended minimum.
5. On the final step, click **⚡ Generate Embeddings**. The first ever run
   downloads the face model (~250 MB, needs internet once); after that it
   takes ~5–15 seconds. You'll see "enrolled!" with balloons when done.
6. Back on the enrollment tab, the person now appears in the "📋 Enrolled
   People" roster with a **✅ READY** badge. The running engine picks up new
   enrollments automatically (within ~30 seconds).

To add more photos for an existing person (e.g. recognition is unreliable),
pick them under "🔄 Existing Person" and click **📷 Add More Images →**, then
regenerate embeddings.

## 4. Uploading waste photos and retraining the waste model

Do this when you have new photos of plates (see
`docs/DATA_COLLECTION_GUIDE.md` for how to take good ones).

**Upload:**
1. **Training** page → **📦 Waste Dataset** tab.
2. Choose the **Category** (EMPTY, LOW_WASTE, MEDIUM_WASTE, or HIGH_WASTE).
3. Drag the photo files into the upload box, click **➕ Add to Dataset**.
   Exact duplicate files are skipped automatically.
4. The bar chart below shows how many images each category has. Aim for a
   similar count in every category.

**Train:**
1. **Training** page → **🚀 Train Model** tab.
2. Training Task: **Waste Classification**. Leave Base Model, Epochs, Image
   Size, and Batch Size at their defaults unless told otherwise. Device:
   **Auto**.
3. Click **🚀 START TRAINING**. Training runs in the background and may take
   several minutes — leave the page open. (It refuses to start with fewer
   than 4 images total, and realistically needs 25+ per category.)
4. When it finishes you'll see "Training complete" with a version number
   (e.g. `v003`) and its accuracy. Click **⚡ Activate this Model**.

**Activate an older/newer version later:**
- **Training** page → **📋 Model Versions** tab → expand a version → click
  **⚡ Activate**. The running engine hot-swaps to the new model within about
  5 seconds — no restart needed.

Note: the **Plate Detection** task in the Train Model tab currently cannot be
run from the dashboard — it needs a labeled dataset prepared by the technical
team. Ask your student developer to run plate training; you only ever need
the Waste Classification task day-to-day.

## 5. Working the review queue

When the system sees a waste event but isn't confident who the person is, it
does not guess — it files the event for you to decide.

1. Open the **Review Queue** page. It lists each unresolved event with the
   evidence photo, the waste category, the confidence, and (when available)
   the closest candidate identities with similarity scores.
2. For each event:
   - If you recognize the person: pick them in the **Person** dropdown and
     click **✅ Confirm**. The transaction is updated with their name (status
     becomes MANUALLY_CONFIRMED).
   - If the event is bogus (no real person / not a real waste event): click
     **❌ Reject**. The transaction is marked REJECTED and excluded from
     per-person stats.
3. Click **🔄 Refresh** to check for new items. A good habit: clear the queue
   once per day after lunch service.

## 6. Exporting transaction data

1. Open the **Transactions** page.
2. Use the filters in the left sidebar (Person, Waste Category, Status, From
   Date / To Date, Max rows) to select what you want.
3. Scroll to the bottom and click **⬇️ Export CSV**. Your browser downloads
   `cafeteria_transactions.csv`, which opens in Excel / Google Sheets /
   Numbers.

You can also inspect any single event: paste its Transaction ID (from the
table) into the "View Transaction Image" box to see the evidence photo and
full details.

## 7. Troubleshooting

| Problem | What to do |
|---------|-----------|
| Camera won't open / feed is black | On macOS: Apple menu → System Settings → **Privacy & Security → Camera** → switch ON for Terminal (or the app that launches the tracker). Then stop and start the engine. Also check no other app (Zoom, FaceTime) is using the camera. |
| Engine won't start (**● ENGINE STOPPED** won't change) | Click **▶ Start Engine** and wait ~10 seconds, then refresh the page. If it still fails, close everything and double-click `Start_Cafeteria_Tracker.command` again. Persistent failures: check the newest file in the `logs/` folder and show it to your technical contact. |
| Dashboard shows "not ready" / "Waste pipeline is DISABLED" banner | The System Readiness panel tells you exactly which piece is MISSING. Waste model missing → train and activate one (section 4). Plate model missing → the technical team must train it. Face engine PENDING → it downloads automatically the first time the engine starts with internet. |
| Model training fails immediately | Most common cause: too few images — the trainer needs at least 4 total, and warns you on the page. Upload more photos (25+ per category) and retry. Training also needs internet the first time, to download the base model. |
| A person keeps showing as "Unknown" | Re-enroll with better photos: even, bright, front-facing light (no window behind them), no cap/mask, and capture all 5 poses. Use **📷 Add More Images →** for that person and regenerate embeddings. Recognition threshold is deliberately strict — unsure events go to the Review Queue instead of being guessed. |
| Camera light turns off by itself | Expected: the engine auto-stops ~30 seconds after the last dashboard tab is closed. Reopen the dashboard and click **▶ Start Engine**. |
| One plate created two transactions | Should not happen (a cooldown period follows every event). If you see it, **❌ Reject** the duplicate in the Review Queue or note its Transaction ID, and report it to the technical team. |

## 8. Data care — where things live and how to back them up

Everything is stored locally inside the project folder. Nothing is uploaded
anywhere.

| What | Where |
|------|-------|
| The database (all transactions, people, reviews) | `database/cafeteria.db` |
| Evidence photos of waste events | `data/captures/` |
| Review-queue evidence copies | `data/review_queue/` |
| Enrollment face photos + identity data | `data/enrollment/<person_id>/` |
| Training photos you uploaded | `data/datasets/waste/` and `data/datasets/plate/` |
| Trained models + version registry | `models/` |
| Logs (for troubleshooting) | `logs/` |

**Backup (weekly recommended):** with the engine STOPPED, copy the
`database/` and `data/` folders to a USB drive or a dated folder. That's a
complete backup of all records and training data.

**Retention advice:**
- Evidence photos (`data/captures/`, `data/review_queue/`): keep only as long
  as needed to resolve reviews and disputes — a 30-day purge of old dated
  folders is reasonable.
- Enrollment data (`data/enrollment/<person_id>/`): delete the person's
  folder immediately if they (or a parent) withdraw consent or leave the
  school. The engine stops recognizing them within ~30 seconds.
- The database itself contains names and timestamps — treat backups with the
  same care as any student record, and store them on school-controlled media
  only.
