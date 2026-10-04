# FallGuard — AI-Powered Fall Detection for Android

> *"Because falling down is an accident. Not getting help shouldn't be."*

FallGuard is a fully offline Android app that detects falls in real time using your phone's built-in sensors and a machine learning model running entirely on-device. No internet. No server. No cloud. Just your phone, your SIM card, and an emergency contact who gets an SMS the moment something goes wrong.

Built for elderly people, people with medical conditions, and anyone who lives or works alone — FallGuard runs silently in the background while you go about your day. The moment it detects a fall, it vibrates, speaks an alert, and gives you 20 seconds to cancel before automatically sending an SOS SMS with your GPS location.

---

## What makes this different

Most fall detection apps send your sensor data to a cloud server for processing. That means they need internet, have latency, cost money to run, and raise privacy concerns.

FallGuard does everything on the phone itself:

- The ML model runs using **ONNX Runtime Android** — no server call needed
- SMS is sent via **Android's native SmsManager** — no Twilio, no API key
- GPS coordinates come from **satellite GPS** — no WiFi or internet
- Voice cancel uses **on-device speech recognition** — works in airplane mode

---

## How it works

```
Phone sensors (50 Hz)
        ↓
9 statistical features extracted
(max, kurtosis, skewness, post-impact)
        ↓
RandomForest model — ONNX Runtime Android
        ↓
Fall detected? (confidence > sensitivity threshold)
        ↓
Strong haptic burst + TTS alert + 20s countdown
        ↓
User cancels?  →  NO  →  SMS sent with GPS location
     ↓
   Safe ✓
```

---

## Features

- **100% offline** — works with no WiFi, no mobile data, no server
- **Always-on background monitoring** — runs even when screen is off, phone locked, or other apps are open
- **Pops up over any app** — SOS screen appears over YouTube, calls, music — whatever is running
- **Auto-restart on reboot** — monitoring resumes automatically after phone restarts
- **Haptic feedback** — Morse SOS vibration pattern for deaf and hard-of-hearing users
- **Voice cancel** — say "Cancel" to stop a false alarm without touching the screen
- **Text-to-Speech alerts** — spoken countdown so user knows what's happening
- **GPS in SMS** — emergency contact gets a Google Maps link with exact location
- **Adjustable sensitivity** — tune detection threshold from 0% to 100%
- **Adjustable countdown** — set SOS delay from 5 to 60 seconds

---

## Tech Stack

### Android App
| Component | Technology |
|---|---|
| Language | Kotlin |
| Min SDK | Android 8.0 (API 26) |
| Target SDK | Android 15 (API 35) |
| UI | ViewBinding + Material Design 3 |
| Background service | Foreground Service + WakeLock |
| Sensor reading | Android SensorManager (50 Hz) |
| On-device ML | ONNX Runtime Android 1.19.2 |
| SMS | Android SmsManager (native) |
| Location | Android LocationManager (GPS satellite) |
| Voice | Android SpeechRecognizer (on-device) |
| Text-to-Speech | Android TextToSpeech |
| Haptics | Android VibratorManager |
| Storage | SharedPreferences |
| Auto-start | BroadcastReceiver (BOOT_COMPLETED) |

### ML Pipeline (Python)
| Component | Technology |
|---|---|
| Language | Python 3 |
| Model | RandomForest (scikit-learn) |
| Features | 9 statistical features from IMU windows |
| Labels | 13 activity classes (4 fall, 1 stumble, 8 normal) |
| Export | skl2onnx → model.onnx |
| Training data | IMU sensor dataset (accelerometer + gyroscope) |

---

## Project Structure

```
FallGuard/
├── app/src/main/
│   ├── java/com/fallguard/
│   │   ├── ml/
│   │   │   ├── FallClassifier.kt          # ONNX inference engine
│   │   │   └── SensorFeatureExtractor.kt  # 9 feature computation
│   │   ├── service/
│   │   │   ├── FallDetectionService.kt    # Always-on foreground service
│   │   │   └── BootReceiver.kt            # Auto-start on reboot
│   │   ├── ui/
│   │   │   ├── MainActivity.kt            # Home screen
│   │   │   ├── SettingsActivity.kt        # Configuration
│   │   │   └── SosAlertActivity.kt        # SOS countdown screen
│   │   └── utils/
│   │       ├── SmsSender.kt               # Native SMS (no Twilio)
│   │       ├── LocationHelper.kt          # GPS with timeout fallback
│   │       ├── HapticManager.kt           # Morse SOS vibration
│   │       ├── TtsManager.kt              # Text-to-speech alerts
│   │       ├── VoiceCommandManager.kt     # Voice cancel listener
│   │       └── Prefs.kt                   # Settings storage
│   ├── assets/
│   │   └── model.onnx                     # Trained ML model (add manually)
│   └── res/                               # Layouts, drawables, themes
├── app/build.gradle                       # Dependencies
└── README.md
```

---

## Getting Started

### Step 1 — Export your model to ONNX

In your Python ML environment, run:

```python
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import joblib

model = joblib.load("model.pkl")
le    = joblib.load("label_encoder.pkl")

onx = convert_sklearn(
    model, "FallGuard",
    [("input", FloatTensorType([None, 9]))]
)
with open("model.onnx", "wb") as f:
    f.write(onx.SerializeToString())

print("Label classes:", list(le.classes_))
```

### Step 2 — Add model to Android project

Copy `model.onnx` into:
```
app/src/main/assets/model.onnx
```

### Step 3 — Open in Android Studio

1. Open Android Studio
2. File → Open → select the `FallGuard` folder
3. Wait for Gradle sync to complete
4. Click ▶ Run

### Step 4 — Configure the app

1. Open the app on your phone
2. Grant all permissions (SMS, Location, Microphone, Notifications)
3. Allow "Display over other apps" when prompted
4. Go to Settings → enter contact name and phone number with country code (e.g. `+919876543210`)
5. Tap Start Monitoring

---

## The 9 Features

These are extracted from a 2-second sliding window of sensor data:

| Feature | Description |
|---|---|
| `acc_max` | Peak acceleration magnitude |
| `gyro_max` | Peak gyroscope magnitude |
| `lin_max` | Peak linear acceleration magnitude |
| `acc_kurtosis` | Sharpness of acceleration spike |
| `gyro_kurtosis` | Sharpness of rotation spike |
| `acc_skewness` | Directional asymmetry of acceleration |
| `gyro_skewness` | Directional asymmetry of rotation |
| `post_gyro_max` | Peak gyroscope in 1s post-event window |
| `post_lin_max` | Peak linear acceleration in 1s post-event window |

---

## Activity Labels

| Category | Labels |
|---|---|
| 🔴 Fall | FOL (Forward), FKL (Kneel), SDL (Sideways), BSC (Back) |
| 🟡 Stumble | STU |
| 🟢 Normal | WAL, JOG, STD, STN, JUM, CSI, CSO, SCH |

---

## Permissions Required

| Permission | Why |
|---|---|
| `SEND_SMS` | Send SOS alert to emergency contact |
| `ACCESS_FINE_LOCATION` | Get GPS coordinates for SMS |
| `RECORD_AUDIO` | Voice cancel command |
| `POST_NOTIFICATIONS` | Show monitoring notification |
| `SYSTEM_ALERT_WINDOW` | Show SOS screen over other apps |
| `HIGH_SAMPLING_RATE_SENSORS` | Read sensors at 50Hz |
| `RECEIVE_BOOT_COMPLETED` | Auto-start after reboot |
| `FOREGROUND_SERVICE_HEALTH` | Background health monitoring |
| `WAKE_LOCK` | Keep CPU active when screen is off |

---

## Team

Built at a hackathon by **Impact Innovators**

| Member | Role |
|---|---|
| Saurav Yadav] | Android app development |
| [Rishabh Pandey] | ML model training and pipeline |
| [Uttam Kumar] | Data collection and research |

---


