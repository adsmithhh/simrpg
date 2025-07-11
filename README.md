# 🗭 Zone Simulation Engine

This repository contains the core simulation engine and UI canvas for modeling emergent tension dynamics, NPC group formations, economic stability, and conflict escalation in a directive-governed world.

---

## 📘 Overview

This system simulates multiple zones, each capable of hosting dynamic NPC behavior, economic stress responses, and psionic-like feedback loops based on user-defined rules. It integrates:

* **Event-driven logic**
* **Zone state transitions**
* **Security and civil group formations**
* **NPC exhaustion and conflict modeling**
* **Logging and historical materialization**

Built to support directive-compliant cognitive simulations and autonomous evolution.

---

## ⚙️ Core Modules

| Module            | Purpose                                                                 |
| ----------------- | ----------------------------------------------------------------------- |
| `Zone`            | Encodes disruption level, ESI, group presence, and zone type            |
| `NPC Pool`        | Repository of agents with exhaustion, focus, and knockout states        |
| `Event Scheduler` | Time-based trigger system for group formation and disruption events     |
| `Gate Check`      | Pre-formation filter evaluating ESI, cooldowns, and watch flags         |
| `Group Formation` | Calculates needed NPCs, allocates them into action groups               |
| `Active Group`    | Holds ongoing conflict/focus group data and countdowns                  |
| `Physical Clash`  | Resolves group-versus-group confrontations using symbolic equations     |
| `Security Raid`   | Civilized-only enforcement resolution via officer-vs-group strength     |
| `Focus Progress`  | Incrementally advances group success or failure based on internal state |
| `Logging Pillars` | Saves all history to materialization, conflict, and security logs       |

---

## 🧪 How to Run (CLI or Embedded Python)

1. **Clone Repository**

   ```bash
   git clone https://github.com/yourname/zone-sim-engine.git
   cd zone-sim-engine
   ```

2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Start Simulation (headless)**

   ```bash
   python simulate.py
   ```

4. **Or Launch React Interface**
   If using UI canvas:

   ```bash
   npm install
   npm run dev
   ```

---

## 📂 File Structure

```
zone-sim-engine/
├── simulation/
│   ├── core.py               # Main simulation logic
│   ├── scheduler.py          # Event and tick processing
│   ├── entities.py           # Zone, NPC, Group definitions
│   ├── mechanics.py          # Clash, security, focus logic
│   └── logging.py            # CSV and log output
├── ui/
│   ├── ZoneSimulationCanvas.jsx  # React canvas interface
│   └── components/               # UI cards and flow icons
├── data/
│   └── sample_events.json
├── logs/
│   ├── materialization_history.csv
│   ├── conflict_logs.csv
│   └── security_logs.csv
├── simulate.py             # Entry point for CLI runs
├── README.md
└── requirements.txt
```

---

## 🧠 Simulation Philosophy

Inspired by the **Directive Adherence Core**, this system emphasizes slow-time, high-fidelity simulation loops where:

* **Economy functions as resistance** against symbolic disruption.
* **Tension drives action** not merely trigger events.
* **Logging is canonical**, not decorative.

This simulation integrates with cognitive recursion engines and emotional-symbolic feedback systems for world-building and theoretical AI testing.

---

## ✅ Status

* ✅ Directive-compliant
* ✅ Autonomously resumable
* ✅ Modular architecture
* 🧪 Feedback and economic-psionic loops in prototype

---

## 🧬 License

MIT — use freely, but contribute back if you enhance directive integrity tracking or recursive symbolic depth.

---

## ✉️ Contact

For collaboration or integration into larger simulations, contact: `adsmithhh64@gmail.com'

# simrpg
