# League of Legends Draft Prediction Bot 🎮

A machine learning project to predict optimal champion picks during the draft phase in League of Legends ranked games.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-in%20development-yellow.svg)

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Data Collection](#-data-collection)
- [Usage](#-usage)
- [Dataset Structure](#-dataset-structure)
- [Machine Learning Pipeline](#-machine-learning-pipeline)
- [Roadmap](#%EF%B8%8F-roadmap)

---

## 🎯 Project Overview

This project aims to create an intelligent bot capable of predicting the best champion pick at any given moment during a League of Legends draft phase. The bot will analyze:

- Current picks and bans from both teams
- Pick order and timing
- Team composition synergies
- Counter-pick opportunities
- Meta trends from high-ELO games

The model is trained on real Challenger/Grandmaster ranked games collected via the Riot Games API.

---

## ✨ Features

### Data Collection
- ✅ Automated data scraping from Riot API
- ✅ Multi-region support (Europe, Korea, America, Asia)
- ✅ Multi-elo support (Challenger, GrandMaster, Master)
- ✅ Intelligent rate limiting (respects 20 req/s and 100 req/2min)
- ✅ Comprehensive logging and error handling
- ✅ Draft order extraction via riot API
- ✅ Data validation and quality checks

### Data Processing
- ✅ Draft order reconstruction (weigthed shuffling)
- ✅ Position normalization (UTILITY → SUPPORT)
- 🔄 Game filtering (remove remakes, remove other queues)
- ✅ Replacing missing bans by random champions
- 🔄 Data augmentation techniques (mirror drafts, partial states)

### Machine Learning (Planned)
- 🔄 Feature engineering (champion synergies, counter-picks, meta stats)
- 🔄 Model training (XGBoost, LightGBM, Neural Networks)
- 🔄 Real-time prediction API
- 🔄 Performance evaluation and metrics

---

## 📁 Project Structure

```
lol-draft-prediction/
│   src
|   └── dataset.py              # Main data collection script (improved version)
|   └── helper.py               # API request utilities and helper functions
│
├── datasets/               # Collected match data (JSON files)
│   └── REGION_picks_and_bans_YYYYMMDD_HHMMSS.json
│
├── logs/                   # Log files
│   └── data_scrapping.log
│
├── models/                 # Trained ML models (future)
│
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore file
├── pyproject.toml             # Formatter config
├── .pre-commit-config.yaml    # Pre-commit config
└── README.md                  # This file
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- Riot Games API Key ([Get one here](https://developer.riotgames.com/))
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/lol-draft-prediction.git
cd lol-draft-prediction
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure API key**
```bash
# Edit dataset.py and add your Riot API key
API_KEY=RGAPI-your-api-key-here
```
---

## 📊 Data Collection

### Quick Start

Collect a small dataset for testing:

```bash
python dataset.py
```

Default configuration:
- Region: Europe
- Queue: Ranked Solo/Duo 5v5 (RANKED_SOLO_5x5)
- Players: 10 Challenger players
- Games per player: 50 games per player
- Elo: Challenger
- Estimated time: ~10-15 minutes
- Expected output: ~300-400 unique matches

### Advanced Collection

For larger datasets, modify the configuration in `dataset.py`:

```python
dataset = Dataset(
    region="EUROPE",           # EUROPE, KOREA, AMERICA, ASIA
    queue="RANKED_SOLO_5x5",
    game_count=100,            # Matches per player
    player_count=200,          # Number of Challenger players
    elo = "challenger"         # Elo of players
)
```

### Multi-Region Collection

```python
regions = ["EUROPE", "KOREA", "AMERICA"]
all_matches = []

for region in regions:
    dataset = Dataset(region=region, ...)
    matches = dataset.extract_match_data()
    all_matches.extend(matches)
```

### Data Collection Strategy

| Phase | Target | Time Estimate | Purpose |
|-------|--------|--------------|---------|
| **Phase 1: MVP** | 10,000 matches | 12-24 hours | Initial prototype, baseline model |
| **Phase 2: Production** | 50,000 matches | 3-5 days | Robust model with good performance |
| **Phase 3: Optimization** | 100,000+ matches | Ongoing | Competitive-level model, continuous updates |

### Important Notes

⚠️ **API Rate Limits**: The Riot API has strict rate limits (20 req/s, 100 req/2min). The scraper handles this automatically but collection takes time.

⚠️ **API Key Expiration**: Development API keys expire after 24 hours. Remember to refresh your key.

---

## 💻 Usage

### 1. Collect Data

```bash
# Basic collection
python dataset.py

# The script will:
# - Fetch Challenger player list
# - Retrieve their recent matches
# - Extract draft data with pick order
# - Save to Datasets/draft_with_order_YYYYMMDD_HHMMSS.json
```

### 2. Analyze Dataset Quality

```bash
python draft_analysis.py

# Provides:
# - Data validation (completeness, consistency)
# - Champion statistics (pick rates, win rates, ban rates)
# - Draft patterns (first pick priority, side win rates)
# - Data freshness (patch versions, date range)
```

### 3. Train Model (Coming Soon)

```bash
python train_model.py --data Datasets/latest.json --model xgboost
```

---

## 📦 Dataset Structure

### Optimal Format (with Timeline API)

```json
{
  "match_id": "EUW1_7566749902",
  "game_version": "15.20",
  "game_duration": 36.85,
  "game_creation": "2025-10-13 16:18:36",
  "blue_side_win": true,
  
  "bans": [
    {
      "side": "blue",
      "championId": 84,
      "order": 1
    },
    {
      "side": "red",
      "championId": 157,
      "order": 2
    }
    // ... 10 bans total
  ],
  
  "picks": [
    {
      "side": "blue",
      "championId": 64,
      "position": "JUNGLE",
      "order": 1
    },
    {
      "side": "red",
      "championId": 22,
      "position": "BOTTOM",
      "order": 2
    }
    // ... 10 picks total
  ]
}
```

### Draft Order (Standard Ranked)

The standard draft order follows this pattern:

**Bans Phase 1** (6 bans):
1. Blue ban (1)
2. Red ban (2)
3. Blue ban (3)
4. Red ban (4)
5. Blue ban (5)
6. Red ban (6)

**Picks Phase 1** (6 picks):
1. Blue pick (1)
2. Red pick (2)
3. Red pick (3)
4. Blue pick (4)
5. Blue pick (5)
6. Red pick (6)

**Bans Phase 2** (4 bans):
1. Red ban (7)
2. Blue ban (8)
3. Red ban (9)
4. Blue ban (10)

**Picks Phase 2** (4 picks):
1. Red pick (7)
2. Blue pick (8)
3. Blue pick (9)
4. Red pick (10)

---

## 🤖 Machine Learning Pipeline

### Feature Engineering (Planned)

**Champion Features:**
- Pick rate, ban rate, win rate (from dataset)
- Role flexibility (flex picks)
- Champion difficulty
- Meta tier (S, A, B, C, D)

**Context Features:**
- Current draft state (picks/bans so far)
- Side (blue vs red)
- Pick order position (early vs late)
- Patch version

**Synergy Features:**
- Team composition type (poke, engage, split-push, etc.)
- Lane matchups
- Jungle-lane synergies
- CC chain potential

**Counter Features:**
- Lane counter relationships
- Team fight counter potential
- Champion-specific counters

### Model Architecture (Planned)

**Phase 1: Baseline Models**
- Random Forest
- XGBoost
- LightGBM

**Phase 2: Advanced Models**
- Deep Neural Networks
- Recurrent Neural Networks (for sequential draft)
- Transformer-based models

**Phase 3: Ensemble**
- Model stacking
- Weighted voting
- Context-aware model selection

### Evaluation Metrics

- **Top-K Accuracy**: Is the correct pick in the top K predictions?
- **Win Rate Correlation**: Do predicted picks correlate with wins?
- **Position Accuracy**: Correct position predicted?
- **Meta Awareness**: Does the model follow current meta trends?

---

## 🗺️ Roadmap

### ✅ Phase 1: Data Collection (Current)
- [x] Implement Riot API scraper
- [x] Rate limiting and error handling
- [x] Timeline API integration for pick order
- [x] Data validation and cleaning
- [x] Multi-region support

### 🔄 Phase 2: Data Processing & EDA
- [ ] Exploratory Data Analysis notebooks
- [ ] Feature engineering pipeline
- [ ] Champion similarity mapping
- [ ] Meta analysis tools
- [ ] Data augmentation implementation

### 📅 Phase 3: Model Development
- [ ] Baseline model implementation
- [ ] Hyperparameter tuning
- [ ] Model evaluation framework
- [ ] Cross-validation strategy
- [ ] Model comparison tools

### 📅 Phase 4: Deployment
- [ ] REST API for predictions
- [ ] Web interface
- [ ] Real-time draft tracking
- [ ] Model updates with patch changes
- [ ] Performance monitoring

---

## 🎓 Technical Challenges

### 1. **Pick Order Accuracy** ⭐ CRITICAL
- **Problem**: Without Timeline API, impossible to know exact pick order by position
- **Solution**: Shuffling with weights according to personal experience
- **Impact**: Pick order is crucial for counter-picks and flex picks detection

### 2. **Data Volume Requirements**
- **Minimum viable**: 10,000 matches
- **Recommended**: 50,000 matches
- **Optimal**: 100,000+ matches
- **Reason**: ~170 champions × 5 positions = massive combinatorial space

### 3. **Meta Drift**
- **Problem**: Game balance patches change champion viability
- **Solution**: Weight recent matches higher, continuous model retraining
- **Strategy**: Discard matches older than 2 patches

### 4. **Imbalanced Data**
- **Problem**: Popular champions have 1000x more data than niche picks
- **Solution**: Stratified sampling, champion-specific models, data augmentation

### 5. **Sequential Decision Making**
- **Problem**: Each pick influences future picks (not independent)
- **Solution**: Recurrent models (LSTM/GRU) or Transformer architecture

---

## 📈 Dataset Statistics

Example statistics from a 10,000 match dataset:

```
Total Matches: 10,000
Unique Champions Picked: 162/168 (96.4%)
Unique Champions Banned: 148/168 (88.1%)

Pick Rate Leaders:
  1. K'Sante: 1,245 picks (12.4%)
  2. Viego: 1,189 picks (11.9%)
  3. Ahri: 1,056 picks (10.6%)

Ban Rate Leaders:
  1. Samira: 4,567 bans (45.7%)
  2. Yasuo: 3,892 bans (38.9%)
  3. Zed: 3,456 bans (34.6%)

Blue Side Win Rate: 51.2%
Red Side Win Rate: 48.8%

Average Game Duration: 28.5 minutes
```

---

## ⚠️ Disclaimer

This project is for educational and research purposes only. 

- This project is not endorsed by Riot Games
- Respect Riot's API Terms of Service
- Do not use for commercial purposes without proper authorization
- The predictions are probabilistic and should not be considered absolute

---

## 🙏 Acknowledgments

- **Riot Games** for providing the official API
- **League of Legends Community** for inspiration and meta insights
- **Python Data Science Community** for amazing ML libraries

---

## 📞 Contact

For questions, suggestions, or collaboration:

- Email: vuillemenotmaelian@gmail.com

---

**Made with ❤️ by Maelian Vuillemenot**

*Last Updated: October 2024*
