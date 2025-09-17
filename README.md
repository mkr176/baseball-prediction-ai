# ⚾ Baseball Prediction AI

A comprehensive baseball game prediction system adapted from the 85% accuracy tennis predictor model. Uses advanced ELO rating systems, machine learning, and comprehensive baseball analytics to predict MLB game outcomes.

## 🎯 Features

- **High Accuracy Target**: 85%+ prediction accuracy (adapted from successful tennis model)
- **ELO Rating System**: Team and pitcher-specific ELO ratings with environment factors
- **Comprehensive Analytics**: Pitching matchups, home/away advantage, game importance
- **Multiple ML Models**: XGBoost, LightGBM, and Random Forest implementations
- **Interactive Interface**: Command-line tool for real-time predictions
- **Realistic Data**: Generates comprehensive baseball statistics and scenarios

## 📊 Key Components

### 🧠 Core Prediction Features
- Team overall ELO ratings
- Pitching vs hitting ELO ratings
- Environment-specific ratings (home/away/neutral/dome)
- Recent form and momentum tracking
- Pitcher matchup analysis
- Game importance weighting (regular season vs playoffs)
- Head-to-head performance tracking

### ⚾ Baseball-Specific Analytics
- Starting pitcher ELO ratings
- Team pitching vs hitting strength
- Home field advantage calculations
- Division game importance
- Playoff and World Series weighting
- Season progression tracking

## 🚀 Quick Start

### Installation

1. Clone or download the project
2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Training the Model

First, train the prediction model:
```bash
python train_baseball_model.py
```

This will:
- Generate 8,000 training games with realistic statistics
- Build ELO ratings chronologically
- Train multiple ML models (XGBoost, LightGBM, Random Forest)
- Save the best performing model
- Target: 85%+ accuracy

### Making Predictions

#### Interactive Mode
```bash
python predict_game.py
```

Follow the prompts to select teams, pitchers, and game context.

#### Batch Mode
```bash
python predict_game.py --team1 "New York Yankees" --team2 "Boston Red Sox" --environment home --game-type division --team1-pitcher "Gerrit Cole" --team2-pitcher "Chris Sale"
```

## 📁 Project Structure

```
baseball-prediction-ai/
├── src/
│   ├── baseball_elo_system.py      # ELO rating system
│   ├── baseball_predictor.py       # Main prediction class
│   └── baseball_data_collector.py  # Training data generation
├── models/                         # Trained models (created after training)
├── data/                          # Training and sample data
├── predict_game.py                # Interactive prediction interface
├── train_baseball_model.py        # Model training script
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

## 🔬 Model Architecture

### ELO Rating System
- **Base ELO**: 1500 for new teams/pitchers
- **Environment Weights**: Home (1.2x), Away (0.9x), Neutral (1.0x)
- **Game Importance**: World Series (50), Playoffs (40), Division (30), Regular (20)
- **Dynamic Updates**: Ratings adjust after each game based on performance

### Machine Learning Features
1. **ELO-based features** (most important):
   - Team ELO differences
   - Pitcher ELO differences
   - Environment-specific ratings

2. **Performance metrics**:
   - Recent form (last 10 games)
   - Momentum tracking
   - Win rate comparisons

3. **Game context**:
   - Home/away advantage
   - Game importance
   - Seasonal progression

### Model Selection
The system trains multiple models and selects the best performer:
- **XGBoost**: Primary model (optimized hyperparameters)
- **LightGBM**: High-performance alternative
- **Random Forest**: Baseline comparison

## 🎯 Accuracy Targets

Based on the successful tennis predictor approach:
- **Target Accuracy**: 85%+
- **High Confidence Predictions**: 75%+ confidence threshold
- **ELO System Accuracy**: ~70% (foundation feature)

## 📊 Sample Usage

```python
from src.baseball_predictor import BaseballPredictor

predictor = BaseballPredictor()

# Make a prediction
prediction = predictor.predict_game(
    team1="New York Yankees",
    team2="Boston Red Sox",
    environment="home",
    game_type="division",
    team1_pitcher="Gerrit Cole",
    team2_pitcher="Chris Sale"
)

print(f"Winner: {prediction['predicted_winner']}")
print(f"Confidence: {prediction['confidence']:.1%}")
```

## 🏟️ Supported Teams

All 30 MLB teams with realistic pitcher rosters:
- **AL East**: Yankees, Red Sox, Blue Jays, Rays, Orioles
- **AL Central**: Astros, Twins, White Sox, Guardians, Tigers
- **AL West**: Mariners, Angels, Rangers, Athletics, Royals
- **NL East**: Braves, Mets, Phillies, Marlins, Nationals
- **NL Central**: Brewers, Cardinals, Cubs, Reds, Pirates
- **NL West**: Dodgers, Padres, Giants, Diamondbacks, Rockies

## 🎮 Game Types Supported

- **Regular Season**: Standard games
- **Division**: Division rivalry games (higher importance)
- **Interleague**: Cross-league matchups
- **Playoffs**: Postseason games
- **World Series**: Championship games (highest importance)

## 🔧 Technical Details

### Dependencies
- Python 3.7+
- pandas, numpy (data handling)
- scikit-learn (ML foundation)
- xgboost, lightgbm (advanced ML models)
- matplotlib (visualization)

### Model Training Process
1. Generate realistic game data (teams, scores, pitcher stats)
2. Process games chronologically to build ELO ratings
3. Extract features for each game prediction
4. Train multiple ML models with cross-validation
5. Select best model and save for predictions

### Adaptation from Tennis Model
This system adapts the successful tennis predictor approach:
- **Teams** instead of individual players
- **Pitchers** as key individual factors (like tennis players)
- **Environments** instead of court surfaces
- **Game importance** instead of tournament levels
- **Season progression** instead of ranking cycles

## 🎯 Future Enhancements

- Real MLB data integration (MLB Stats API)
- Weather factor integration
- Injury and roster status
- Advanced pitching metrics (pitch mix, velocity)
- Betting odds comparison
- Historical performance validation

## 📈 Performance Monitoring

The system tracks:
- Model accuracy over time
- Feature importance rankings
- ELO rating progressions
- Prediction confidence distributions

## 🤝 Based on Tennis Predictor

This baseball predictor is adapted from a tennis match prediction system that achieved 85% accuracy. The core ELO system and ML approach have been successfully translated to baseball analytics while maintaining the proven methodology.

---

**🎯 Target: 85%+ Accuracy | ⚾ Comprehensive Baseball Analytics | 🧠 Advanced ML Pipeline**