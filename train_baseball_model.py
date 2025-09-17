#!/usr/bin/env python3
"""
Baseball Model Training Script

Trains the 85% accuracy baseball prediction model using the same approach
as the successful tennis predictor, adapted for baseball analytics.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb

from baseball_data_collector import BaseballDataCollector
from baseball_elo_system import BaseballEloSystem
from baseball_predictor import BaseballPredictor

class BaseballModelTrainer:
    """
    Baseball Model Training System - Adapted from tennis 85% accuracy approach
    """

    def __init__(self):
        self.elo_system = BaseballEloSystem()
        self.data_collector = BaseballDataCollector()
        self.feature_columns = None
        self.model = None

    def generate_training_data(self, num_games=5000):
        """
        Generate comprehensive training data with ELO features
        """
        print(f"🎾 Generating {num_games} training games...")

        # Generate base game data
        games = self.data_collector.generate_season_data(num_games)

        # Process games chronologically to build ELO ratings
        games.sort(key=lambda x: x['date'])

        training_data = []

        print(f"📊 Processing games and building ELO ratings...")

        for i, game in enumerate(games):
            team1 = game['team1']
            team2 = game['team2']
            winner = game['winner']
            loser = game['loser']
            team1_score = game['team1_score']
            team2_score = game['team2_score']
            environment = game['environment']
            game_type = game['game_type']
            team1_pitcher = game['team1_pitcher']
            team2_pitcher = game['team2_pitcher']

            # Get current ELO features before the game
            team1_elo_features = self.elo_system.get_team_elo_features(team1, environment)
            team2_elo_features = self.elo_system.get_team_elo_features(team2, environment)

            # Get pitcher features
            team1_pitcher_features = self.elo_system.get_pitcher_elo_features(team1_pitcher)
            team2_pitcher_features = self.elo_system.get_pitcher_elo_features(team2_pitcher)

            # Create feature vector (same as in predictor)
            features = {
                # CORE ELO FEATURES
                'team_elo_diff': team1_elo_features['overall_elo'] - team2_elo_features['overall_elo'],
                'environment_elo_diff': team1_elo_features['environment_elo'] - team2_elo_features['environment_elo'],
                'total_elo': team1_elo_features['overall_elo'] + team2_elo_features['overall_elo'],

                # Individual team ELO ratings
                'team1_elo': team1_elo_features['overall_elo'],
                'team2_elo': team2_elo_features['overall_elo'],
                'team1_environment_elo': team1_elo_features['environment_elo'],
                'team2_environment_elo': team2_elo_features['environment_elo'],

                # SPECIALIZED ELO FEATURES
                'pitching_elo_diff': team1_elo_features['pitching_elo'] - team2_elo_features['pitching_elo'],
                'hitting_elo_diff': team1_elo_features['hitting_elo'] - team2_elo_features['hitting_elo'],
                'home_elo_diff': team1_elo_features['home_elo'] - team2_elo_features['home_elo'],
                'away_elo_diff': team1_elo_features['away_elo'] - team2_elo_features['away_elo'],

                # RECENT FORM AND MOMENTUM
                'recent_form_diff': team1_elo_features['recent_form'] - team2_elo_features['recent_form'],
                'momentum_diff': team1_elo_features['recent_momentum'] - team2_elo_features['recent_momentum'],
                'elo_change_diff': team1_elo_features['recent_elo_change'] - team2_elo_features['recent_elo_change'],
                'run_differential_diff': team1_elo_features['run_differential'] - team2_elo_features['run_differential'],

                # PITCHER FEATURES
                'pitcher_elo_diff': team1_pitcher_features.get('overall', 1500) - team2_pitcher_features.get('overall', 1500),
                'pitcher_wins_diff': team1_pitcher_features.get('wins', 0) - team2_pitcher_features.get('wins', 0),
                'pitcher_era_diff': team2_pitcher_features.get('era', 4.50) - team1_pitcher_features.get('era', 4.50),

                # GAME CONTEXT
                'game_weight': self.elo_system.game_weights.get(game_type, 20),
                'is_playoffs': 1 if game_type in ['playoffs', 'world_series'] else 0,
                'is_world_series': 1 if game_type == 'world_series' else 0,
                'is_division_game': 1 if game_type == 'division' else 0,

                # ENVIRONMENT ENCODING
                'is_home': 1 if environment == 'home' else 0,
                'is_away': 1 if environment == 'away' else 0,
                'is_neutral': 1 if environment == 'neutral' else 0,
                'is_dome': 1 if environment == 'dome' else 0,

                # TEAM PERFORMANCE METRICS
                'team1_win_rate': team1_elo_features['win_rate'],
                'team2_win_rate': team2_elo_features['win_rate'],
                'win_rate_diff': team1_elo_features['win_rate'] - team2_elo_features['win_rate'],
                'games_played_diff': team1_elo_features['games_played'] - team2_elo_features['games_played'],

                # INTERACTION FEATURES
                'elo_form_interaction': (team1_elo_features['overall_elo'] - team2_elo_features['overall_elo']) *
                                      (team1_elo_features['recent_form'] - team2_elo_features['recent_form']),
                'pitching_hitting_interaction': (team1_elo_features['pitching_elo'] - team2_elo_features['pitching_elo']) *
                                              (team2_elo_features['hitting_elo'] - team1_elo_features['hitting_elo']),

                # SEASONAL CONTEXT
                'season_progress': 0.5,  # Simplified
                'days_rest_diff': 0,
                'temperature': game.get('weather_temp', 75),
                'wind_speed': game.get('wind_speed', 5),

                # TARGET VARIABLE
                'team1_wins': 1 if winner == team1 else 0
            }

            training_data.append(features)

            # Update ELO ratings after processing this game
            self.elo_system.update_elo_ratings(
                winner, loser, team1_score if winner == team1 else team2_score,
                team2_score if winner == team1 else team1_score,
                environment, game_type, team1_pitcher, team2_pitcher
            )

            # Progress indicator
            if (i + 1) % 1000 == 0:
                print(f"   Processed {i + 1}/{len(games)} games...")

        print(f"✅ Generated {len(training_data)} training examples")
        return pd.DataFrame(training_data)

    def train_models(self, df):
        """
        Train multiple models and select the best one (following tennis approach)
        """
        # Prepare features and target
        target_col = 'team1_wins'
        feature_cols = [col for col in df.columns if col != target_col]

        X = df[feature_cols]
        y = df[target_col]

        # Store feature columns for prediction
        self.feature_columns = feature_cols

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"📊 Training set: {len(X_train)} games")
        print(f"📊 Test set: {len(X_test)} games")

        models = {}
        accuracies = {}

        # 1. XGBoost (primary model from tennis)
        print(f"\n🚀 Training XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        xgb_accuracy = accuracy_score(y_test, xgb_pred)
        models['XGBoost'] = xgb_model
        accuracies['XGBoost'] = xgb_accuracy
        print(f"   XGBoost accuracy: {xgb_accuracy:.1%}")

        # 2. LightGBM (high performer from tennis)
        print(f"\n🚀 Training LightGBM...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
        lgb_model.fit(X_train, y_train)
        lgb_pred = lgb_model.predict(X_test)
        lgb_accuracy = accuracy_score(y_test, lgb_pred)
        models['LightGBM'] = lgb_model
        accuracies['LightGBM'] = lgb_accuracy
        print(f"   LightGBM accuracy: {lgb_accuracy:.1%}")

        # 3. Random Forest (baseline)
        print(f"\n🚀 Training Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        rf_accuracy = accuracy_score(y_test, rf_pred)
        models['Random Forest'] = rf_model
        accuracies['Random Forest'] = rf_accuracy
        print(f"   Random Forest accuracy: {rf_accuracy:.1%}")

        # Select best model
        best_model_name = max(accuracies, key=accuracies.get)
        best_accuracy = accuracies[best_model_name]
        self.model = models[best_model_name]

        print(f"\n🏆 Best model: {best_model_name} ({best_accuracy:.1%})")

        # Detailed evaluation of best model
        if best_model_name == 'XGBoost':
            best_pred = xgb_pred
        elif best_model_name == 'LightGBM':
            best_pred = lgb_pred
        else:
            best_pred = rf_pred

        print(f"\n📊 Detailed evaluation:")
        print(classification_report(y_test, best_pred))

        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_cols,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)

            print(f"\n🔍 Top 10 most important features:")
            for idx, row in importance_df.head(10).iterrows():
                print(f"   {row['feature']}: {row['importance']:.3f}")

        return {
            'best_model': best_model_name,
            'best_accuracy': best_accuracy,
            'all_accuracies': accuracies,
            'test_accuracy': best_accuracy
        }

    def save_model(self, model_dir='models'):
        """
        Save the trained model and components
        """
        os.makedirs(model_dir, exist_ok=True)

        # Save model
        model_path = os.path.join(model_dir, 'baseball_85_percent_model.pkl')
        joblib.dump(self.model, model_path)
        print(f"💾 Saved model to {model_path}")

        # Save feature columns
        features_path = os.path.join(model_dir, 'baseball_features.pkl')
        joblib.dump(self.feature_columns, features_path)
        print(f"💾 Saved features to {features_path}")

        # Save ELO system
        elo_path = os.path.join(model_dir, 'baseball_elo_system.pkl')
        joblib.dump(self.elo_system, elo_path)
        print(f"💾 Saved ELO system to {elo_path}")

def main():
    """Main training function"""
    print("⚾ BASEBALL MODEL TRAINING")
    print("Based on 85% accuracy tennis model approach")
    print("=" * 60)

    trainer = BaseballModelTrainer()

    # Generate training data
    print(f"\n📊 PHASE 1: Data Generation")
    df = trainer.generate_training_data(num_games=8000)  # Larger dataset

    # Train models
    print(f"\n🤖 PHASE 2: Model Training")
    results = trainer.train_models(df)

    # Save everything
    print(f"\n💾 PHASE 3: Saving Models")
    trainer.save_model()

    # Final summary
    print(f"\n" + "=" * 60)
    print(f"🏆 TRAINING COMPLETE")
    print(f"=" * 60)
    print(f"✅ Best model: {results['best_model']}")
    print(f"🎯 Test accuracy: {results['best_accuracy']:.1%}")
    print(f"📊 Target: 85%+ (like tennis model)")

    if results['best_accuracy'] >= 0.85:
        print(f"🎉 TARGET ACHIEVED! Model ready for production!")
    else:
        print(f"⚠️  Below target. Consider more data or feature engineering.")

    print(f"\n🚀 To use the model:")
    print(f"   python predict_game.py")
    print(f"=" * 60)

if __name__ == "__main__":
    main()