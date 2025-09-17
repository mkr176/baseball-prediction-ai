#!/usr/bin/env python3
"""
Baseball Game Predictor - Interactive Command Line Interface

Adapted from tennis predictor for baseball game predictions.
Uses the 85% accuracy approach adapted for baseball analytics.

Usage: python predict_game.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from baseball_predictor import BaseballPredictor
from baseball_elo_system import BaseballEloSystem
from baseball_data_collector import BaseballDataCollector
import argparse

def display_header():
    """Display the program header"""
    print("\n" + "=" * 70)
    print("⚾ BASEBALL GAME PREDICTOR")
    print("Based on 85% Accuracy Tennis Model Adaptation")
    print("=" * 70)
    print("🎯 Target Accuracy: 85%+")
    print("🧠 Features: ELO ratings, pitching matchups, environment factors")
    print("📊 Model: XGBoost with comprehensive baseball analytics")
    print("=" * 70)

def get_team_input(prompt, teams):
    """Get team input with validation"""
    while True:
        team = input(f"\n{prompt}: ").strip()

        if not team:
            print("❌ Please enter a team name")
            continue

        # Try exact match first
        if team in teams:
            return team

        # Try case-insensitive partial match
        matches = [t for t in teams if team.lower() in t.lower()]

        if len(matches) == 1:
            print(f"✅ Found: {matches[0]}")
            return matches[0]
        elif len(matches) > 1:
            print(f"Multiple matches found:")
            for i, match in enumerate(matches, 1):
                print(f"  {i}. {match}")
            try:
                choice = int(input("Select team (number): ")) - 1
                if 0 <= choice < len(matches):
                    return matches[choice]
                else:
                    print("❌ Invalid selection")
            except ValueError:
                print("❌ Please enter a valid number")
        else:
            print(f"❌ Team '{team}' not found")
            print("💡 Try: Yankees, Dodgers, Red Sox, etc.")

def get_pitcher_input(team, team_pitchers):
    """Get pitcher input for a team"""
    print(f"\n🎯 Starting pitcher for {team}:")

    if team in team_pitchers:
        print("Common pitchers:")
        for i, pitcher in enumerate(team_pitchers[team][:4], 1):
            print(f"  {i}. {pitcher}")
        print("  5. Enter custom pitcher")

        while True:
            try:
                choice = input("Select pitcher (1-5) or press Enter to skip: ").strip()

                if not choice:
                    return None

                choice_num = int(choice)
                if 1 <= choice_num <= 4:
                    return team_pitchers[team][choice_num - 1]
                elif choice_num == 5:
                    return input("Enter pitcher name: ").strip()
                else:
                    print("❌ Invalid selection")
            except ValueError:
                print("❌ Please enter a valid number")
    else:
        pitcher = input("Enter pitcher name (or press Enter to skip): ").strip()
        return pitcher if pitcher else None

def get_environment_input():
    """Get game environment input"""
    environments = {
        '1': 'home',
        '2': 'away',
        '3': 'neutral',
        '4': 'dome'
    }

    print(f"\n🏟️ Game environment:")
    print("  1. Home")
    print("  2. Away")
    print("  3. Neutral site")
    print("  4. Dome/Indoor")

    while True:
        choice = input("Select environment (1-4): ").strip()
        if choice in environments:
            return environments[choice]
        print("❌ Invalid selection")

def get_game_type_input():
    """Get game type input"""
    game_types = {
        '1': 'regular',
        '2': 'division',
        '3': 'interleague',
        '4': 'playoffs',
        '5': 'world_series'
    }

    print(f"\n🏆 Game importance:")
    print("  1. Regular season")
    print("  2. Division game")
    print("  3. Interleague")
    print("  4. Playoffs")
    print("  5. World Series")

    while True:
        choice = input("Select game type (1-5): ").strip()
        if choice in game_types:
            return game_types[choice]
        print("❌ Invalid selection")

def display_prediction(prediction):
    """Display prediction results in a formatted way"""
    if not prediction:
        print("❌ Unable to make prediction")
        return

    print("\n" + "=" * 70)
    print("🔮 PREDICTION RESULTS")
    print("=" * 70)

    # Main prediction
    print(f"🏟️ {prediction['team1']} vs {prediction['team2']}")
    print(f"📍 Environment: {prediction['environment'].title()}")
    print(f"🏆 Game type: {prediction['game_type'].title()}")

    if prediction.get('team1_pitcher') and prediction.get('team2_pitcher'):
        print(f"⚾ Pitching: {prediction['team1_pitcher']} vs {prediction['team2_pitcher']}")

    print(f"\n🎯 PREDICTED WINNER: {prediction['predicted_winner']}")
    print(f"🔥 Confidence: {prediction['confidence']:.1%}")

    # Win probabilities
    print(f"\n📊 Win Probabilities:")
    print(f"   {prediction['team1']}: {prediction['team1_win_probability']:.1%}")
    print(f"   {prediction['team2']}: {prediction['team2_win_probability']:.1%}")

    # Confidence level
    if prediction['is_high_confidence']:
        print(f"✅ HIGH CONFIDENCE PREDICTION (>{prediction.get('confidence_threshold', 75)}%)")
    else:
        print(f"⚠️  Medium confidence prediction")

    # ELO comparison
    if 'elo_favorite' in prediction:
        print(f"\n⚖️  ELO Analysis:")
        print(f"   ELO favorite: {prediction['elo_favorite']}")
        print(f"   ELO confidence: {prediction['elo_confidence']:.1%}")
        if 'elo_difference' in prediction:
            print(f"   ELO difference: {prediction['elo_difference']:.0f} points")

    # Model info
    print(f"\n🧠 Model: {prediction['prediction_method']}")
    print(f"🎯 Target Accuracy: {prediction['model_accuracy_target']:.0%}")

    print("=" * 70)

def interactive_mode():
    """Run interactive prediction mode"""
    # Initialize predictor
    predictor = BaseballPredictor()

    # Initialize with ELO system if model not available
    if not predictor.elo_system:
        print("🔄 Initializing ELO system...")
        predictor.elo_system = BaseballEloSystem()

    # Get team list from data collector
    collector = BaseballDataCollector()
    teams = collector.teams
    team_pitchers = collector.team_pitchers

    display_header()

    print(f"\n📋 Available teams: {len(teams)} MLB teams")
    print("💡 You can type partial team names (e.g., 'Yankees', 'Dodgers')")

    while True:
        try:
            # Get teams
            print(f"\n" + "-" * 50)
            print("🆚 TEAM SELECTION")
            team1 = get_team_input("Enter first team", teams)
            team2 = get_team_input("Enter second team", teams)

            if team1 == team2:
                print("❌ Teams must be different!")
                continue

            # Get pitchers
            print(f"\n" + "-" * 50)
            print("⚾ PITCHING MATCHUP")
            team1_pitcher = get_pitcher_input(team1, team_pitchers)
            team2_pitcher = get_pitcher_input(team2, team_pitchers)

            # Get game context
            print(f"\n" + "-" * 50)
            print("🎮 GAME CONTEXT")
            environment = get_environment_input()
            game_type = get_game_type_input()

            # Make prediction
            print(f"\n🔄 Making prediction...")
            prediction = predictor.predict_game(
                team1=team1,
                team2=team2,
                environment=environment,
                game_type=game_type,
                team1_pitcher=team1_pitcher,
                team2_pitcher=team2_pitcher
            )

            # Display results
            display_prediction(prediction)

            # Ask for another prediction
            print(f"\n🔄 Make another prediction?")
            choice = input("Enter 'y' for yes, anything else to quit: ").strip().lower()
            if choice not in ['y', 'yes']:
                break

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again...")

def batch_mode(args):
    """Run batch prediction mode"""
    predictor = BaseballPredictor()

    if not predictor.elo_system:
        predictor.elo_system = BaseballEloSystem()

    prediction = predictor.predict_game(
        team1=args.team1,
        team2=args.team2,
        environment=args.environment,
        game_type=args.game_type,
        team1_pitcher=args.team1_pitcher,
        team2_pitcher=args.team2_pitcher
    )

    display_prediction(prediction)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Baseball Game Predictor')

    # Batch mode arguments
    parser.add_argument('--team1', help='First team name')
    parser.add_argument('--team2', help='Second team name')
    parser.add_argument('--environment', default='home',
                       choices=['home', 'away', 'neutral', 'dome'],
                       help='Game environment')
    parser.add_argument('--game-type', default='regular',
                       choices=['regular', 'division', 'interleague', 'playoffs', 'world_series'],
                       help='Game type/importance')
    parser.add_argument('--team1-pitcher', help='Team 1 starting pitcher')
    parser.add_argument('--team2-pitcher', help='Team 2 starting pitcher')

    args = parser.parse_args()

    # Run in batch mode if teams are provided
    if args.team1 and args.team2:
        batch_mode(args)
    else:
        interactive_mode()

if __name__ == "__main__":
    main()