import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import json

class BaseballEloSystem:
    """
    Baseball ELO Rating System - Adapted from tennis 85% accuracy model.

    Key insights from the tennis model adapted for baseball:
    - ELO as primary feature for team strength evaluation
    - Stadium/environment-specific ELO (home/away advantage)
    - Pitcher-specific ratings for starting pitchers
    - Team hitting vs pitching strength separation
    - Season progression and momentum tracking
    """

    def __init__(self):
        self.default_elo = 1500
        self.k_factor_base = 32

        # Stadium/environment importance (adapted from tennis surfaces)
        self.environment_weights = {
            'home': 1.2,          # Home field advantage
            'away': 0.9,          # Away disadvantage
            'neutral': 1.0,       # Neutral site
            'dome': 1.1,          # Dome/indoor advantage
            'outdoor': 1.0        # Standard outdoor
        }

        # Game importance multipliers (adapted from tennis tournaments)
        self.game_weights = {
            'world_series': 50,    # World Series
            'playoffs': 40,        # Playoffs
            'division': 30,        # Division games
            'interleague': 25,     # Interleague games
            'regular': 20,         # Regular season
            'spring': 10,          # Spring training
            'exhibition': 5        # Exhibition games
        }

        # Team ELO ratings storage
        self.team_elo = {}              # Overall team ELO
        self.pitching_elo = {}          # Team pitching ELO
        self.hitting_elo = {}           # Team hitting ELO
        self.environment_elo = {}       # Environment-specific ELO
        self.elo_history = {}           # ELO progression over time
        self.team_stats = {}            # Additional team statistics
        self.pitcher_elo = {}           # Individual pitcher ELO

    def initialize_team(self, team_name):
        """Initialize ELO ratings for a new team"""
        if team_name not in self.team_elo:
            self.team_elo[team_name] = self.default_elo
            self.pitching_elo[team_name] = self.default_elo
            self.hitting_elo[team_name] = self.default_elo
            self.environment_elo[team_name] = {
                'home': self.default_elo,
                'away': self.default_elo,
                'neutral': self.default_elo,
                'dome': self.default_elo,
                'outdoor': self.default_elo
            }
            self.elo_history[team_name] = []
            self.team_stats[team_name] = {
                'games_played': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0.0,
                'runs_scored': 0,
                'runs_allowed': 0,
                'run_differential': 0
            }

    def initialize_pitcher(self, pitcher_name):
        """Initialize ELO ratings for a pitcher"""
        if pitcher_name not in self.pitcher_elo:
            self.pitcher_elo[pitcher_name] = {
                'overall': self.default_elo,
                'vs_lefties': self.default_elo,
                'vs_righties': self.default_elo,
                'games_started': 0,
                'wins': 0,
                'losses': 0,
                'era': 4.50
            }

    def get_k_factor(self, game_type='regular', environment='home', score_diff=0):
        """
        Calculate K-factor based on game importance and context
        Higher K-factor = more rating change for important games
        """
        base_k = self.game_weights.get(game_type.lower(), 20)
        env_multiplier = self.environment_weights.get(environment.lower(), 1.0)

        # Adjust for blowout vs close game
        if abs(score_diff) >= 10:  # Blowout
            score_multiplier = 0.8
        elif abs(score_diff) <= 1:  # Extra close game
            score_multiplier = 1.3
        elif abs(score_diff) <= 3:  # Close game
            score_multiplier = 1.1
        else:
            score_multiplier = 1.0

        return base_k * env_multiplier * score_multiplier

    def expected_score(self, team_a_elo, team_b_elo):
        """
        Calculate expected score (win probability) for team A
        Same formula as chess/tennis ELO system
        """
        rating_diff = team_a_elo - team_b_elo
        expected = 1 / (1 + 10 ** (-rating_diff / 400))
        return expected

    def update_elo_ratings(self, winner, loser, winner_score, loser_score,
                          environment='home', game_type='regular',
                          winner_pitcher=None, loser_pitcher=None, game_date=None):
        """
        Update ELO ratings after a game - core of the baseball model
        """
        # Initialize teams if needed
        self.initialize_team(winner)
        self.initialize_team(loser)

        if winner_pitcher:
            self.initialize_pitcher(winner_pitcher)
        if loser_pitcher:
            self.initialize_pitcher(loser_pitcher)

        if game_date is None:
            game_date = datetime.now()

        # Get current ELO ratings
        winner_elo = self.team_elo[winner]
        loser_elo = self.team_elo[loser]

        # Calculate expected outcomes
        winner_expected = self.expected_score(winner_elo, loser_elo)
        loser_expected = 1 - winner_expected

        # Calculate K-factor
        score_diff = winner_score - loser_score
        k_factor = self.get_k_factor(game_type, environment, score_diff)

        # Update overall ELO ratings
        winner_new_elo = winner_elo + k_factor * (1 - winner_expected)
        loser_new_elo = loser_elo + k_factor * (0 - loser_expected)

        self.team_elo[winner] = winner_new_elo
        self.team_elo[loser] = loser_new_elo

        # Update pitching/hitting ELO based on game performance
        # Pitching ELO influenced by runs allowed
        winner_runs_allowed_ratio = loser_score / max(1, (winner_score + loser_score))
        loser_runs_allowed_ratio = winner_score / max(1, (winner_score + loser_score))

        # Update pitching ELO (inverse correlation with runs allowed)
        self.pitching_elo[winner] += k_factor * 0.5 * (1 - winner_runs_allowed_ratio)
        self.pitching_elo[loser] -= k_factor * 0.5 * loser_runs_allowed_ratio

        # Update hitting ELO (correlation with runs scored)
        winner_runs_scored_ratio = winner_score / max(1, (winner_score + loser_score))
        loser_runs_scored_ratio = loser_score / max(1, (winner_score + loser_score))

        self.hitting_elo[winner] += k_factor * 0.5 * winner_runs_scored_ratio
        self.hitting_elo[loser] -= k_factor * 0.5 * (1 - loser_runs_scored_ratio)

        # Update environment-specific ELO
        if environment in self.environment_elo[winner]:
            self.environment_elo[winner][environment] += k_factor * 0.3 * (1 - winner_expected)
        if environment in self.environment_elo[loser]:
            self.environment_elo[loser][environment] += k_factor * 0.3 * (0 - loser_expected)

        # Update pitcher ELO if provided
        if winner_pitcher and winner_pitcher in self.pitcher_elo:
            self.pitcher_elo[winner_pitcher]['wins'] += 1
            self.pitcher_elo[winner_pitcher]['overall'] += k_factor * 0.4 * (1 - winner_expected)

        if loser_pitcher and loser_pitcher in self.pitcher_elo:
            self.pitcher_elo[loser_pitcher]['losses'] += 1
            self.pitcher_elo[loser_pitcher]['overall'] += k_factor * 0.4 * (0 - loser_expected)

        # Update team statistics
        self.team_stats[winner]['games_played'] += 1
        self.team_stats[winner]['wins'] += 1
        self.team_stats[winner]['runs_scored'] += winner_score
        self.team_stats[winner]['runs_allowed'] += loser_score

        self.team_stats[loser]['games_played'] += 1
        self.team_stats[loser]['losses'] += 1
        self.team_stats[loser]['runs_scored'] += loser_score
        self.team_stats[loser]['runs_allowed'] += winner_score

        # Update win rates and run differentials
        for team in [winner, loser]:
            stats = self.team_stats[team]
            stats['win_rate'] = stats['wins'] / stats['games_played']
            stats['run_differential'] = stats['runs_scored'] - stats['runs_allowed']

        # Store ELO history
        self.elo_history[winner].append({
            'date': game_date,
            'elo': winner_new_elo,
            'opponent': loser,
            'result': 'W',
            'score': f"{winner_score}-{loser_score}"
        })

        self.elo_history[loser].append({
            'date': game_date,
            'elo': loser_new_elo,
            'opponent': winner,
            'result': 'L',
            'score': f"{loser_score}-{winner_score}"
        })

        return {
            'winner_elo_change': winner_new_elo - winner_elo,
            'loser_elo_change': loser_new_elo - loser_elo,
            'k_factor': k_factor,
            'expected_winner_prob': winner_expected
        }

    def predict_game_outcome(self, team1, team2, environment='home',
                           team1_pitcher=None, team2_pitcher=None):
        """
        Predict game outcome between two teams
        """
        self.initialize_team(team1)
        self.initialize_team(team2)

        # Get base ELO ratings
        team1_elo = self.team_elo[team1]
        team2_elo = self.team_elo[team2]

        # Adjust for environment
        if environment in self.environment_elo[team1]:
            team1_elo = (team1_elo + self.environment_elo[team1][environment]) / 2
        if environment in self.environment_elo[team2]:
            team2_elo = (team2_elo + self.environment_elo[team2][environment]) / 2

        # Factor in pitching matchup if provided
        if team1_pitcher and team1_pitcher in self.pitcher_elo:
            pitcher1_elo = self.pitcher_elo[team1_pitcher]['overall']
            team1_elo = (team1_elo * 0.7) + (pitcher1_elo * 0.3)

        if team2_pitcher and team2_pitcher in self.pitcher_elo:
            pitcher2_elo = self.pitcher_elo[team2_pitcher]['overall']
            team2_elo = (team2_elo * 0.7) + (pitcher2_elo * 0.3)

        # Calculate win probabilities
        team1_win_prob = self.expected_score(team1_elo, team2_elo)
        team2_win_prob = 1 - team1_win_prob

        # Determine favorite
        favorite = team1 if team1_win_prob > 0.5 else team2
        confidence = max(team1_win_prob, team2_win_prob)

        return {
            'team1': team1,
            'team2': team2,
            'team1_win_probability': team1_win_prob,
            'team2_win_probability': team2_win_prob,
            'favorite': favorite,
            'confidence': confidence,
            'team1_elo': team1_elo,
            'team2_elo': team2_elo,
            'elo_difference': abs(team1_elo - team2_elo)
        }

    def get_team_elo_features(self, team_name, environment='home'):
        """
        Get comprehensive ELO features for a team (similar to tennis)
        """
        self.initialize_team(team_name)

        # Get recent form (last 10 games)
        recent_games = self.elo_history[team_name][-10:] if self.elo_history[team_name] else []
        recent_wins = sum(1 for game in recent_games if game['result'] == 'W')
        recent_form = recent_wins / max(1, len(recent_games))

        # Calculate momentum (ELO change over last 10 games)
        if len(recent_games) >= 2:
            recent_momentum = recent_games[-1]['elo'] - recent_games[-10]['elo'] if len(recent_games) >= 10 else recent_games[-1]['elo'] - recent_games[0]['elo']
            recent_elo_change = recent_games[-1]['elo'] - recent_games[-5]['elo'] if len(recent_games) >= 5 else 0
        else:
            recent_momentum = 0
            recent_elo_change = 0

        return {
            'overall_elo': self.team_elo[team_name],
            'pitching_elo': self.pitching_elo[team_name],
            'hitting_elo': self.hitting_elo[team_name],
            'environment_elo': self.environment_elo[team_name].get(environment, self.default_elo),
            'home_elo': self.environment_elo[team_name]['home'],
            'away_elo': self.environment_elo[team_name]['away'],
            'recent_form': recent_form,
            'recent_momentum': recent_momentum,
            'recent_elo_change': recent_elo_change,
            'games_played': self.team_stats[team_name]['games_played'],
            'win_rate': self.team_stats[team_name]['win_rate'],
            'run_differential': self.team_stats[team_name]['run_differential']
        }

    def get_pitcher_elo_features(self, pitcher_name):
        """
        Get ELO features for a specific pitcher
        """
        if pitcher_name not in self.pitcher_elo:
            self.initialize_pitcher(pitcher_name)

        return self.pitcher_elo[pitcher_name].copy()

    def save_system(self, filepath):
        """Save the ELO system to file"""
        data = {
            'team_elo': self.team_elo,
            'pitching_elo': self.pitching_elo,
            'hitting_elo': self.hitting_elo,
            'environment_elo': self.environment_elo,
            'elo_history': self.elo_history,
            'team_stats': self.team_stats,
            'pitcher_elo': self.pitcher_elo
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, default=str, indent=2)

    def load_system(self, filepath):
        """Load the ELO system from file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            self.team_elo = data.get('team_elo', {})
            self.pitching_elo = data.get('pitching_elo', {})
            self.hitting_elo = data.get('hitting_elo', {})
            self.environment_elo = data.get('environment_elo', {})
            self.elo_history = data.get('elo_history', {})
            self.team_stats = data.get('team_stats', {})
            self.pitcher_elo = data.get('pitcher_elo', {})

            return True
        except Exception as e:
            print(f"Error loading ELO system: {e}")
            return False

def main():
    """Test the baseball ELO system"""
    print("⚾ BASEBALL ELO RATING SYSTEM")
    print("Adapted from 85% accuracy tennis model")
    print("=" * 50)

    elo_system = BaseballEloSystem()

    # Test with some sample games
    print("\n📊 Testing with sample games...")

    # Sample games data
    games = [
        ("New York Yankees", "Boston Red Sox", 8, 4, "home", "division", "Gerrit Cole", "Chris Sale"),
        ("Los Angeles Dodgers", "San Francisco Giants", 5, 3, "away", "division", "Walker Buehler", "Logan Webb"),
        ("Houston Astros", "Seattle Mariners", 7, 2, "home", "regular", "Justin Verlander", "Logan Gilbert"),
        ("Atlanta Braves", "New York Mets", 4, 6, "away", "division", "Spencer Strider", "Jacob deGrom")
    ]

    for team1, team2, score1, score2, env, game_type, pitcher1, pitcher2 in games:
        winner = team1 if score1 > score2 else team2
        loser = team2 if score1 > score2 else team1
        winner_score = max(score1, score2)
        loser_score = min(score1, score2)
        winner_pitcher = pitcher1 if score1 > score2 else pitcher2
        loser_pitcher = pitcher2 if score1 > score2 else pitcher1

        result = elo_system.update_elo_ratings(
            winner, loser, winner_score, loser_score,
            env, game_type, winner_pitcher, loser_pitcher
        )

        print(f"Game: {team1} {score1}-{score2} {team2}")
        print(f"  Winner ELO change: {result['winner_elo_change']:.1f}")
        print(f"  Expected win prob: {result['expected_winner_prob']:.1%}")

    # Test predictions
    print(f"\n🔮 Testing predictions...")

    prediction = elo_system.predict_game_outcome(
        "New York Yankees", "Boston Red Sox", "home", "Gerrit Cole", "Chris Sale"
    )

    print(f"Yankees vs Red Sox (Yankee Stadium):")
    print(f"  Yankees win probability: {prediction['team1_win_probability']:.1%}")
    print(f"  Red Sox win probability: {prediction['team2_win_probability']:.1%}")
    print(f"  Favorite: {prediction['favorite']}")
    print(f"  Confidence: {prediction['confidence']:.1%}")

    print(f"\n✅ Baseball ELO system ready!")

if __name__ == "__main__":
    main()