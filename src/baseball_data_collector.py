import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import json

class BaseballDataCollector:
    """
    Baseball Game Data Collector - Adapted from tennis data collector.

    Generates comprehensive baseball game data with realistic statistics
    for training the 85% accuracy prediction model.
    """

    def __init__(self):
        # MLB Teams (30 teams)
        self.teams = [
            # American League East
            "New York Yankees", "Boston Red Sox", "Toronto Blue Jays",
            "Tampa Bay Rays", "Baltimore Orioles",

            # American League Central
            "Houston Astros", "Minnesota Twins", "Chicago White Sox",
            "Cleveland Guardians", "Detroit Tigers",

            # American League West
            "Seattle Mariners", "Los Angeles Angels", "Texas Rangers",
            "Oakland Athletics", "Kansas City Royals",

            # National League East
            "Atlanta Braves", "New York Mets", "Philadelphia Phillies",
            "Miami Marlins", "Washington Nationals",

            # National League Central
            "Milwaukee Brewers", "St. Louis Cardinals", "Chicago Cubs",
            "Cincinnati Reds", "Pittsburgh Pirates",

            # National League West
            "Los Angeles Dodgers", "San Diego Padres", "San Francisco Giants",
            "Arizona Diamondbacks", "Colorado Rockies"
        ]

        # Top pitchers by team (simplified)
        self.team_pitchers = {
            "New York Yankees": ["Gerrit Cole", "Nestor Cortes", "Carlos Rodon", "Clay Holmes"],
            "Boston Red Sox": ["Chris Sale", "Brayan Bello", "Nick Pivetta", "Kutter Crawford"],
            "Los Angeles Dodgers": ["Walker Buehler", "Julio Urias", "Tony Gonsolin", "Evan Phillips"],
            "Houston Astros": ["Justin Verlander", "Framber Valdez", "Cristian Javier", "Ryan Pressly"],
            "Atlanta Braves": ["Spencer Strider", "Max Fried", "Charlie Morton", "A.J. Minter"],
            "San Francisco Giants": ["Logan Webb", "Alex Cobb", "Jakob Junis", "Camilo Doval"],
            "New York Mets": ["Jacob deGrom", "Max Scherzer", "Edwin Diaz", "David Robertson"],
            "Seattle Mariners": ["Logan Gilbert", "George Kirby", "Robbie Ray", "Paul Sewald"],
            "Philadelphia Phillies": ["Zack Wheeler", "Aaron Nola", "Ranger Suarez", "Jose Alvarado"],
            "Tampa Bay Rays": ["Shane McClanahan", "Tyler Glasnow", "Drew Rasmussen", "Pete Fairbanks"],
            "Toronto Blue Jays": ["Alek Manoah", "Kevin Gausman", "Jose Berrios", "Jordan Romano"],
            "Milwaukee Brewers": ["Corbin Burnes", "Brandon Woodruff", "Eric Lauer", "Josh Hader"],
            "St. Louis Cardinals": ["Jordan Montgomery", "Jack Flaherty", "Andre Pallante", "Ryan Helsley"],
            "San Diego Padres": ["Joe Musgrove", "Yu Darvish", "Blake Snell", "Josh Hader"],
        }

        # Fill in remaining teams with generic pitchers
        for team in self.teams:
            if team not in self.team_pitchers:
                self.team_pitchers[team] = [f"{team} Ace", f"{team} #2", f"{team} #3", f"{team} Closer"]

        # Game environments and types
        self.environments = ['home', 'away', 'neutral']
        self.game_types = ['regular', 'division', 'interleague', 'playoffs', 'world_series']

        # Team quality tiers (affects performance)
        self.team_tiers = {
            'elite': ["New York Yankees", "Los Angeles Dodgers", "Houston Astros", "Atlanta Braves"],
            'good': ["Boston Red Sox", "Seattle Mariners", "Philadelphia Phillies", "New York Mets",
                    "Toronto Blue Jays", "Milwaukee Brewers", "San Diego Padres", "Tampa Bay Rays"],
            'average': ["St. Louis Cardinals", "San Francisco Giants", "Minnesota Twins", "Chicago Cubs",
                       "Miami Marlins", "Cleveland Guardians", "Texas Rangers", "Arizona Diamondbacks"],
            'poor': ["Baltimore Orioles", "Detroit Tigers", "Chicago White Sox", "Oakland Athletics",
                    "Kansas City Royals", "Washington Nationals", "Cincinnati Reds", "Pittsburgh Pirates",
                    "Colorado Rockies", "Los Angeles Angels"]
        }

    def get_team_tier(self, team):
        """Get the performance tier of a team"""
        for tier, teams in self.team_tiers.items():
            if team in teams:
                return tier
        return 'average'

    def generate_realistic_score(self, team1, team2, environment, game_type):
        """
        Generate realistic baseball scores based on team strength and context
        """
        # Base scoring based on team tiers
        tier_scoring = {
            'elite': (5.2, 1.8),    # mean, std for runs scored
            'good': (4.8, 1.6),
            'average': (4.4, 1.5),
            'poor': (4.0, 1.4)
        }

        team1_tier = self.get_team_tier(team1)
        team2_tier = self.get_team_tier(team2)

        team1_mean, team1_std = tier_scoring[team1_tier]
        team2_mean, team2_std = tier_scoring[team2_tier]

        # Environment adjustments
        env_boost = {
            'home': 0.3,    # Home field advantage
            'away': -0.2,   # Away disadvantage
            'neutral': 0.0  # No adjustment
        }

        team1_mean += env_boost.get(environment, 0)

        # Game type adjustments (playoff games tend to be lower scoring)
        if game_type in ['playoffs', 'world_series']:
            team1_mean *= 0.9
            team2_mean *= 0.9

        # Generate scores (Poisson-like distribution)
        team1_score = max(0, int(np.random.normal(team1_mean, team1_std)))
        team2_score = max(0, int(np.random.normal(team2_mean, team2_std)))

        # Ensure we don't have ties (extra innings)
        if team1_score == team2_score:
            if random.random() < 0.5:
                team1_score += 1
            else:
                team2_score += 1

        return team1_score, team2_score

    def generate_pitcher_stats(self, pitcher_name, team_tier, game_outcome):
        """
        Generate realistic pitcher statistics based on performance and outcome
        """
        # Base pitcher performance by team tier
        tier_pitcher_stats = {
            'elite': {'era': 3.2, 'whip': 1.15, 'strikeouts_per_9': 9.5},
            'good': {'era': 3.8, 'whip': 1.25, 'strikeouts_per_9': 8.8},
            'average': {'era': 4.3, 'whip': 1.35, 'strikeouts_per_9': 8.2},
            'poor': {'era': 4.8, 'whip': 1.45, 'strikeouts_per_9': 7.5}
        }

        base_stats = tier_pitcher_stats[team_tier]

        # Adjust based on game outcome
        if game_outcome == 'win':
            era_multiplier = random.uniform(0.7, 1.1)
            k_multiplier = random.uniform(1.0, 1.4)
        else:
            era_multiplier = random.uniform(1.1, 1.8)
            k_multiplier = random.uniform(0.6, 1.1)

        return {
            'pitcher': pitcher_name,
            'innings_pitched': round(random.uniform(4.0, 8.0), 1),
            'hits_allowed': random.randint(3, 12),
            'runs_allowed': random.randint(1, 8),
            'earned_runs': random.randint(1, 7),
            'strikeouts': random.randint(3, 15),
            'walks': random.randint(0, 6),
            'era_game': round(base_stats['era'] * era_multiplier, 2),
            'pitch_count': random.randint(80, 120)
        }

    def generate_team_stats(self, team, score, opponent_score, environment):
        """
        Generate comprehensive team statistics for a game
        """
        # Hitting stats
        hits = max(score + random.randint(-2, 4), 1)
        doubles = random.randint(0, min(hits//3, 4))
        triples = random.randint(0, 1) if random.random() < 0.1 else 0
        home_runs = random.randint(0, min(score//2 + 1, 5))

        # Fielding stats
        errors = random.randint(0, 3) if random.random() < 0.25 else 0

        # Base running
        stolen_bases = random.randint(0, 3)
        caught_stealing = random.randint(0, 1)

        return {
            'team': team,
            'runs': score,
            'hits': hits,
            'doubles': doubles,
            'triples': triples,
            'home_runs': home_runs,
            'rbis': score + random.randint(-1, 2),
            'walks': random.randint(2, 8),
            'strikeouts': random.randint(6, 15),
            'stolen_bases': stolen_bases,
            'caught_stealing': caught_stealing,
            'errors': errors,
            'left_on_base': random.randint(4, 12),
            'batting_average': round(random.uniform(0.220, 0.320), 3),
            'on_base_percentage': round(random.uniform(0.280, 0.380), 3),
            'slugging_percentage': round(random.uniform(0.350, 0.550), 3)
        }

    def generate_game_data(self, team1, team2, game_date=None, environment='home', game_type='regular'):
        """
        Generate complete game data with all statistics
        """
        if game_date is None:
            game_date = datetime.now()

        # Generate scores
        team1_score, team2_score = self.generate_realistic_score(team1, team2, environment, game_type)

        # Determine winner
        winner = team1 if team1_score > team2_score else team2
        loser = team2 if team1_score > team2_score else team1

        # Get random pitchers
        team1_pitcher = random.choice(self.team_pitchers.get(team1, [f"{team1} Pitcher"]))
        team2_pitcher = random.choice(self.team_pitchers.get(team2, [f"{team2} Pitcher"]))

        # Generate pitcher stats
        team1_pitcher_stats = self.generate_pitcher_stats(
            team1_pitcher, self.get_team_tier(team1), 'win' if winner == team1 else 'loss'
        )
        team2_pitcher_stats = self.generate_pitcher_stats(
            team2_pitcher, self.get_team_tier(team2), 'win' if winner == team2 else 'loss'
        )

        # Generate team stats
        team1_stats = self.generate_team_stats(team1, team1_score, team2_score, environment)
        team2_stats = self.generate_team_stats(team2, team2_score, team1_score,
                                              'away' if environment == 'home' else 'home')

        # Create comprehensive game record
        game_data = {
            'game_id': f"{team1}_{team2}_{game_date.strftime('%Y%m%d')}",
            'date': game_date.isoformat(),
            'team1': team1,
            'team2': team2,
            'team1_score': team1_score,
            'team2_score': team2_score,
            'winner': winner,
            'loser': loser,
            'environment': environment,
            'game_type': game_type,
            'innings': 9,  # Standard game length

            # Pitching matchup
            'team1_pitcher': team1_pitcher,
            'team2_pitcher': team2_pitcher,
            'team1_pitcher_stats': team1_pitcher_stats,
            'team2_pitcher_stats': team2_pitcher_stats,

            # Team statistics
            'team1_stats': team1_stats,
            'team2_stats': team2_stats,

            # Game context
            'attendance': random.randint(15000, 50000),
            'game_duration_minutes': random.randint(150, 240),
            'weather_temp': random.randint(65, 95),
            'wind_speed': random.randint(0, 15),
            'wind_direction': random.choice(['N', 'S', 'E', 'W', 'NE', 'NW', 'SE', 'SW']),
        }

        return game_data

    def generate_season_data(self, num_games=2430, start_date=None, season_year=2024):
        """
        Generate a full season worth of baseball games (162 games per team)
        """
        if start_date is None:
            start_date = datetime(season_year, 4, 1)  # Season starts in April

        games = []

        print(f"🎾 Generating {num_games} baseball games for {season_year} season...")

        for i in range(num_games):
            # Random date during baseball season (April-October)
            random_days = random.randint(0, 180)  # 6-month season
            game_date = start_date + timedelta(days=random_days)

            # Select random teams
            team1, team2 = random.sample(self.teams, 2)

            # Determine environment (70% home games for team1, 25% away, 5% neutral)
            environment = random.choices(
                ['home', 'away', 'neutral'],
                weights=[70, 25, 5]
            )[0]

            # Determine game type (90% regular season, 8% division, 2% playoffs)
            game_type = random.choices(
                ['regular', 'division', 'interleague', 'playoffs'],
                weights=[85, 10, 4, 1]
            )[0]

            # Generate game
            game_data = self.generate_game_data(team1, team2, game_date, environment, game_type)
            games.append(game_data)

            # Progress indicator
            if (i + 1) % 500 == 0:
                print(f"   Generated {i + 1}/{num_games} games...")

        print(f"✅ Generated {len(games)} games successfully!")
        return games

    def save_games_to_csv(self, games, filepath):
        """
        Save games to CSV file for training
        """
        # Flatten the nested game data for CSV format
        flattened_games = []

        for game in games:
            flat_game = {
                'game_id': game['game_id'],
                'date': game['date'],
                'team1': game['team1'],
                'team2': game['team2'],
                'team1_score': game['team1_score'],
                'team2_score': game['team2_score'],
                'winner': game['winner'],
                'loser': game['loser'],
                'environment': game['environment'],
                'game_type': game['game_type'],
                'team1_pitcher': game['team1_pitcher'],
                'team2_pitcher': game['team2_pitcher'],

                # Team 1 stats
                'team1_hits': game['team1_stats']['hits'],
                'team1_home_runs': game['team1_stats']['home_runs'],
                'team1_errors': game['team1_stats']['errors'],
                'team1_batting_avg': game['team1_stats']['batting_average'],

                # Team 2 stats
                'team2_hits': game['team2_stats']['hits'],
                'team2_home_runs': game['team2_stats']['home_runs'],
                'team2_errors': game['team2_stats']['errors'],
                'team2_batting_avg': game['team2_stats']['batting_average'],

                # Pitcher stats
                'team1_pitcher_era': game['team1_pitcher_stats']['era_game'],
                'team1_pitcher_strikeouts': game['team1_pitcher_stats']['strikeouts'],
                'team2_pitcher_era': game['team2_pitcher_stats']['era_game'],
                'team2_pitcher_strikeouts': game['team2_pitcher_stats']['strikeouts'],

                # Game context
                'attendance': game['attendance'],
                'game_duration': game['game_duration_minutes'],
                'temperature': game['weather_temp'],
                'wind_speed': game['wind_speed']
            }

            flattened_games.append(flat_game)

        df = pd.DataFrame(flattened_games)
        df.to_csv(filepath, index=False)
        print(f"💾 Saved {len(games)} games to {filepath}")

    def save_games_to_json(self, games, filepath):
        """
        Save complete game data to JSON file
        """
        with open(filepath, 'w') as f:
            json.dump(games, f, indent=2, default=str)
        print(f"💾 Saved {len(games)} complete games to {filepath}")

def main():
    """Generate sample baseball data"""
    print("⚾ BASEBALL DATA COLLECTOR")
    print("Adapted from tennis data generator")
    print("=" * 50)

    collector = BaseballDataCollector()

    # Generate a small sample first
    print("\n📊 Generating sample games...")
    sample_games = collector.generate_season_data(num_games=100, season_year=2024)

    # Show a sample game
    print(f"\n🎾 Sample game:")
    sample = sample_games[0]
    print(f"   {sample['team1']} {sample['team1_score']}-{sample['team2_score']} {sample['team2']}")
    print(f"   Winner: {sample['winner']}")
    print(f"   Environment: {sample['environment']}")
    print(f"   Pitchers: {sample['team1_pitcher']} vs {sample['team2_pitcher']}")

    # Save sample data
    import os
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)

    collector.save_games_to_csv(sample_games, os.path.join(data_dir, 'sample_baseball_games.csv'))
    collector.save_games_to_json(sample_games, os.path.join(data_dir, 'sample_baseball_games.json'))

    print(f"\n✅ Baseball data collector ready!")
    print(f"Use generate_season_data() to create full training datasets!")

if __name__ == "__main__":
    main()