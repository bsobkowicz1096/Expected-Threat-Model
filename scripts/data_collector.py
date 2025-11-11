import os
import warnings
import pandas as pd
import statsbombpy.sb as sb

# Defining Constants
DEFAULT_SEASON = '2015/2016'
TOP5_LEAGUES = ['Italy', 'England', 'Spain', 'Germany', 'France']

# Simple path: one level up from scripts folder to project root, then data
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    
warnings.filterwarnings("ignore", category=UserWarning, module="statsbombpy")

def collect_events_data(league, save_path=DATA_DIR, season_name=DEFAULT_SEASON, save_files=True):
    """
    Retrieves and saves all event data from selected league for specified season.
    
    Parameters:
    -----------
    league : str or list
        League name or list of leagues (e.g. 'Italy', 'England', 'Spain', 'Germany', 'France')
    save_path: str, optional
        Target file save path (default: 'data' folder in current directory)
    season_name: str, optional
        Season name (default: '2015/2016')
    save_files: bool, optional
        Whether to save CSV files (default: True)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with all events
    """
    # Create directory if it doesn't exist
    if save_files and not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Created directory: {save_path}")
    
    # Handle single league or list of leagues
    if isinstance(league, str):
        leagues_to_process = [league]
    else:
        leagues_to_process = league
    
    all_events_data = []
    
    # Process each league separately
    for current_league in leagues_to_process:
        print(f"\nStarting data retrieval from league: {current_league}")
        
        # Retrieve league data
        try:
            free_comps = sb.competitions()
            
            # Filter selected league
            league_data = free_comps[(free_comps['season_name']==season_name) & 
                               (free_comps['country_name']==current_league)]
            
            if league_data.empty:
                print(f"No data found for league {current_league} in season {season_name}. Skipping.")
                continue
            
            competitions = list(league_data['competition_id'])
            
            # Retrieve match IDs
            season_id = league_data['season_id'].iloc[0]
            all_matches = pd.concat([sb.matches(competition_id=comp_id, season_id=season_id) 
                                  for comp_id in competitions])
            matches_id = list(all_matches['match_id'])
            print(f"Found {len(matches_id)} matches to analyze")
        except Exception as e:
            print(f"Error retrieving matches for league {current_league}: {str(e)}")
            continue
        
        # Retrieve event data
        event_data = []
        
        for idx, match_id in enumerate(matches_id):
            try:
                print(f"Processing match {idx+1}/{len(matches_id)}", end='\r')
                
                # Get all events for this match
                events = sb.events(match_id=match_id)
                
                # Add match_id to the events for tracking
                if not events.empty:
                    events['match_id'] = match_id
                    event_data.append(events)
                                
            except Exception as e:
                print(f"\nError with match {match_id}: {str(e)}")
                continue
        
        if event_data:
            # Combine data from this league
            print("\nCombining data...")
            league_events = pd.concat(event_data, ignore_index=True)
            
            # Basic data info
            print(f"Total events collected: {len(league_events)}")
            print(f"Event types found: {league_events['type'].nunique()}")
            print(f"Most common events:")
            print(league_events['type'].value_counts().head(10))
            
            # Save file for this league
            if save_files:
                # Create proper filename
                season_str = season_name.replace("/", "_")
                
                # Save full data as parquet (backup/research)
                full_filename = os.path.join(save_path, f'all_events_{current_league}_{season_str}.parquet')
                league_events.to_parquet(full_filename, index=False)
                print(f"Full data saved to: {full_filename}")
                
                # Save essential columns for xT model
                essential_cols = ['id', 'index', 'type', 'location', 'possession', 'possession_team', 
                 'period', 'minute', 'second', 'team', 'player', 'match_id', 'timestamp',
                 'pass_end_location', 'carry_end_location', 'shot_outcome']
                # Filter columns that actually exist
                available_cols = [col for col in essential_cols if col in league_events.columns]
                
                xt_filename = os.path.join(save_path, f'xt_events_{current_league}_{season_str}.parquet')
                league_events[available_cols].to_parquet(xt_filename, index=False)
                print(f"xT-focused data saved to: {xt_filename}")
            
            # Add to collective data
            all_events_data.append(league_events)
    
    # Combine data from all leagues if there's more than one
    if len(all_events_data) > 0:
        all_events = pd.concat(all_events_data, ignore_index=True)
        
        # Print summary statistics
        print(f"\n=== SUMMARY STATISTICS ===")
        print(f"Total events: {len(all_events)}")
        print(f"Total matches: {all_events['match_id'].nunique()}")
        print(f"Event types: {all_events['type'].nunique()}")
        print(f"\nTop 15 event types:")
        print(all_events['type'].value_counts().head(15))
        
        # Save collective file if more than one league was processed
        if save_files and len(leagues_to_process) > 1:
            season_str = season_name.replace("/", "_")
            
            # Full data backup
            full_filename = os.path.join(save_path, f'all_events_combined_{season_str}.parquet')
            all_events.to_parquet(full_filename, index=False)
            print(f"\nFull collective data saved to: {full_filename}")
            
            # xT-focused data
            essential_cols = ['id', 'index', 'type', 'location', 'possession', 'possession_team', 
                            'period', 'minute', 'second', 'team', 'player', 'match_id', 'timestamp']
            available_cols = [col for col in essential_cols if col in all_events.columns]
            
            xt_filename = os.path.join(save_path, f'xt_events_combined_{season_str}.parquet')
            all_events[available_cols].to_parquet(xt_filename, index=False)
            print(f"xT-focused collective data saved to: {xt_filename}")
        
        return all_events
    else:
        print("No data retrieved.")
        return None

if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created data directory: {DATA_DIR}")
    
    # Retrieve data for TOP5 leagues
    print(f"Starting data retrieval for leagues: {', '.join(TOP5_LEAGUES)}")
    events_df = collect_events_data(league=TOP5_LEAGUES, save_path=DATA_DIR)