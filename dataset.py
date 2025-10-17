from helper import convert_unix_timestamp_to_date, replace_missed_bans, riot_request, save_json_to_dir, shuffle_picks_order_with_weights
from tqdm import tqdm
import logging

# Logger config
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_scrapping.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

################# CONSTANTS #################
API_KEY = "RGAPI-4c99730c-230c-4998-862e-22a1a49591e7" # Need to change it every 24 hours
HEADERS = {"X-Riot-Token" : API_KEY}
SAVE_AFTER_ITERATION = 1000

class Dataset():
    """
    Enables creation of dataset according of the region wanted 
    and other params
    """
    
    def __init__(self, region, queue, game_count, player_count, elo) -> None:
        self.region = region
        self.queue = queue
        self.game_count = game_count
        self.player_count = player_count
        self.headers = HEADERS
        self.elo = elo
        
        # API's urls
        self.player_list_url = ("/lol/league/v4/{elo}leagues/by-queue/").format(elo = self.elo)
        self.matches_url = "/lol/match/v5/matches/by-puuid/"
        self.match_data_url = "/lol/match/v5/matches/"
        
        # Hosts
        match region.upper():
            case "EUROPE":
                self.host = "euw1.api.riotgames.com"
                self.match_host = f"europe.api.riotgames.com"
            case "AMERICA":
                self.host = "na1.api.riotgames.com"
                self.match_host = f"americas.api.riotgames.com"
            case "KOREA":
                self.host = "kr.api.riotgames.com"
                self.match_host = f"asia.api.riotgames.com"
            case _:
                self.host = "euw1.api.riotgames.com"
                self.match_host = f"europe.api.riotgames.com"
                
        # Caching challenger player list to avoid requesting too many times
        self._challenger_players = None
        
        logger.info(f"Dataset created for {self.region} - Queue {self.queue}")
        logger.info(f"Host: {self.host} | Match host: {self.match_host}")
    
    def get_challenger_player_list(self)-> list:
        """Get challenger players on the server"""
        if self._challenger_players is not None:
            return self._challenger_players
        
        request_url = f"https://{self.host}{self.player_list_url}{self.queue}"
        logger.info(f"Getting the {self.elo} player list...")
        data = riot_request(url=request_url, headers=self.headers)
        if not data:
            logger.error("Failed to get the challenger player list")
            return []
        
        entries = data["entries"]
        summoners_puuid = [e["puuid"] for e in entries]
        
        logger.info(f"Number of {self.elo} players found : {len(summoners_puuid)}")
        
        self._challenger_players = summoners_puuid[:self.player_count]
        return self._challenger_players
    
    def get_match_ids(self):
        """Get unique match ids for every player"""
        match_ids = set()
        players = self.get_challenger_player_list()
        
        logger.info(f"Getting matches for {len(players)} players...")
        
        for player_puuid in tqdm(players, desc="Player count"):
            request_url = f"https://{self.match_host}{self.matches_url}{player_puuid}/ids?count={self.game_count}"
            data = riot_request(url=request_url, headers=self.headers)
            
            if data and isinstance(data, list):
                match_ids.update(data)
            else:
                logger.warning(f"No match found for player {player_puuid[:8]}...")
        
        logger.info(f"Total of unique matches found: {len(match_ids)} without filtering queue")
        return match_ids
    
    def get_winner(self, match_data):
        teams = match_data.get("info", {}).get("teams", [])
        for team in teams:
            if team.get("win"):
                return team.get("teamId")
        return None
    
    def extract_match_data(self):
        """Extract detailed data from every match"""
        game_data = []
        match_ids = self.get_match_ids()
        game_missing_ban_count = 0
        
        logger.info(f"Extracting data for {len(match_ids)} matches...")
        idx = 0
        for game_id in tqdm(list(match_ids), desc="Game count per player"):
            idx += 1
            request_url = f"https://{self.match_host}{self.match_data_url}{game_id}"
            data = riot_request(url=request_url, headers=self.headers)
            
            if not data:
                idx -= 1
                continue
            
            try:
                game_info = data.get("info", {})
                teams = game_info.get("teams", [])
                participants = game_info.get("participants", [])
                
                # Ensure that only soloqueues are taken into account
                if game_info.get("queueId") != 420:
                    idx -= 1
                    continue
                
                picks = []
                bans = []
                missing_ban = False
                
                # Extracting bans
                for team in teams:
                    team_id = "blue" if team.get("teamId") == 100 else "red"
                    if team_id:
                        bans.extend([{"side" : team_id, "championId" : champ["championId"], "order" : order + 1} for order, champ in enumerate(team.get("bans", []))])
                
                    for b in bans:
                        if b["championId"] == -1 :
                            missing_ban = True
                    replace_missed_bans(bans)
                                
                if missing_ban :
                    game_missing_ban_count +=1            
                
                # Extracting picks
                for order,p in enumerate(participants):
                    team_id = "blue" if p.get("teamId") == 100 else "red"
                    position = p.get("teamPosition") if p.get("teamPosition") != "UTILITY" else "SUPPORT"
                    champ = p.get("championId")
                    
                    picks.append({"side" : team_id, "championId" : champ,"position" : position, "order" : order + 1 if team_id == 'blue' else order - 4})
                    
                
                game_data.append({
                    "match_id": game_id,
                    "game_version": game_info.get("gameVersion", "")[:5],
                    "game_duration": round(game_info.get("gameDuration", 0) / 60,2),
                    "game_creation": convert_unix_timestamp_to_date(game_info.get("gameCreation", 0)/1000),
                    "picks" : shuffle_picks_order_with_weights(picks),
                    "bans" : bans,
                    "blue_side_win" : self.get_winner(data) == 100,
                    "missing_ban" : missing_ban
                })
                
                if idx % SAVE_AFTER_ITERATION == 0 and idx !=0:
                    save_json_to_dir(game_data, "Datasets", self.region, idx)
                
            except Exception as e:
                logger.error(f"Error when dealing with match {game_id}: {e}")
                continue
        
        save_json_to_dir(game_data, "Datasets", self.region, len(game_data))
        
        logger.info(f"Extraction finished: {len(game_data)} matches successfully analyzed")
        logger.info(f"Games with at least one missing ban: {game_missing_ban_count}, thus modified so it can be used")
        return game_data
        
if __name__ == "__main__":          
    try:
        # Creating dataset
        europe__chall_dataset = Dataset(
            region="EUROPE",
            queue="RANKED_SOLO_5x5",
            game_count=1,
            player_count=1,
            elo="challenger"
        )

        # Extracting data
        european_chall_matches = europe__chall_dataset.extract_match_data()
        
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        raise