import spotipy
from spotipy.oauth2 import SpotifyOAuth
import configparser
from time import sleep

if __name__ == '__main__':

    config = configparser.ConfigParser()
    config.read('config.cfg')

    scope = "user-read-playback-state,user-modify-playback-state"
    sp = spotipy.Spotify(client_credentials_manager=SpotifyOAuth(
                         client_id=config['spotify']['client_id'],
                         client_secret=config['spotify']['client_secret'],
                         redirect_uri=config['spotify']['redirect_ui'], 
                         scope=scope,
                         username=config['spotify']['username']))

    me = sp.me()
    print(me)

    res = sp.devices()
    print(res)

    print("Starting playback!")
    sp.start_playback()

    sleep(5)
    print("Next track")
    sp.next_track()
    
    sleep(5)
    print("Prev track")
    sp.previous_track()

    sp.volume(100)
    sleep(2)
    print("Changing volume")
    sp.volume(50)
    sleep(2)
    sp.volume(100)

    sleep(5)
    print("Pause")
    sp.pause_playback()
