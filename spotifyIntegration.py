import spotipy
from spotipy.oauth2 import SpotifyOAuth
import configparser
from time import sleep
import logging
import pprint

class SpotifyClient():
    def __init__(self, logging_level=logging.WARNING):
        logging.basicConfig(format='%(asctime)s - %(message)s', level=logging_level)
        config = configparser.ConfigParser()
        config.read('config.cfg')

        scope = "user-read-playback-state,user-modify-playback-state"
        self.sp = spotipy.Spotify(client_credentials_manager=SpotifyOAuth(
                            client_id=config['spotify']['client_id'],
                            client_secret=config['spotify']['client_secret'],
                            redirect_uri=config['spotify']['redirect_ui'], 
                            scope=scope,
                            username=config['spotify']['username']))

        self.vol = 0
        try:
            cpl = self.sp.current_playback()
            self.vol = cpl['device']['volume_percent']
        except:
            print("Cannot get volume")

    def me(self):
        return self.sp.me()

    def playpause(self):
        st = self.status()
        if st is None:
            raise ConnectionError("Not connected")
        if st['is_playing']:
            self.pause()
        else:
            self.play()

    def play(self):
        logging.info("Play")
        self.sp.start_playback()

    def pause(self):
        logging.info("Pause")
        self.sp.pause_playback()

    def prev(self):
        logging.info("Prev track")
        self.sp.previous_track()

    def next(self):
        logging.info("Next track")
        self.sp.next_track()

    def devices(self):
        return self.devices()

    def status(self):
        return self.sp.currently_playing()

    def status2(self):
        return self.sp.current_playback()

    def getvolume(self):
        try:
            cpl = self.status2()
            return cpl['device']['volume_percent']
        except:
            return 100
    
    def mute(self):
        '''Switches mute/unmute'''
        vol = self.getvolume()
        if vol > 0:
            logging.info("Mute")
            self.vol = vol
            self.sp.volume(0)
        else:
            logging.info("Unmute")
            self.sp.volume(self.vol)

def test(spclient):
    me = spclient.me()
    pp = pprint.PrettyPrinter(depth=2)
    pp.pprint(me)

    st = spclient.status()
    st2 = spclient.status2()
    pp.pprint(st2)

    spclient.playpause()

    sleep(5)
    spclient.next()
    
    sleep(5)
    spclient.prev()

    sleep(5)
    spclient.mute()

    sleep(5)
    spclient.mute()

if __name__ == '__main__':
    spclient = SpotifyClient(logging_level=logging.INFO)
    test(spclient)
