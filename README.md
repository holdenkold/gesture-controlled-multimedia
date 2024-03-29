# gesture-controlled-multimedia

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/holdenkold/gesture-controlled-multimedia/blob/master/modelTraining.ipynb)

Gesture Controlled Multimedia is an app allowing to control Spotify by hand gestures.

### Gestures available

- 0 other gesture

- 1 ✋ play / pause

- 2 🤏 mute

- 3 👆previous track

- 4 👌 next track

- 5 ✌️ volume up

- 6 ☝️ volume down

## Dataset 

### Download link (public)

- https://pages.mini.pw.edu.pl/~gorzynskik/obrazki/

### Contribute with synced directory (MiNI account required)

- Generate rsa keypair (https://serverfault.com/a/330740)

    Enter the following command to start generating a rsa keypair:

        # ssh-keygen

    When the message 'Enter file in which to save the key' appears, just leave the filename blank by pressing Enter.

    When the terminal asks you to enter a passphrase, just leave this blank too and press Enter.

    Then copy the keypair onto the server with one simple command:

        # ssh-copy-id userid@ssh.mini.pw.edu.pl

    you can now log in without a password:

        # ssh userid@ssh.mini.pw.edu.pl

- Set SSH user (https://serverfault.com/a/680274)

    Add to your .ssh/config

        Host ssh.mini.pw.edu.pl
            User userid

    You can now log in automatically without typing username:

        ssh ssh.mini.pw.edu.pl


- Now you can sync your dataset folder with others 

        rsync -arO --perms --chmod=a+r dataset/ ssh.mini.pw.edu.pl:/home/samba/gorzynskik/public_html/obrazki/dataset

        (ask for write access)
        rsync -arO --perms --chmod=a+r ssh.mini.pw.edu.pl:/home/samba/gorzynskik/public_html/obrazki/dataset dataset/
