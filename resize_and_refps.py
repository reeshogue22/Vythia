import subprocess
import glob
import os
import time
def main():
    videos = glob.glob("../videos/*.mp4")
    
    if not os.path.exists('../data'):
        os.mkdir('../data')

    for i, video in enumerate(videos):
        identity = video[-14:-4]
        print(identity)
        try:
            os.mkdir('../data/{}'.format(identity))
        except FileExistsError as e:
            print("Not doing this as the files supposedly already exist.")
            continue
        subprocess.call(["ffmpeg", "-i", video, '-vf', 
        'scale=256:256,fps=8', "-threads", "40", "-q:v", "3", '../data/{}/%07d.jpg'.format(identity)])
        print("File {0} out of {1} files extracted. Was file {2}.".format(i, len(videos), video))

main()


#':force_original_aspect_ratio=decrease,pad=288:288:-1:-1:color=black,setsar=1:1'
