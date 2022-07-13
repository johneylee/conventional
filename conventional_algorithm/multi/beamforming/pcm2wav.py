import os
from glob import glob
import sys

rootpath = '/home/leesunghyun/Downloads/LG_work/data/present_data/clean_kor_ai'
savepath = '/home/leesunghyun/Downloads/LG_work/data/present_data/clean_kor_ai_wav'

fid = open(savepath + os.sep + 'kor_ai_speech_full', 'w')

ksponspeech = sorted(glob(rootpath + os.sep + 'KsponSpeech_*'))
for datapath in ksponspeech:
    zs = datapath + os.sep + '*' + os.sep + '*.pcm'
    pcmlist = sorted(glob(datapath + os.sep + '*' + os.sep + '*.pcm'))

    for idx, pcmpath in enumerate(pcmlist):
        token = pcmpath.split(os.sep)
        filename = token[-1]
        foldername = token[-2]
        diskname = token[-3]

        savename = savepath + os.sep + diskname + os.sep + foldername + os.sep + filename.replace('.pcm', '.wav')

        if not os.path.exists(os.path.dirname(savename)):
            os.makedirs(os.path.dirname(savename))
        if os.path.exists(savename):
            continue
        cmd = 'ffmpeg -f s16le -ar 16k -ac 1 -i %s %s -loglevel quiet' % (pcmpath, savename)
        os.system(cmd)
        fid.write(savename + '\n')

        sys.stdout.write('\r[%d/%d] file' % (idx+1, len(pcmlist)))
        sys.stdout.flush()
fid.close()

# os.system('for i in *.pcm')
          # 'do name=`echo "$i" | cut -d '.' -f1`' \
          # 'echo "$name"' \
          # 'ffmpeg -f s16le -ar 16k -ac 1 -i "$i" "${name}.wav"')

# os.system('do name=`echo "$i" | cut -d '.' -f1`')
# os.system('echo "$name"')
# os.system('ffmpeg -f s16le -ar 16k -ac 1 -i "$i" "${name}.wav"')
# os.system('done')

