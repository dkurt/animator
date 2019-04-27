import os
import sys
import hashlib
from urllib2 import urlopen

# Inspired by https://github.com/opencv/opencv_extra/blob/master/testdata/dnn/download_models.py

MB = 1024*1024
BUFSIZE = 10*MB

def verify(filePath, targetSHA):
    sha = hashlib.sha1()
    with open(filePath, 'rb') as f:
        while True:
            buf = f.read(BUFSIZE)
            if not buf:
                break
            sha.update(buf)
    return targetSHA == sha.hexdigest()


def checkOrDownload(filePath, url, sha):
    if not os.path.exists(filePath) or not verify(filePath, sha):
        print("%s doesn't exist. Downloading..." % filePath)
        print("URL: " + url)

        baseDir = os.path.dirname(filePath)
        if not os.path.exists(baseDir):
            os.makedirs(baseDir)

        r = urlopen(url)
        with open(filePath, 'wb') as f:
            sys.stdout.write('  progress ')
            sys.stdout.flush()
            while True:
                buf = r.read(BUFSIZE)
                if not buf:
                    break
                f.write(buf)
                sys.stdout.write('>')
                sys.stdout.flush()

        if not verify(filePath, sha):
            print('Check sum failed!')
            exit(0)
