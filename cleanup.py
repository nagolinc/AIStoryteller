
import os
import glob
import shutil


#delete all images in ./static/samples
files = glob.glob('./static/samples/*')
for f in files:
    #skip .gitkeep
    if f != './static/samples/.gitkeep':
        os.remove(f)