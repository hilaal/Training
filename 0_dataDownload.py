import numpy as np
import os.path, random
import torch
from torch.utils.data import Dataset

print("Torch version {} ".format(torch.__version__))
# get training data
dir = "./"
if True:
    # download
    if not os.path.isfile('data-airfoils.npz'):
        import requests
        print("Downloading training data (300MB), this can take a few minutes the first time...")
        with open("data-airfoils.npz", 'wb') as datafile: 
            resp = requests.get('https://dataserv.ub.tum.de/s/m1615239/download?path=%2F&files=dfp-data-400.npz', verify=False)
            datafile.write(resp.content)
else:
    # alternative: load from google drive (upload there beforehand):
    from google.colab import drive
    drive.mount('/content/gdrive')
    dir = "./gdrive/My Drive/"

npfile=np.load(dir+'data-airfoils.npz')
print("Loaded data, {} training, {} validation samples".format(len(npfile["inputs"]),len(npfile["vinputs"])))
print("Size of the inputs array: "+format(npfile["inputs"].shape))

