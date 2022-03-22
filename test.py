# Load samples from generator, taken while training
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import torch
import io

class CPU_Unpickler(pkl.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

''' Load samples from generator, taken while training '''
def view_samples(epoch, samples):
    fig, axes = plt.subplots(figsize=(16,4), nrows=2, ncols=8, sharey=True, sharex=True)
    
    for ax, img in zip(axes.flatten(), samples[epoch]):
        img = img.detach().cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = ((img + 1)*255 / (2)).astype(np.uint8)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        im = ax.imshow(img.reshape((32,32,3)))
        
    plt.savefig(f'output.png')
        

try:        
    with open('model/train_samples.pkl', 'rb') as f:
        samples = pkl.load(f)   
        _ = view_samples(-1, samples)

except: 
    with open('model/train_samples.pkl', 'rb') as f:
        samples = CPU_Unpickler(f).load()
        _ = view_samples(-1, samples)

        

