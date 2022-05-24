# Face-Generation-Using-DCGAN
Training a DCGAN on a dataset of faces to get the generator network to generate new images that look like realistic faces. 
The model was trained on a subset of the [celebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset. 

## Architecture 
![plot](/assets/img.png)

## Usage 

### Train : 
``` pyhton3 train.py ``` 

### Test
``` pyhton3 test.py ``` 

## Visualising Loss
Plot of the training losses for the generator and discriminator:


![plot](/assets/loss.png)

## Results
![plot](/assets/output.png)
