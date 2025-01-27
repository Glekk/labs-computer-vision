# labs-computer-vision

## This repository contains the laboratory works for the Computer Vision course.
All the labs contain notebooks folder with the code and the results of the labs, src folder with the rearranged code from the notebooks and .env file with the environment variables (there is .env.example files in each folder).

There are 4 labs in total written in Python using PyTorch:
- CNN BLocks lab - contains the experiments with different CNN blocks like Sqeeze and Excitation, Residual block, etc. (it uses the following dataset: [kaggle](https://www.kaggle.com/datasets/tongpython/cat-and-dog))
- DCGAN lab - contains the implementation of the DCGAN model. (it uses the following dataset: [kaggle](https://www.kaggle.com/datasets/splcher/animefacedataset))
- Segmentation lab - contains the implementation of the DeepLabV3 model for semantic segmentation. (it uses the following dataset: [kaggle](https://www.kaggle.com/datasets/faizalkarim/flood-area-segmentation))
- ViT lab - contains the implementation of the Vision Transformer model. (it uses the following dataset: [kaggle](https://www.kaggle.com/datasets/tongpython/cat-and-dog))

## Installation
To download the repository use the following command:
```
git clone 
```

All required libraries are listed in the requirements.txt file in each lab folder. You need to change the directory to the lab folder and run the following command:
```
pip install -r requirements.txt
```

## Usage
You can run the notebooks in the notebooks folder using VS Code or Jupyter Notebook.  
You can use functions from the src folder in your code.