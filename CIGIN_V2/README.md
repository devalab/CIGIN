# Chemically Interpretable Graph Interaction Network 

## Installation:
* Create a new conda environment using:\
    `$ conda create -n cigin`
* Installing RDKit:\
    `$ conda install -c rdkit rdkit==2019.03.1`
* Installing other dependencies:\
    `$ conda install -c pytorch pytorch `\
    `$ pip install dgl` (Please check [here](https://docs.dgl.ai/en/0.4.x/install/) for 
     installing for different cuda builds)\
     `$ pip install numpy`\
     `$ pip install pandas`
     
## Dataset:

The dataset Solv@TUM used in this work can be downloaded from [here](https://mediatum.ub.tum.de/1452571?v=1). A sample of train and validation files required for this repository is provided [here](https://github.com/devalab/CIGIN/tree/master/CIGIN_V2/data). 
The MNSol dataset provided [here](https://conservancy.umn.edu/handle/11299/213300) can also be used to train the models.


## Running the Code:

To run the code following isntructions can be followed.
```python
python main.py --name cigin --interaction dot --batch_size 32
```
