# Taxonomy_thesis  

This repository is dedicated to Contrastive Learning for Taxonomy Enrichment task.  
The stricture is following:  

```plaintext
Contrastive_thesis/
├── SigmoidLoss/              
│   ├── losses.py      
│   └── set_constructior.py          
│   └── train.py 
│   └── loader_config.yml  
│  
├── data/                    
│   ├── MAG_CS/            
│   └── WordNet/           
│  
├── notebooks/                
│   ├── Triplet_loss_pipeline.ipynb
│   └── setup.py  
│   └── tripletTrain.py
│   └── triplet_config.yml
```

## Data
All the data used for training, evaluation and test is available in the ```data``` folder. Checkpoints we obtained are available at the same directory.

## Usage  
In order to run our architecture, you need to specify appropriate paths in ```.yml``` file and run ```train.py```.  
```bash
python3 train.py
```
