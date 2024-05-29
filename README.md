# Taxonomy_thesis  

This repository is dedicated to Contrastive Learning for Taxonomy Enrichment task.  
The structure is following:  

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
├── TripletLoss/                
│   ├── Triplet_loss_pipeline.ipynb
│   └── setup.py  
│   └── trainTriplet.py
│   └── triplet_config.yml
├── requirements.txt
```

## Data
All the data used for training, evaluation and test is available in the ```data``` folder. 

## Usage  
In order to run our architecture, you need to specify appropriate paths in ```.yml``` file and run ```train.py```.  
```bash
pip install -r requirements. txt
```
```bash
python3 train.py
```

Evaluation script is available in ```Triplet_loss_pipeline.ipynb```.  

