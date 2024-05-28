import networkx as nx
from torch.utils.data import  DataLoader
import torch
from tqdm import tqdm
import yaml

import wandb

import torch
from transformers import AdamW, BertTokenizer, BertModel

from losses import  ContrastiveLoss
from set_contructor import TripletDataset, GraphDataCollator

with open('/raid/rabikov/contrastive_data/loader_config.yml', 'r') as file:
    config = yaml.safe_load(file)
    print('Configs are read')



def train():
    model.train()
    
    total_train_loss = 0

    for batch_idx, batch in enumerate(tqdm(train_dataloader, desc='Training')):
            
        batch_final = {k: v.to(device) for k, v in batch.items()}
        
        outputs_final = model(**batch_final)
        outputs_cls = outputs_final.last_hidden_state[:, 0, :] 
        
        outputs_1 = outputs_cls[:(len(outputs_cls) // 2)]
        outputs_2 = outputs_cls[(len(outputs_cls) // 2):]
        
        loss = criterion.get_loss(outputs_1, outputs_2)
        total_train_loss += loss
        
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
               
        if batch_idx % 200 == 0:
            print(f'Loss on {batch_idx} batch: {loss}')
            wandb.log({"loss": loss})

    avg_train_loss = total_train_loss/len(train_dataloader)
    print(f'Average trainig loss {avg_train_loss}')
    return avg_train_loss

def evaluate():
    model.eval()
    total_eval_loss = 0
    
    for batch_idx, batch in enumerate(tqdm(val_dataloader, desc="Evaluating")):
        batch_final = {k: v.to(device) for k, v in batch.items()}
        
        with torch.no_grad():
            outputs_final = model(**batch_final)
            outputs_cls = outputs_final.last_hidden_state[:, 0, :] 
            
            outputs_1 = outputs_cls[:(len(outputs_cls) // 2)]
            outputs_2 = outputs_cls[(len(outputs_cls) // 2):]
            
        loss = criterion.get_loss(outputs_1, outputs_2)
        print(loss)
        total_eval_loss += loss
        
#         if batch_idx > 20:
#             break
    avg_eval_loss = total_eval_loss / len(val_dataloader)
    print(f"Evaluation loss: {avg_eval_loss}")
    return avg_eval_loss






if __name__ == '__main__':

    wandb.login()
    wandb.init(project="contrastive_thesis_sigmoid_full")

    if config['MODEL'][0] == 'BERT':

        model = BertModel.from_pretrained('bert-base-uncased', num_labels=2)
        optimizer = AdamW(model.parameters(), lr=0.001)
        criterion = SigmoidLoss()
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.to(device)

    G=nx.read_edgelist(config['EDGE_LIST_PATH'][0], create_using=nx.DiGraph)
    dataset = TripletDataset(G, config['NEIGHBOUR_MATRIX_PATH'][0])

    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=32, collate_fn=GraphDataCollator(dataset.matrix, dataset.indices, tokenizer), shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=32, collate_fn=GraphDataCollator(dataset.matrix, dataset.indices), shuffle=True)

    epochs = config['NUM_EPOCHS'][0]

    best_val_loss = float('inf')
    patience = 3
    epochs_no_improve = 0
    best_model_wts = model.state_dict() 

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        avg_train = train()
        avg_eval = evaluate()
        wandb.log({"train_loss": avg_train, "eval_loss": avg_eval, "epoch": epoch})

        if avg_eval < best_val_loss:
            best_val_loss = avg_eval
            best_model_wts = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print('Early stopping')
                break

    dir_to_save = config['CHECKPOINTS_DIR'][0]

    torch.save({"model": best_model_wts}, f'{dir_to_save}/best_checkpoint.pth')
    
    tokenizer.save_pretrained(dir_to_save)