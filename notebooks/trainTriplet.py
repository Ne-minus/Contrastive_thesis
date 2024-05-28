import networkx as nx
from torch.utils.data import  DataLoader
import torch
from tqdm import tqdm
import yaml

import wandb

import torch
from transformers import get_linear_schedule_with_warmup, BertTokenizer
from torch.optim import AdamW
from setup import TripletDataset, SentenceBERT, TripletLoss


with open('./triplet_config.yml', 'r') as file:
    config = yaml.safe_load(file)
    print('Configs are read')


wandb.login()
wandb.init(project="contrastive_thesis_sbert_all_miniLM")

if __name__ == '__main__':
    num_epochs = config['NUM_EPOCHS'][0]

    model_name = config['MODEL_NAME'][0]
    tokenizer = BertTokenizer.from_pretrained(model_name)
    sbert_model = SentenceBERT(model_name)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    sbert_model.to(device)

    G = nx.read_edgelist(config['TRAIN_EDGES_PATH'][0], create_using=nx.DiGraph)
    dataset = TripletDataset(G, tokenizer)

    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size]) 

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=128,shuffle=False)

    optimizer = AdamW(sbert_model.parameters(), lr=2e-5, weight_decay=1e-4)
    num_training_steps = len(train_dataloader) * num_epochs
    num_warmup_steps = int(0.1 * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    triplet_loss = TripletLoss(margin=1.0)

    # training loop
    best_val_loss = float('inf')
    patience = 3
    epochs_no_improve = 0
    best_model_wts = ''

    for epoch in range(num_epochs):
        print(f'Epoch: {epoch + 1}')
        sbert_model.train()
        total_loss = 0
        for batch in tqdm(train_dataloader):
            batch_final = {k: v.to(device) for k, v in batch.items()}
            # print('All on device')
            optimizer.zero_grad()

            # Split the embeddings into anchor, positive, and negative embeddings
            anchor_embeddings = sbert_model(batch_final['anchor_input_ids'], batch_final['anchor_attention_mask'])
            positive_embeddings = sbert_model(batch_final['positive_input_ids'], batch_final['positive_attention_mask'])
            negative_embeddings = sbert_model(batch_final['negative_input_ids'], batch_final['negative_attention_mask'])

            loss = triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
            loss.backward()
            optimizer.step()
            scheduler.step()
            wandb.log({'Loss': loss})

            total_loss += loss.item()
        
        print(f'Loss: {total_loss / len(train_dataloader)}')
        wandb.log({'Average Training Loss': total_loss / len(train_dataloader)})
        
        sbert_model.eval()
        total_eval_loss = 0
        for batch in tqdm(val_dataloader):
            batch_final = {k: v.to(device) for k, v in batch.items()}

            with torch.no_grad():
                # all_embeddings = sbert_model(**batch_final)

                anchor_embeddings = sbert_model(batch_final['anchor_input_ids'], batch_final['anchor_attention_mask'])
                positive_embeddings = sbert_model(batch_final['positive_input_ids'], batch_final['positive_attention_mask'])
                negative_embeddings = sbert_model(batch_final['negative_input_ids'], batch_final['negative_attention_mask'])

                loss = triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings)

                total_eval_loss += loss.item()

        print(f'Validation Loss: {total_eval_loss / len(val_dataloader)}')
        wandb.log({'Average Validation Loss': total_eval_loss / len(val_dataloader)})
        
        avg_eval = total_eval_loss / len(val_dataloader)
        if avg_eval < best_val_loss:
            best_val_loss = avg_eval
            best_model_wts = sbert_model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print('Early stopping')
                break

    save_dir = config['OUTPUT_CHECKPOINTS_DIR'][0]
    torch.save({"model": best_model_wts.state_dict()}, f'{save_dir}/best_triplet.pth')