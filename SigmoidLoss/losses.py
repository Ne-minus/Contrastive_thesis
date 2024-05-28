import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Literal, Union, Tuple


class BaseContrastiveLoss(nn.Module):
    def __init__(
        self,
        project_dim: Union[int, None] = None,
        projector: Literal["Identity", "Linear", "MLP"] = "Identity",
    ):
        super().__init__()

        if projector == "Identity":
            self.net = nn.Identity()
        elif projector == "Linear":
            assert project_dim is not None
            self.net = nn.LazyLinear(project_dim)
        elif projector == "MLP":
            assert project_dim is not None
            self.net = nn.Sequential(
                nn.LazyLinear(project_dim),
                nn.ReLU(inplace=True),
                nn.LazyLinear(project_dim),
            )
        else:
            raise ValueError(
                f"Unkown projector: {projector}. "
                "Valid values are: {identiy, linear, MLP}"
            )

    def project(self, x):
        return self.net(x)


class ContrastiveLoss(BaseContrastiveLoss):

    def __init__(self, margin=1, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.margin = margin
        
    def sample_in_pairs(self, a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.tensor, torch.tensor]:

        negative_pairs = []
        positive_pairs = []

        for i in range(len(a)):
            for j in range(len(b)):
                if i == j:
                    positive_pairs.append([a[i], b[j]])
                else:
                    negative_pairs.append([a[i], b[j]])
                    
        return positive_pairs, negative_pairs 

    def to_format(self, embeddings: List) -> Tuple[torch.tensor, torch.tensor]:
        outputs_1 = embeddings[:(len(embeddings) // 2)]
        outputs_2 = embeddings[(len(embeddings) // 2):]
        
        positive_pairs, negative_pairs = self.sample_in_pairs(outputs_1, outputs_2)
        
        return positive_pairs, negative_pairs

    def forward(self, embeddings: torch.tensor) -> float:
        
        embeddings = self.project(embeddings) 
        positive, negative = self.to_format(embeddings)

        positive_tensor = torch.stack([torch.stack(row) for row in positive])
        negative_tensor = torch.stack([torch.stack(row) for row in negative])

        positive_loss = F.pairwise_distance(
            positive_tensor[:, 0], positive_tensor[:, 1]).pow(2)
        
        negative_loss = F.relu(
            self.margin
            - F.pairwise_distance(
                negative_tensor[:, 0],
                negative_tensor[:, 1],
            )
        ).pow(2)

        loss = torch.cat([positive_loss, negative_loss], dim=0)

        return loss.sum()  # / (len(positive_pairs) + len(negative_pairs))


class SigmoidLoss(nn.Module):
    def __init__(self,
                init_temperature=10.0,
                init_bias=-10.0):
        
        super(SigmoidLoss, self).__init__()

        temp_tensor = torch.tensor(init_temperature, dtype=torch.float32)
        bias_tensor = torch.tensor(init_bias, dtype=torch.float32)
        
        self.temperature = nn.Parameter(torch.log(temp_tensor))
        self.bias = nn.Parameter(bias_tensor)

    def get_loss(self, first_pair_output, second_pair_output):

        z_recon = F.normalize(first_pair_output)
        z_contrastive = F.normalize(second_pair_output)
         
        logits = (z_recon @ z_contrastive.T) * self.temperature + self.bias
        m1_diag1 = -torch.ones_like(logits) + 2 * torch.eye(logits.size(0)).to(
            logits.device
        )
        loglik = F.logsigmoid(m1_diag1 * logits)
        nll = -torch.sum(loglik, axis=-1)

        return nll.mean() 
    
    
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_similarity = F.pairwise_distance(anchor, positive, p=2)
        neg_similarity = F.pairwise_distance(anchor, negative, p=2)
        loss = torch.mean(F.relu(self.margin + pos_similarity - neg_similarity))
        return loss