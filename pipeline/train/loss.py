class ContrastiveLoss:
    """
    Contrastive loss using cosine similarity.
    """
    def __init__(self, margin=1.0):
        """
        Initialize the contrastive loss with a margin.
        
        Args:
            margin (float): Margin for contrastive loss.
        """
        self.margin = margin
    
    def __call__(self, query, positive, negatives):
        """
        Compute contrastive loss.
        
        Args:
            query (torch.Tensor): Tensor of shape (D,), representing the query embedding.
            positive (torch.Tensor): Tensor of shape (D,), representing the positive embedding.
            negatives (torch.Tensor): Tensor of shape (N, D), where N is the number of negative samples and D is the embedding dimension.
        
        Returns:
            torch.Tensor: Contrastive loss value.
        """
        assert negatives.shape[0] >= 1, "At least one negative sample is required for contrastive loss."
        
        pos_dist = 1 - F.cosine_similarity(query.unsqueeze(0), positive.unsqueeze(0))
        neg_dist = torch.cat([1 - F.cosine_similarity(query.unsqueeze(0), neg.unsqueeze(0)).unsqueeze(0) for neg in negatives])
        
        loss = torch.mean(F.relu(pos_dist - neg_dist + self.margin))
        return loss

    # Contrastive Loss
    def contrastive_loss(query_emb, pos_emb, neg_emb, temperature=TEMPERATURE):
        pos_sim = F.cosine_similarity(query_emb, pos_emb)
        neg_sim = F.cosine_similarity(query_emb, neg_emb)
        
        loss = -torch.log(torch.exp(pos_sim / temperature) / 
                        (torch.exp(pos_sim / temperature) + torch.exp(neg_sim / temperature)))
        return loss.mean()