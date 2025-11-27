import torch


def supervised_contrastive_loss(
    embeddings: torch.Tensor, labels: torch.Tensor, temperature: float = 0.01
):
    """
    embeddings: [N, D]
    labels: [N]
    """
    device = embeddings.device
    labels = labels.view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)

    # compute cosine similarity
    sim = torch.mm(embeddings, embeddings.T) / temperature

    # for numerical stability
    sim_max = torch.max(sim, dim=-1, keepdim=True).values
    sim = sim - sim_max.detach()

    # mask out self-similarity
    mask.fill_diagonal_(0)

    # positive pairs numerator
    exp_sim = torch.exp(sim)
    numerator = exp_sim * mask

    # denominator = all except self
    denominator = exp_sim.sum(dim=1, keepdim=True)

    loss = -torch.log((numerator.sum(dim=1) + 1e-9) / (denominator + 1e-9))
    return loss.mean()
