import torch
import torch.nn as nn
import torch.nn.functional as f


def print_one_batch(batch):
    tensors, pos_g, neg_g, blocks = batch
    print("tensors.keys:", tensors.keys())
    print("x.item:", tensors['item'].shape)
    print("x.user:", tensors['user'].shape)
    print("pos_g:", pos_g)
    print("neg_g:", neg_g)
    print("blocks:", blocks)


class Trainer:
    def __init__(self,
                 model,
                 optimizer,
                 criterion,
                 device,
                 print_every=1,
                 gpu_id=0,
                 use_ddp=False,
                 ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.print_every = print_every

        # only use for multiple GPUs
        self.gpu_id = gpu_id
        self.use_ddp = use_ddp

    def _forward(self, pos_g, neg_g, blocks):
        blocks = [b.to(self.device) for b in blocks]
        pos_g = pos_g.to(self.device)
        neg_g = neg_g.to(self.device)
        input_features = blocks[0].srcdata['features']

        self.optimizer.zero_grad()
        _, pos_score, neg_score = self.model(blocks, input_features, pos_g, neg_g)
        loss = self.criterion(pos_score, neg_score)
        return loss

    def train(self, edge_loader):
        self.model.train()
        total_loss = 0
        for i, (_, pos_g, neg_g, blocks) in enumerate(edge_loader):
            loss = self._forward(pos_g, neg_g, blocks)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            if (i + 1) % self.print_every == 0:
                print("Batch {}/{}: loss = {:.5f}".format(i + 1, len(edge_loader), loss.item()))

        avg_loss = total_loss / len(edge_loader)
        return avg_loss

    def calculate_loss(self, edge_loader):

        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for i, (_, pos_g, neg_g, blocks) in enumerate(edge_loader):
                loss = self._forward(pos_g, neg_g, blocks)
                total_loss += loss.item()
                if (i + 1) % self.print_every == 0:
                    print("Batch {}/{}: loss = {:.5f}".format(i + 1, len(edge_loader), loss.item()))

            avg_loss = total_loss / len(edge_loader)
        return avg_loss


def get_embeddings(model,
                   graph,
                   node_loader,
                   embed_dim: int,
                   print_every: int = 1
                   ):
    """
    Fetch the embeddings for all the nodes in the node_loader.

    Node Loader is preferable when computing embeddings because we can specify which nodes to compute the embedding for,
    and only have relevant nodes in the computational blocks. Whereas Edgeloader is preferable for training, because
    we generate negative edges also.
    """
    device = next(model.parameters()).device
    embed_dict = {node_type: torch.zeros(graph.num_nodes(node_type), embed_dim).to(device)
                  for node_type in graph.ntypes}

    for i, (input_nodes, output_nodes, blocks) in enumerate(node_loader):
        blocks = [b.to(device) for b in blocks]
        input_features = blocks[0].srcdata['features']
        h = model.get_repr(blocks, input_features)
        for node_type, embedding in h.items():
            embed_dict[node_type][output_nodes[node_type]] = embedding

        if (i + 1) % print_every == 0:
            print("Batch {}/{}".format(i, len(node_loader)))
    return embed_dict
