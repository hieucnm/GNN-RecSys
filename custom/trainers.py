import torch


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
