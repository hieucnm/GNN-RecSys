from datetime import timedelta
import time

import dgl
import torch

from src.metrics import get_metrics_at_k
from src.utils import save_txt


def print_one_batch(batch):
    tensors, pos_g, neg_g, blocks = batch
    # print("tensors.keys:", tensors.keys())
    # print("x.item:", tensors['item'].shape)
    # print("x.user:", tensors['user'].shape)
    print("pos_g:", pos_g)
    # print("neg_g:", neg_g)
    # print("blocks:", blocks)


def train_model(model,
                num_epochs,
                num_batches_train,
                num_batches_val_loss,
                edgeloader_train,
                edgeloader_valid,
                loss_fn,
                delta,
                save_dir,
                neg_sample_size,
                use_recency=False,
                device=None,
                optimizer=torch.optim.Adam,
                lr=0.001,
                get_metrics=False,
                train_graph=None,
                valid_graph=None,
                nodeloader_valid=None,
                nodeloader_subtrain=None,
                k=None,
                out_dim=None,
                bought_eids=None,
                ground_truth_subtrain=None,
                ground_truth_valid=None,
                remove_already_bought=True,
                result_filepath=None,
                start_epoch=0,
                patience=5,
                pred=None,
                remove_false_negative=False,
                gpu_id=0,
                use_ddp=False
                ):
    """
    Main function to train a GNN, using max margin loss on positive and negative examples.

    Process:
        - A full training epoch
            - Batch by batch. 1 batch is composed of multiple computational blocks, required to compute embeddings
              for all the nodes related to the edges in the batch.
            - Input the initial features. Compute the embeddings & the positive and negative scores
            - Also compute other considerations for the loss function: negative mask, recency scores
            - Loss is returned, then backward, then step.
            - Metrics are computed on the subtraining set (using nodeloader)
        - Validation set
            - Loss is computed (in model.eval() mode) for validation edge for early stopping purposes
            - Also, metrics are computed on the validation set (using nodeloader)
        - Logging & early stopping
            - Everything is logged, best metrics are saved.
            - Using the patience parameter, early stopping is applied when val_loss stops going down.
    """
    cuda = device is not None and device.type != 'cpu'
    model.train_loss_list = []
    model.train_precision_list = []
    model.train_recall_list = []
    model.train_coverage_list = []
    model.train_auc_list = []
    model.val_loss_list = []
    model.val_precision_list = []
    model.val_recall_list = []
    model.val_coverage_list = []
    model.val_auc_list = []
    best_metrics = {}  # For visualization
    max_metric = -0.1
    patience_counter = 0  # For early stopping
    min_loss = 1.1

    opt = optimizer(model.parameters(), lr=lr)

    # TRAINING
    for epoch in range(start_epoch, num_epochs):

        if use_ddp:
            edgeloader_train.set_epoch(epoch)  # <--- necessary for data_loader with DDP.

        start_time = time.time()
        model.train()  # Because if not, after eval, dropout would be still be inactive
        total_loss = 0
        print(f'--> Epoch {epoch}/{num_epochs} : Training ...')
        for i, (_, pos_g, neg_g, blocks) in enumerate(edgeloader_train):
            # print out what inside a batch to debug, remember to break
            # print_one_batch((_, pos_g, neg_g, blocks))
            # break

            opt.zero_grad()

            # Negative mask
            negative_mask = {}
            if remove_false_negative:
                nids = neg_g.ndata[dgl.NID]
                for etype in pos_g.canonical_etypes:
                    neg_src, neg_dst = neg_g.edges(etype=etype)
                    neg_src = nids[etype[0]][neg_src]
                    neg_dst = nids[etype[2]][neg_dst]
                    negative_mask_tensor = valid_graph.has_edges_between(neg_src, neg_dst, etype=etype)
                    negative_mask[etype] = negative_mask_tensor.type(torch.float).to(device)
            if cuda:
                blocks = [b.to(device) for b in blocks]
                pos_g = pos_g.to(device)
                neg_g = neg_g.to(device)

            input_features = blocks[0].srcdata['features']
            # recency (TO BE CLEANED)
            recency_scores = None
            if use_recency:
                recency_scores = pos_g.edata['recency']

            _, pos_score, neg_score = model(blocks, input_features, pos_g, neg_g)

            # print("pos_g.clicks      :", pos_g.num_edges(('user', 'clicks', 'item')))
            # print("pos_score.clicks  :", pos_score[('user', 'clicks', 'item')].shape)
            # print("pos_g.converts    :", pos_g.num_edges(('user', 'converts', 'item')))
            # print("pos_score.converts:", pos_score[('user', 'converts', 'item')].shape)
            # print('-------------------------------------------------------------------')

            loss = loss_fn(pos_score,
                           neg_score,
                           recency_scores=recency_scores,
                           negative_mask=negative_mask
                           )

            if (i + 1) % 10 == 0:
                print("Epoch {}/{} - Batch {}/{}: loss = {:.5f}".format(
                    epoch, num_epochs, i + 1, len(edgeloader_train), loss.item()))

            if epoch > 0:  # For the epoch 0, no training (just report loss)
                loss.backward()
                opt.step()
            total_loss += loss.item()

            if epoch == 0 and i > 10:
                break  # For the epoch 0, report loss on only subset

        # only for debug ==
        #     break
        # continue
        # =================

        train_avg_loss = total_loss / (i + 1)
        model.train_loss_list.append(train_avg_loss)

        if gpu_id > 0:
            continue
        print(f'--> Epoch {epoch}/{num_epochs} : validating ...')
        model.eval()
        with torch.no_grad():
            total_loss = 0
            for i, (_, pos_g, neg_g, blocks) in enumerate(edgeloader_valid):

                # Negative mask
                negative_mask = {}
                if remove_false_negative:
                    nids = neg_g.ndata[dgl.NID]
                    for etype in pos_g.canonical_etypes:
                        neg_src, neg_dst = neg_g.edges(etype=etype)
                        neg_src = nids[etype[0]][neg_src]
                        neg_dst = nids[etype[2]][neg_dst]
                        negative_mask_tensor = valid_graph.has_edges_between(neg_src, neg_dst, etype=etype)
                        negative_mask[etype] = negative_mask_tensor.type(torch.float)
                        if cuda:
                            negative_mask[etype] = negative_mask[etype].to(device)

                if cuda:
                    blocks = [b.to(device) for b in blocks]
                    pos_g = pos_g.to(device)
                    neg_g = neg_g.to(device)

                input_features = blocks[0].srcdata['features']
                _, pos_score, neg_score = model(blocks,
                                                input_features,
                                                pos_g,
                                                neg_g,
                                                )
                # recency (TO BE CLEANED)
                recency_scores = None
                if use_recency:
                    recency_scores = pos_g.edata['recency']

                val_loss = loss_fn(pos_score,
                                   neg_score,
                                   recency_scores=recency_scores,
                                   negative_mask=negative_mask
                                   )
                total_loss += val_loss.item()
                if (i + 1) % 10 == 0:
                    print("Epoch {}/{} - Batch {}/{}: loss = {:.5f}".format(
                        epoch, num_epochs, i + 1, len(edgeloader_valid), val_loss.item()))
            val_avg_loss = total_loss / (i + 1)
            model.val_loss_list.append(val_avg_loss)

        ############
        # METRICS PER EPOCH 
        if get_metrics and epoch > 0:
            model.eval()
            with torch.no_grad():
                print(f'--> Epoch {epoch}/{num_epochs} : calculating training metrics ...')
                y = get_embeddings(train_graph, out_dim, model, nodeloader_subtrain, device)

                train_precision, train_recall, train_coverage, train_auc = get_metrics_at_k(
                    y, train_graph, model, out_dim, ground_truth_subtrain, bought_eids, k, True, device, pred)

                # validation metrics
                print(f'--> Epoch {epoch}/{num_epochs} : calculating validation metrics ...')
                y = get_embeddings(valid_graph, out_dim, model, nodeloader_valid, device)

                val_precision, val_recall, val_coverage, val_auc = get_metrics_at_k(
                    y, valid_graph, model, out_dim, ground_truth_valid, bought_eids, k, remove_already_bought,
                    device, pred
                )
                sentence = '''--> Finish epoch {:02d}/{:02d} 
                || Training Loss {:.5f} | Precision {:.3f}% | Recall {:.3f}% | Coverage {:.2f} | AUC {:.2f}% 
                || Validation Loss {:.5f} | Precision {:.3f}% | Recall {:.3f}% | Coverage {:.2f}% | AUC {:.2f}% '''\
                    .format(
                    epoch, num_epochs,
                    train_avg_loss, train_precision * 100, train_recall * 100, train_coverage * 100, train_auc * 100,
                    val_avg_loss, val_precision * 100, val_recall * 100, val_coverage * 100, val_auc * 100
                )
                print(sentence)
                save_txt(sentence, result_filepath, mode='a')

                model.train_precision_list.append(train_precision * 100)
                model.train_recall_list.append(train_recall * 100)
                model.train_coverage_list.append(train_coverage * 10)
                model.train_auc_list.append(train_auc * 100)
                model.val_precision_list.append(val_precision * 100)
                model.val_recall_list.append(val_recall * 100)
                model.val_coverage_list.append(val_coverage * 10)  # just *10 for viz purposes
                model.val_auc_list.append(val_auc * 100)  # just *10 for viz purposes

                # Visualization of best metric
                if val_recall > max_metric:
                    max_metric = val_recall
                    best_metrics = {'recall': val_recall, 'precision': val_precision,
                                    'coverage': val_coverage, 'auc': val_auc}

        else:
            sentence = "--> Finish epoch {:02d}/{:02d} | Training Loss {:.5f} | Validation Loss {:.5f} | ".format(
                epoch, num_epochs, train_avg_loss, val_avg_loss)
            print(sentence)
            save_txt(sentence, result_filepath, mode='a')

        if val_avg_loss < min_loss:
            min_loss = val_avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter == patience:
            break

        elapsed = time.time() - start_time
        result_to_save = f'--> Epoch took {timedelta(seconds=elapsed)} \n'
        print(result_to_save)
        save_txt(result_to_save, result_filepath, mode='a')

        # save model every epoch after first epoch
        if epoch > 0:
            torch.save(model.state_dict(), f'{save_dir}/model_ep_{epoch}.pth')
            print("Model saved!")

    viz = {'train_loss_list': model.train_loss_list,
           'train_precision_list': model.train_precision_list,
           'train_recall_list': model.train_recall_list,
           'train_coverage_list': model.train_coverage_list,
           'train_auc_list': model.train_auc_list,
           'val_loss_list': model.val_loss_list,
           'val_precision_list': model.val_precision_list,
           'val_recall_list': model.val_recall_list,
           'val_coverage_list': model.val_coverage_list,
           'val_auc_list': model.val_auc_list}

    print('Training completed.')
    return model, viz, best_metrics  # model will already be to 'cuda' device?


def get_embeddings(g,
                   out_dim: int,
                   trained_model,
                   node_loader,
                   device=None):
    """
    Fetch the embeddings for all the nodes in the node_loader.

    Node Loader is preferable when computing embeddings because we can specify which nodes to compute the embedding for,
    and only have relevant nodes in the computational blocks. Whereas Edgeloader is preferable for training, because
    we generate negative edges also.
    """
    cuda = device is not None and device.type != 'cpu'
    y = {ntype: torch.zeros(g.num_nodes(ntype), out_dim)
         for ntype in g.ntypes}

    # not sure if I need to put the 'result' tensor to device
    if cuda:
        trained_model = trained_model.to(device)
        y = {ntype: torch.zeros(g.num_nodes(ntype), out_dim).to(device)
             for ntype in g.ntypes}

    for i2, (input_nodes, output_nodes, blocks) in enumerate(node_loader):
        if i2 % 10 == 0:
            print("Computing embeddings: batch {}/{}".format(i2, len(node_loader)))
        if cuda:
            blocks = [b.to(device) for b in blocks]
        input_features = blocks[0].srcdata['features']
        input_features['user'] = trained_model.user_embed(input_features['user'])
        input_features['item'] = trained_model.item_embed(input_features['item'])
        h = trained_model.get_repr(blocks, input_features)
        for ntype in h.keys():
            y[ntype][output_nodes[ntype]] = h[ntype]
    return y
