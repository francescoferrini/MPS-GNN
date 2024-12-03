import torch
import torch.nn as nn
import random
from torch_geometric.nn import GINConv, GIN, MLP
import torch.nn.functional as F

if torch.cuda.is_available():
    device = torch.device("cuda") 
else:
    device = torch.device("cpu")
class ScoringFunction(nn.Module):
    def __init__(self, input_dim, edge_index, edge_type, rel, num_nodes, alpha, bags, labels, edge_dict):
        super(ScoringFunction, self).__init__()
        self.rel = rel
        self.edge_index = edge_index
        self.edge_dict = edge_dict
        self.labels = labels
        self.alpha = alpha
        self.bags = bags
        self.node_weights = nn.Parameter(torch.rand(num_nodes))
        self.theta = nn.Linear(in_features=input_dim, out_features=1, bias=False)
    def forward(self, x, sampled_bags, alpha_values):
        

        predictions = torch.zeros(len(sampled_bags))

        for j, bag in enumerate(sampled_bags):
            h_values = self.theta(x[bag])
            feat_values = torch.mul(h_values, alpha_values[j].to(device))
            neighbor_sums = []
            for source in bag:
                try:
                    if source in self.edge_dict:
                        sum_value = torch.sum(self.node_weights[self.edge_dict[source]].to(device))
                    else:
                        sum_value = torch.tensor(1.0, device=device)
                    neighbor_sums.append(sum_value)
                except Exception as e:
                    print(f"self.edge_dict: {self.edge_dict[source]}")
                    print(f"Error processing source: {source}")
                    print(f"bag: {bag}")
                    print(f"Relation: {self.rel}")
                    raise e  


            neighbor_sum = torch.stack(neighbor_sums)
            try:
                predictions[j] = torch.sum(neighbor_sum * feat_values)
            except Exception as e:
                print(f"bag: {bag}")
                print(f"h values: {h_values}")
                print(f"alpha_values: {alpha_values[j]}")
                print(f"neigh sum: {neighbor_sum}, {feat_values}")
                raise e
        
        return predictions
    def custom_loss(self, pred, sampled_bags_indices):
        sampled_predictions = pred
        sampled_labels = torch.masked_select(self.labels, torch.tensor(sampled_bags_indices, device=device))
        m = nn.ReLU()
        n = nn.Sigmoid()
        pos_predictions = sampled_predictions[sampled_labels == 1]
        neg_predictions = sampled_predictions[sampled_labels == 0]

        pos_count = pos_predictions.size(0)
        neg_count = neg_predictions.size(0)
        all_pairs = neg_predictions.view(1, neg_count) - pos_predictions.view(pos_count, 1) 
        loss = torch.sum(n(all_pairs))
        return loss
    
    def sampling_bags(self, sampling_size):
        y = self.labels.tolist()
        num_bags = len(self.bags)
        num_y_1 = sum(y)
        num_y_0 = len(self.labels) - num_y_1
        num_selected_y_1 = int(num_y_1 * sampling_size)
        num_selected_y_0 = int(num_y_0 * sampling_size)
        
        selected_indices_y_1 = random.sample([i for i, label in enumerate(y) if label], num_selected_y_1)
        selected_indices_y_0 = random.sample([i for i, label in enumerate(y) if not label], num_selected_y_0)
        
        selected_indices = selected_indices_y_1 + selected_indices_y_0
        selection_vector = [False] * num_bags
        for index in selected_indices:
            selection_vector[index] = True
        
        return selection_vector
    
from torch_geometric.nn import MessagePassing

class MetaPathGNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, relation_index):
        super(MetaPathGNNLayer, self).__init__(aggr='add', flow="target_to_source")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.relation_index = relation_index
        # Theta 0
        self.w_0 = nn.Linear(self.in_channels, self.out_channels)
        # Theta l
        self.w_l = nn.Linear(self.in_channels, self.out_channels)
        # Theta 1
        self.w_1 = nn.Linear(self.in_channels, self.out_channels)
        
    def forward(self, x, edge_index, edge_type, h):
        # Propagate only on nodes connected via relation relation_index
        neig_info = self.propagate(edge_index[:, (edge_type == self.relation_index)], x=h, edge_type=edge_type)
        skipp_conn = self.w_1(x)
        node_hl = self.w_0(h)
        out = neig_info + skipp_conn + node_hl
        
        return out

    def message(self, x_j):
        return x_j
        
    def update(self, aggr_out):
        return self.w_l(aggr_out)

    def adapt_input_dim(self, new_in_channels):
        self.w_0 = nn.Linear(new_in_channels, self.out_channels)
        self.w_l = nn.Linear(new_in_channels, self.out_channels)
        self.inp = new_in_channels

class MetaPathGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, metapath_list):
        super(MetaPathGNN, self).__init__()
        self.metapath_list_length = len(metapath_list)
        for metatree in metapath_list:
            self.gnn_layers = nn.ModuleList([
                    MetaPathGNNLayer(hidden_dim*2, hidden_dim, relation_index)
                    if i == 0 else
                    MetaPathGNNLayer(hidden_dim, hidden_dim, relation_index)
                    for i, relation_index in enumerate(metatree)
                ])
        
        self.mlp_linear = MLP(in_channels=input_dim, hidden_channels=hidden_dim, out_channels=hidden_dim*2, num_layers=3)
        self.fc1 = torch.nn.Linear(hidden_dim * self.metapath_list_length, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        self.log_softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, x, edge_index, edge_type):
        embeddings = []
        
        x = self.mlp_linear(x)
        for j in range(self.metapath_list_length):
            for i, layer in enumerate(self.gnn_layers):
                if i == 0:
                    emb = x
                emb = torch.relu(layer(emb, edge_index, edge_type, emb))
                emb = F.dropout(emb, p=0.5, training=self.training)
            embeddings.append(emb)
        concatenated_embedding = torch.cat(embeddings, dim=1)
        
        emb = F.relu(self.fc1(concatenated_embedding))
        emb = self.fc2(emb)
        emb = self.log_softmax(emb)
        return emb
