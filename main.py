import time
import pickle
import numpy as np

from sklearn.metrics import f1_score

import torch 
import torch.optim as optim
from torch_geometric.data import Data
torch.autograd.set_detect_anomaly(True)

from models import ScoringFunction
from utils import mpgnn
import argparse
import ast

    
if torch.cuda.is_available():
    device = torch.device("cuda") 
else:
    device = torch.device("cpu")



def create_new_data(data_object, theta, rel):

    data_clone = data_object.clone()
    edge_dict = {k.item(): data_clone.edge_index[1, (data_clone.edge_index[0] == k) & (data_clone.edge_type == rel)].tolist() for k in torch.unique(data_clone.edge_index[0])}
    intermediate_bags = [list({neighbor for node in old_bag if node in edge_dict for neighbor in edge_dict[node]}) for old_bag in data_clone.bags]

    #theta  = theta[rel].detach().squeeze(0)
    theta  = theta.detach().squeeze(0)
    alpha = {}
    for i in range(len(data_clone.bags)):
        #print(data_clone.bags[i])
        for new_node_in_new_bag in intermediate_bags[i]:
            #print('\t', new_node_in_new_bag)
            for old_node in data_clone.bags[i]:
                if new_node_in_new_bag in edge_dict[old_node]:
                    if (new_node_in_new_bag, tuple(intermediate_bags[i])) not in alpha:
                        alpha[(new_node_in_new_bag, tuple(intermediate_bags[i]))] = 0
                    alpha[(new_node_in_new_bag, tuple(intermediate_bags[i]))] += data_clone.alpha[(old_node, tuple(data_clone.bags[i]))]*theta.dot(data_clone.x[old_node])
            alpha[(new_node_in_new_bag, tuple(intermediate_bags[i]))] = alpha[(new_node_in_new_bag, tuple(intermediate_bags[i]))].item()
    
    data_clone.alpha = alpha
    new_bags = [lst for lst in intermediate_bags if lst]
    new_labels = torch.tensor([data_clone.y[i] for i in range(len(data_clone.y)) if intermediate_bags[i]])
    data_clone.bags, data_clone.y = new_bags, new_labels
    
    flattened_list = [item for sublist in data_clone.bags for item in sublist]    
    unique_elements = set(flattened_list)
    data_clone.target_ids = sorted(unique_elements)
    del data_clone.edge_dict_rel
    return data_clone

def extract_relation_types(edge_index, edge_type, target_ids):
    mask = torch.isin(edge_index[0].cpu(), torch.tensor(target_ids))
    filtered_edge_type = torch.unique(edge_type[mask]).tolist()
    return filtered_edge_type
    
def data_object(data):
    # Alpha
    alphas = torch.ones(data.x.size(0))

    # Bags
    bags = []

    bags = [[i] for i in range(len(data.y))]
    # Alphas
    alpha = { (elm, tuple(bag)): 1. for bag in bags for elm in bag }
    target_ids = [i for i in range(len(data.y))]
    data_obj = Data(edge_index=data.edge_index, edge_type=data.edge_type, x=data.x, y=data.y, target_ids=target_ids, num_nodes=data.x.size(0), alpha=alpha, bags=bags)
    return data_obj

def score(relation_type, data_obj, sampling_size):
    scores = {}
    alpha_values = []
    # Sum aggregation for first iteration then max
    mask = data_obj.edge_type == relation_type
    edge_index_rel = data_obj.edge_index[:, mask]
    edge_index_rel = edge_index_rel.to(device)
    
    target_ids_tensor = torch.tensor(data_obj.target_ids, dtype=torch.long)
    target_ids_tensor = target_ids_tensor.to(device)
    unique_nodes = torch.unique(edge_index_rel[0])
    unique_nodes = unique_nodes.to(device)
    filtered_nodes = unique_nodes[torch.isin(unique_nodes, target_ids_tensor)]
    edge_dict_rel = {k.item(): list(set(edge_index_rel[1, (edge_index_rel[0] == k)].tolist())) for k in filtered_nodes}
    data_obj.edge_dict_rel = edge_dict_rel
    data_obj = data_obj.to(device)
    prev_loss, count = float('inf'), 0
    # Iterate various different targets set
    scores[relation_type] = []
            
    scoring_function = ScoringFunction(data_obj.x.size(1), 
                                        data_obj.edge_index, 
                                        data_obj.edge_type, 
                                        relation_type, 
                                        data_obj.num_nodes, 
                                        data_obj.alpha, 
                                        data_obj.bags, 
                                        data_obj.y,
                                        data_obj.edge_dict_rel)
    scoring_function.to(device)
    optimizer = optim.Adam(scoring_function.parameters(), lr=0.1)
    start_time = time.time()
        
    sampled_bags_indices = scoring_function.sampling_bags(sampling_size)
    sampled_bags = [elem for elem, selezionato in zip(data_obj.bags, sampled_bags_indices) if selezionato]
    for j, bag in enumerate(sampled_bags):
        alpha_values.append(torch.tensor([data_obj.alpha[(i, tuple(bag))] for i in bag]))

    for epoch in range(500):
            optimizer.zero_grad()
            predictions = scoring_function(data_obj.x, sampled_bags, alpha_values)
            predictions = predictions.to(device)
            loss = scoring_function.custom_loss(predictions, sampled_bags_indices)
            loss.backward()
            optimizer.step()
            with torch.no_grad():        
                #scoring_function.node_weights[:] = torch.clamp(scoring_function.node_weights, min = 0.0, max = 1.0)
                scoring_function.theta.weight[:] = torch.clamp(scoring_function.theta.weight, min = 0.0)#, max = 1.0)
            if loss >= prev_loss:
                count +=1
            else:
                prev_loss = loss
            if count == 10:
                break
    parameters = scoring_function.theta.weight
    end_time = time.time()
    elapsed_time = end_time-start_time
    #th[relation_type.item()] = scoring_function.theta.weight[:]


    # print(f'Relation {relation_type.item()}, Final Loss: {loss}, time: {elapsed_time}, rank: {rank}')  
    return relation_type, loss.item(), parameters

def best_result(result, n):
    sorted_tuples = sorted(result, key=lambda x: x[1])
    if n == 2:
        if len(sorted_tuples) == 0:
            return None, None
        elif len(sorted_tuples) > 1:
            min_tuple = sorted_tuples[0]
            second_min_tuple = sorted_tuples[1]
            return min_tuple, second_min_tuple
        else: 
            min_tuple = sorted_tuples[0]
            return min_tuple, None
    elif n == 1:
        if len(sorted_tuples) == 0:
            return None
        elif len(sorted_tuples) > 0:
            min_tuple = sorted_tuples[0]
            return min_tuple
        
def main(args):  

    metapaths, finalmetapaths, final = [], [], []
    K, L_MAX, iteration = 2, 4, 0
    partial_results, res = {}, {}
    
    with open(args.folder+"data_"+args.dataset+".pkl", "rb") as file:
        data = pickle.load(file)
    data_mpgnn = data.clone()
    data = data_object(data) 
    actual_relations = extract_relation_types(data.edge_index, data.edge_type, data.target_ids)

    result = []
    for rel in actual_relations:
        partial_result = score(rel, data, sampling_size=0.4)
        result.append(partial_result)

    tupla, tupla2 = best_result(result, n=K)
    relat, theta, relat2, theta2 = tupla[0], tupla[2], tupla2[0], tupla2[2]

    metapaths.insert(0, [relat])
    metapaths.insert(0, [relat2])
    
    finalmetapaths.insert(0, [relat])
    finalmetapaths.insert(0, [relat2])
    
    test_f1, test_auc = mpgnn(data_mpgnn, [[relat]])

    partial_results[str([relat])] = {}
    partial_results[str([relat])]['relation'] = relat
    partial_results[str([relat])]['meta'] = [relat]
    partial_results[str([relat])]['theta'] = theta
    partial_results[str([relat])]['f1'] = test_f1
    partial_results[str([relat])]['data'] = data.clone()
    
    test_f1, test_auc = mpgnn(data_mpgnn, [[relat2]])

    partial_results[str([relat2])] = {}
    partial_results[str([relat2])]['relation'] = relat2
    partial_results[str([relat2])]['meta'] = [relat2]
    partial_results[str([relat2])]['theta'] = theta2
    partial_results[str([relat2])]['f1'] = test_f1
    partial_results[str([relat2])]['data'] = data.clone()
    
    while(iteration < L_MAX):
        tmp_metapaths = []
        for i in range(0, len(metapaths)):
            data, theta, relation = partial_results[str(metapaths[i])]['data'],  partial_results[str(metapaths[i])]['theta'], partial_results[str(metapaths[i])]['relation']
            data_new = create_new_data(data, theta, relation)
            if len(data_new.target_ids) > 0:
                actual_relations = extract_relation_types(data.edge_index, data.edge_type, data.target_ids)
                
                result = []
                for rel in actual_relations:
                    partial_result = score(rel, data_new, sampling_size=0.4)
                    result.append(partial_result)
                tupla = best_result(result, n=1)
                relat, theta = tupla[0], tupla[2]
                
                tmp_meta = partial_results[str(metapaths[i])]['meta']
                tmp_meta.insert(0, relat)
                
                tmp_metapaths.append(tmp_meta)
                finalmetapaths.append(tmp_meta)
                test_f1, test_auc = mpgnn(data_mpgnn, [tmp_meta])
     
                partial_results[str(tmp_meta)] = {}
                partial_results[str(tmp_meta)]['relation'] = relat
                partial_results[str(tmp_meta)]['meta'] = tmp_meta
                partial_results[str(tmp_meta)]['theta'] = theta
                partial_results[str(tmp_meta)]['f1'] = test_f1
                partial_results[str(tmp_meta)]['data'] = data_new.clone()

        metapaths=tmp_metapaths
        iteration += 1

    for element in finalmetapaths:
        if element not in final: final.append(element)
    for element in final:
        res[str(element)] = [partial_results[str(element)]['f1']]
        # print(partial_results[str(element)]['meta'])
        # print(partial_results[str(element)]['f1'])
    
    ordered_res = dict(sorted(res.items(), key=lambda item: item[1][0], reverse=True))
    meta_1_1 = [ast.literal_eval(list(ordered_res.keys())[0]), ast.literal_eval(list(ordered_res.keys())[1])]
    f_1_1, test_auc = mpgnn(data_mpgnn, meta_1_1)
    
    meta = [ast.literal_eval(list(ordered_res.keys())[0]), ordered_res[list(ordered_res.keys())[0]]]
    
    if f_1_1 > ordered_res[list(ordered_res.keys())[0]]:
        meta_2_1 = [ast.literal_eval(list(ordered_res.keys())[0]), ast.literal_eval(list(ordered_res.keys())[1]), ast.literal_eval(list(ordered_res.keys())[2])]
        f_2_1, test_auc = mpgnn(data_mpgnn, meta_2_1)
        meta = [meta_1_1, f_1_1]
        if f_2_1 > f_1_1:
            meta = [meta_2_1, f_2_1]
    print(meta)
 
        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='learning meta-paths')
    parser.add_argument("--hidden_dim", type=int, required=True,
            help="hidden dimension")
    parser.add_argument("--dataset", type=str, required=True,
            help="dataset")
    parser.add_argument("--folder", type=str, required=True,
            help="folder")
    args = parser.parse_args()
    #print(args, flush=True)
    main(args)
