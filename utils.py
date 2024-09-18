from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import *
if torch.cuda.is_available():
    device = torch.device("cuda") 
else:
    device = torch.device("cpu")
    

def get_optimizer(net, lr):
  return torch.optim.Adam(net.parameters(), lr=lr)#, weight_decay=wd)

def get_cost_function(class_weights):
  return nn.NLLLoss(weight=class_weights)

def test(net, data_test_val, cost_function, arg):
  net.eval() 
  with torch.no_grad():
    # Load data into GPU
    targets = data_test_val.mpgnn_y.to(device)
    # Forward pass
    outputs = net(data_test_val.x, data_test_val.edge_index, data_test_val.edge_type)
    # Calculate the loss
    if arg == "test": loss = cost_function(outputs.squeeze()[data_test_val.test_mask], targets[data_test_val.test_mask])
    else: loss = cost_function(outputs.squeeze()[data_test_val.val_mask], targets[data_test_val.val_mask])
    # F1
    predicted = torch.argmax(outputs, 1)
    if arg == "test":  
      f1 = f1_score(targets.cpu().numpy()[data_test_val.test_mask], predicted.detach().cpu().numpy()[data_test_val.test_mask], average = 'micro')
      auc = roc_auc_score(targets.cpu().numpy()[data_test_val.test_mask], predicted.cpu().detach().numpy()[data_test_val.test_mask], average = "macro")
    else: 
      f1 = f1_score(targets.cpu().numpy()[data_test_val.val_mask], predicted.detach().cpu().numpy()[data_test_val.val_mask], average = 'micro')
      auc = roc_auc_score(targets.cpu().numpy()[data_test_val.val_mask], predicted.cpu().detach().numpy()[data_test_val.val_mask], average = "macro")
    return loss.item(), f1, auc

def train(net, data_train, optim, cost_function): 
  net.train() 
  # Load data into GPU
  targets = data_train.mpgnn_y.to(device)
  # Forward pass
  outputs = net(data_train.x, data_train.edge_index, data_train.edge_type)
  loss = cost_function(outputs.squeeze()[data_train.train_mask], targets[data_train.train_mask])
  # Backward pass
  loss.backward()
  optim.step() 
  # Reset the gradients
  optim.zero_grad()
  # F1
  predicted = torch.argmax(outputs, 1)
  f1 = f1_score(targets.cpu().numpy()[data_train.train_mask], predicted.detach().cpu().numpy()[data_train.train_mask], average = 'micro')
  auc = roc_auc_score(targets.cpu().numpy()[data_train.train_mask], predicted.detach().cpu().numpy()[data_train.train_mask], average = "macro")
  return loss.item(), f1, auc

def mpgnn(data_mpgnn, meta_path, pre_trained_model, hidden_dim):
  input_dim = data_mpgnn.x.size(1)
  output_dim = len(torch.unique(data_mpgnn.mpgnn_y))
  
  if pre_trained_model:
    model = MetaPathGNN(input_dim, hidden_dim, output_dim, meta_path)
    model.load_state_dict(torch.load(pre_trained_model[0]))
    model.to(device)
    data_mpgnn.to(device)
    class_weights=torch.tensor([1., torch.tensor(data_mpgnn.y.tolist().count(0.)/data_mpgnn.y.tolist().count(1.))/2]).to(device)
    cost_function = get_cost_function(class_weights=class_weights)
    test_loss, test_f1, test_auc = test(model, data_mpgnn, cost_function, arg="test")
    return test_f1, test_auc
    
  best_val, best_model = 0., None
  learning_rate = 0.001
  epochs = 1000

  model = MetaPathGNN(input_dim, hidden_dim, output_dim, meta_path)
  model.to(device)
  data_mpgnn.to(device)
  optimizer = get_optimizer(model, learning_rate)#, weight_decay)
  # Creates the cost function
  class_weights=torch.tensor([1., torch.tensor(data_mpgnn.y.tolist().count(0.)/data_mpgnn.y.tolist().count(1.))/2]).to(device)
  cost_function = get_cost_function(class_weights=class_weights)
  # Training procedure
  prev_val=0
  for e in range(epochs):
    train_loss, train_f1, train_auc = train(model, data_mpgnn, optimizer, cost_function)
    val_loss, val_f1, val_auc = test(model, data_mpgnn, cost_function, arg="val")

    if val_auc > prev_val: 
      best_model = copy.deepcopy(model)
      prev_val = val_auc
      count = 0
    else:
      count += 1 
    if count == 200:
      break 
  
  #trainable_params = sum(p.numel() for p in best_model.parameters() if p.requires_grad)   
  test_loss, test_f1, test_auc = test(best_model, data_mpgnn, cost_function, arg="test")
  #torch.save(best_model.state_dict(), "/mnt/cimec-storage6/users/antonio.longa/tmp/anto2/MPS-GNN/models/mondial.pth")
  return test_f1, test_auc
