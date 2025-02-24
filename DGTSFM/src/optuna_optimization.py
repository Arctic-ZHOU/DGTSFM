import torch
import torch.nn as nn
import optuna
from gnn import GATModel

def objective(trial, data, targets):
    hidden_dim = trial.suggest_categorical('hidden_dim', [32, 64, 128, 256])
    num_heads = trial.suggest_categorical('num_heads', [4, 8, 16])
    dropout = trial.suggest_float('dropout', 0.2, 0.8)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    num_layers = trial.suggest_int('num_layers', 1, 3)
    num_epochs = trial.suggest_int('num_epochs', 500, 1500, step=100) 

    model = GATModel(
        num_in_features=data[0].shape[1],
        num_hidden_features=hidden_dim,
        num_out_features=targets.shape[1],
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model.train()
    loss_values = []
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output_tuple = model(data)
        output = output_tuple[0]  
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        loss_values.append(loss.item())

        trial.report(loss.item(), epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return loss_values[-1]

def optimize(data, targets):
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda trial: objective(trial, data, targets), n_trials=50)

    print(f"Best trial: {study.best_trial.params}")
    return study.best_trial.params
