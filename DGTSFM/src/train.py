import torch
import torch.nn as nn
import torch.optim as optim

def train_model(model, data, targets, num_epochs, learning_rate, weight_decay):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
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
        
        if (epoch+1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    return model, loss_values
