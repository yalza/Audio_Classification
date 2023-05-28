# ----------------------------
# Inference
# ----------------------------

import torch
from Training import *

def inference (model, test_dl):
    correct_prediction = 0
    total_prediction = 0

    # Disable gradient updates
    with torch.no_grad():
        for data in test_dl:
            # Get the input features and target labels, and put them on the GPU
            inputs, labels = data[0].to(device), data[1].to(device)

            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            # Get predictions
            outputs = model(inputs)

            # Get the predicted class with the highest score
            _, prediction = torch.max(outputs,1)
            # Count of predictions that matched the target label
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]
        
    acc = correct_prediction/total_prediction
    print(f'Accuracy: {acc:.2f}, Total items: {total_prediction}')

# Run inference on trained model with the validation set load best model weights
model_inf = nn.DataParallel(AudioClassifier())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_inf = model_inf.to(device)
model_inf.load_state_dict(torch.load('model.pt'))
model_inf.eval()

inference(model_inf, val_dl)