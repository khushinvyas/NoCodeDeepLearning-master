from flask import Flask, request, jsonify, send_file
import uuid
import io
import textwrap

app = Flask(__name__)

cnn_configs = {}

@app.route('/get_convolutional_neural_net', methods=['POST'])
def get_convolutional_neural_net():
    try:
        data = request.json
        
        required_keys = [
            "inputLayer", "convLayers", "activationLayers", "poolingLayers",
            "FullyConnectedLayer", "outputLayer", "optimizer", "lossFunction",
            "learningRate", "numEpochs"
        ]
        
        if not all(key in data for key in required_keys):
            return jsonify({"error": "Invalid data format"}), 400

        # Validate nested structures
        if not all(key in data["inputLayer"] for key in ["width", "height", "channels", "batchSize"]):
            return jsonify({"error": "Invalid input layer format"}), 400

        if not all(key in data["outputLayer"] for key in ["units", "activation"]):
            return jsonify({"error": "Invalid output layer format"}), 400

        # Generate a unique ID for this configuration
        config_id = str(uuid.uuid4())
        
        # Store the configuration
        cnn_configs[config_id] = data
        
        response_data = {
            "id": config_id,
            "config": data
        }
        
        print(f"Stored CNN configuration: {response_data}")
        
        return jsonify(response_data), 200
    except Exception as e:
        # Handle errors
        return jsonify({"error": str(e)}), 500
    

@app.route('/download_cnn_file/<config_id>', methods=['GET'])
def download_cnn_file(config_id):
    if config_id not in cnn_configs:
        return jsonify({"error": "Configuration not found"}), 404
    
    config = cnn_configs[config_id]
    pytorch_code = pytorch_gen_file(config)
    
    # Create an in-memory file-like object
    buffer = io.BytesIO()
    buffer.write(pytorch_code.encode('utf-8'))
    buffer.seek(0)
    
    return send_file(
        buffer,
        as_attachment=True,
        download_name='dynamic_cnn.py',
        mimetype='text/x-python'
    )
    

def pytorch_gen_file(config):
    code = f"""
import torch
import torch.nn as nn
import torch.optim as optim

class DynamicCNN(nn.Module):
    def __init__(self):
        super(DynamicCNN, self).__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        in_channels = {config['inputLayer']['channels']}
        
        # Convolutional layers
        {generate_conv_layers(config)}
        
        # Calculate the size of the flattened feature map
        with torch.no_grad():
            x = torch.randn(1, {config['inputLayer']['channels']}, 
                            {config['inputLayer']['height']}, 
                            {config['inputLayer']['width']})
            for layer in self.layers:
                x = layer(x)
            flattened_size = x.view(1, -1).size(1)
        
        # Fully connected layers
        {generate_fc_layers(config)}
        
        # Output layer
        self.layers.append(nn.Linear(flattened_size, {config['outputLayer']['units']}))
        self.layers.append(self.get_activation('{config['outputLayer']['activation']}'))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    @staticmethod
    def get_activation(name):
        if name.lower() == 'relu':
            return nn.ReLU()
        elif name.lower() == 'softmax':
            return nn.Softmax(dim=1)
        else:
            raise ValueError(f"Unsupported activation function: {{name}}")

def create_model_from_config():
    model = DynamicCNN()
    
    # Set up optimizer
    optimizer = optim.{config['optimizer'].capitalize()}(model.parameters(), lr={config['learningRate']})
    
    # Set up loss function
    criterion = nn.CrossEntropyLoss()
    
    return model, optimizer, criterion

def main():
    model, optimizer, criterion = create_model_from_config()
    
    num_epochs = {config['numEpochs']}
    
    print(model)
    print(f"Optimizer: {{optimizer}}")
    print(f"Loss function: {{criterion}}")
    print(f"Number of epochs: {{num_epochs}}")

if __name__ == "__main__":
    main()
"""
    return textwrap.dedent(code)

def generate_conv_layers(config):
    conv_code = ""
    for i, (conv, act, pool) in enumerate(zip(config['convLayers'], config['activationLayers'], config['poolingLayers'])):
        conv_code += f"""
        self.layers.append(nn.Conv2d(in_channels, {conv['filters']}, {conv['kernelSize']}, 
                                     stride={conv['stride']}, padding='{conv['padding']}'))
        self.layers.append(self.get_activation('{act}'))
        self.layers.append(nn.MaxPool2d({pool['poolSize']}, stride={pool['stride']}))
        in_channels = {conv['filters']}
        """
    return textwrap.dedent(conv_code)

def generate_fc_layers(config):
    fc_code = ""
    for fc in config['FullyConnectedLayer']:
        fc_code += f"""
        self.layers.append(nn.Linear(flattened_size, {fc['units']}))
        self.layers.append(self.get_activation('{fc['activation']}'))
        flattened_size = {fc['units']}
        """
    return textwrap.dedent(fc_code)

if __name__ == '__main__':
    app.run(debug=True)
