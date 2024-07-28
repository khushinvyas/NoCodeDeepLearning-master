from flask import Flask, request, jsonify, send_file
import uuid
import io
import textwrap

app = Flask(__name__)

rnn_configs = {}

@app.route('/get_recurrent_neural_net', methods=['POST'])
def get_recurrent_neural_net():
    try:
        data = request.json
        
        required_keys = [
            "inputLayer", "rnnLayers", "FullyConnectedLayer", "outputLayer", 
            "optimizer", "lossFunction", "learningRate", "numEpochs"
        ]
        
        if not all(key in data for key in required_keys):
            return jsonify({"error": "Invalid data format"}), 400

        # Validate nested structures
        if not all(key in data["inputLayer"] for key in ["inputSize", "sequenceLength", "batchSize"]):
            return jsonify({"error": "Invalid input layer format"}), 400

        if not all(key in data["outputLayer"] for key in ["units", "activation"]):
            return jsonify({"error": "Invalid output layer format"}), 400

        # Generate a unique ID for this configuration
        config_id = str(uuid.uuid4())
        
        # Store the configuration
        rnn_configs[config_id] = data
        
        response_data = {
            "id": config_id,
            "config": data
        }
        
        print(f"Stored RNN configuration: {response_data}")
        
        return jsonify(response_data), 200
    except Exception as e:
        # Handle errors
        return jsonify({"error": str(e)}), 500

@app.route('/download_rnn_file/<config_id>', methods=['GET'])
def download_rnn_file(config_id):
    if config_id not in rnn_configs:
        return jsonify({"error": "Configuration not found"}), 404
    
    config = rnn_configs[config_id]
    pytorch_code = pytorch_gen_file(config)
    
    # Create an in-memory file-like object
    buffer = io.BytesIO()
    buffer.write(pytorch_code.encode('utf-8'))
    buffer.seek(0)
    
    return send_file(
        buffer,
        as_attachment=True,
        download_name='dynamic_rnn.py',
        mimetype='text/x-python'
    )

def pytorch_gen_file(config):
    code = f"""
import torch
import torch.nn as nn
import torch.optim as optim

class DynamicRNN(nn.Module):
    def __init__(self):
        super(DynamicRNN, self).__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        input_size = {config['inputLayer']['inputSize']}
        sequence_length = {config['inputLayer']['sequenceLength']}
        
        # RNN layers
        {generate_rnn_layers(config)}
        
        # Fully connected layers
        {generate_fc_layers(config)}
        
        # Output layer
        self.layers.append(nn.Linear(hidden_size, {config['outputLayer']['units']}))
        self.layers.append(self.get_activation('{config['outputLayer']['activation']}'))

    def forward(self, x):
        # RNN layers
        for layer in self.rnn_layers:
            x, _ = layer(x)
        
        # Take the output of the last time step
        x = x[:, -1, :]
        
        # Fully connected and output layers
        for layer in self.fc_layers:
            x = layer(x)
        return x

    @staticmethod
    def get_activation(name):
        if name.lower() == 'relu':
            return nn.ReLU()
        elif name.lower() == 'tanh':
            return nn.Tanh()
        elif name.lower() == 'softmax':
            return nn.Softmax(dim=1)
        else:
            raise ValueError(f"Unsupported activation function: {{name}}")

def create_model_from_config():
    model = DynamicRNN()
    
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

def generate_rnn_layers(config):
    rnn_code = """
    self.rnn_layers = nn.ModuleList()
    """
    for i, rnn in enumerate(config['rnnLayers']):
        rnn_type = rnn['type']
        hidden_size = rnn['hiddenSize']
        num_layers = rnn.get('numLayers', 1)
        bidirectional = rnn.get('bidirectional', False)
        dropout = rnn.get('dropout', 0)
        
        rnn_code += f"""
        self.rnn_layers.append(nn.{rnn_type}(
            input_size={'hidden_size' if i > 0 else 'input_size'},
            hidden_size={hidden_size},
            num_layers={num_layers},
            bidirectional={bidirectional},
            dropout={dropout},
            batch_first=True
        ))
        """
        if bidirectional:
            hidden_size *= 2
    
    rnn_code += f"""
    hidden_size = {hidden_size}
    """
    return textwrap.dedent(rnn_code)

def generate_fc_layers(config):
    fc_code = """
    self.fc_layers = nn.ModuleList()
    """
    for fc in config['FullyConnectedLayer']:
        fc_code += f"""
        self.fc_layers.append(nn.Linear(hidden_size, {fc['units']}))
        self.fc_layers.append(self.get_activation('{fc['activation']}'))
        hidden_size = {fc['units']}
        """
    return textwrap.dedent(fc_code)

if __name__ == '__main__':
    app.run(debug=True)