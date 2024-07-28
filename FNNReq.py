from flask import Flask, request, jsonify, send_file
import uuid

import io
neural_net_configs = {}

def FnnNueralNet():
    try:
            
            data = request.json
            
            if not all(key in data for key in ["inputLayerSize", "hiddenLayerSize", "HiddenAct", "OutputlayerSize", "OutputAct"]):
                return jsonify({"error": "Invalid data format"}), 400

            config_id = str(uuid.uuid4())
            
            neural_net_configs[config_id] = data
            
            response_data = {
                "id": config_id,
                "config": data
            }
            
            print(f"Stored configuration: {response_data}")
            
            return jsonify(response_data), 200
    except Exception as e:
        # Handle errors
        return jsonify({"error": str(e)}), 500


def FnnDownloadFile(config_id):
        if config_id not in neural_net_configs:
            return jsonify({"error": "Configuration not found"}), 404

        config = neural_net_configs[config_id]

        neural_net_code = f'''
                import torch
                import torch.nn as nn

                def get_activation(activation_name):
                    if activation_name == "ReLU":
                        return nn.ReLU()
                    elif activation_name == "Softmax":
                        return nn.Softmax(dim=-1)
                    elif activation_name == "Sigmoid":
                        return nn.Sigmoid()
                    else:
                        raise ValueError(f"Unsupported activation function: {{activation_name}}")

                def create_neural_network():
                    class NeuralNetwork(nn.Module):
                        def __init__(self):
                            super(NeuralNetwork, self).__init__()
                            self.input_layer = nn.Linear({config['inputLayerSize']}, {config['hiddenLayerSize']})
                            self.hidden_activation = get_activation("{config['HiddenAct']}")
                            self.output_layer = nn.Linear({config['hiddenLayerSize']}, {config['OutputlayerSize']})
                            self.output_activation = get_activation("{config['OutputAct']}")
                        
                        def forward(self, x):
                            x = self.input_layer(x)
                            x = self.hidden_activation(x)
                            x = self.output_layer(x)
                            x = self.output_activation(x)
                            return x
                    
                    return NeuralNetwork()

                # Create and print the neural network
                model = create_neural_network()
                print(model)
            '''
            
        # Create a BytesIO object
        buffer = io.BytesIO()
        buffer.write(neural_net_code.encode('utf-8'))
        buffer.seek(0)

        return send_file(buffer,
                        as_attachment=True,
                        download_name=f'neural_network_{config_id}.py',
                        mimetype='text/x-python')

