import torch
import pickle

class SteeringLayer(torch.nn.Module):
    """Our custom steering layer for an LLM."""
    
    def __init__(self, layer_of_interest):
        super().__init__()
        self.layer_of_interest = layer_of_interest
        if next(self.layer_of_interest.parameters()).device.type == 'cuda':
            self.device = torch.device(f"{next(self.layer_of_interest.parameters()).device.type}:{next(self.layer_of_interest.parameters()).device.index}")
        else:
            self.device = torch.device('cpu')
            
        self.steering_vector = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.empty((1, 4096), device=self.device,  dtype=torch.float32)))
        self.add_steering = True
        self.ignore_activations = False
        self.shift_with_new_idea = False
        self.a = 1.0
        self.b = 1.0

    def reset_steering_vector(self):
        if next(self.layer_of_interest.parameters()).device.type == 'cuda':
            self.device = torch.device(f"{next(self.layer_of_interest.parameters()).device.type}:{next(self.layer_of_interest.parameters()).device.index}")
        else:
            self.device = torch.device('cpu')
        self.steering_vector = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.empty((1, 4096), device=self.device,  dtype=torch.float32)))

    def load_steering_vector(self, steering_vector_path):
        if next(self.layer_of_interest.parameters()).device.type == 'cuda':
            self.device = torch.device(f"{next(self.layer_of_interest.parameters()).device.type}:{next(self.layer_of_interest.parameters()).device.index}")
        else:
            self.device = torch.device('cpu')
        with open(steering_vector_path, "rb") as input_file:
            sd = pickle.load(input_file)
        self.steering_vector = torch.nn.Parameter(sd[steering_vector_path.split('/')[-1].split('.')[0].split('_')[-1]].to(self.device))
        pass

    def forward(self, *x): 

        if self.shift_with_new_idea:
              return self.layer_of_interest(*x) + self.b * (self.steering_vector - self.layer_of_interest(*x))

        if self.add_steering:
            if self.ignore_activations:
                return self.layer_of_interest(*x) * 0.0 + self.b * self.steering_vector
            return self.a * self.layer_of_interest(*x) + self.b * self.steering_vector
        return self.layer_of_interest(*x) 