import torch
import torch.nn as nn


class ReLU_MLP(nn.Module):
    def __init__(self, n_inputs, n_layers, n_hiddens):
        super(ReLU_MLP, self).__init__()
        self.n_inputs = n_inputs
        self.n_layers = n_layers
        self.n_hiddens = n_hiddens

        self.input_layer = nn.Sequential(nn.Linear(n_inputs, n_hiddens))
                                          
        self.hidden_layers = nn.Sequential()
        for _ in range(n_layers):
            self.hidden_layers.append(nn.Linear(n_hiddens, n_hiddens))
            self.hidden_layers.append(nn.ReLU())

        self.output_layer = nn.Sequential(nn.Linear(n_hiddens, n_inputs))

    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x



class AdditiveCouplingLayer(nn.Module):
    def __init__(self, n_inputs, n_layers, n_hiddens, type_):
        super(AdditiveCouplingLayer, self).__init__()

        self.n_inputs = n_inputs
        self.n_layers = n_layers
        self.n_hiddens = n_hiddens
        self.type_ = type_

        self.m = ReLU_MLP(n_inputs=n_inputs, n_layers=n_layers, n_hiddens=n_hiddens)
    
    def forward(self, params):
        x0, x1 = params
        if self.type_ == "odd":
            y0 = x0
            y1 = x1 + self.m(x0)
        else:
            y1 = x1
            y0 = x0 + self.m(x1)
        return y0, y1

    def inverse(self, params):
        y0, y1 = params
        if self.type_ == "odd":
            x0 = y0
            x1 = y1 - self.m(y0)
        else:
            x1 = y1
            x0 = y0 - self.m(y1)
        return x0, x1



class NICEModel(nn.Module):
    def __init__(self, n_inputs, n_layers, n_hiddens):
        super(NICEModel, self).__init__()
        self.n_inputs = n_inputs
        self.n_layers = n_layers
        self.n_hiddens = n_hiddens

        self.layer1 = AdditiveCouplingLayer(n_inputs=int(n_inputs/2), n_layers=n_layers, n_hiddens=n_hiddens, type_="odd")
        self.layer2 = AdditiveCouplingLayer(n_inputs=int(n_inputs/2), n_layers=n_layers, n_hiddens=n_hiddens, type_="even")
        self.layer3 = AdditiveCouplingLayer(n_inputs=int(n_inputs/2), n_layers=n_layers, n_hiddens=n_hiddens, type_="odd")
        self.layer4 = AdditiveCouplingLayer(n_inputs=int(n_inputs/2), n_layers=n_layers, n_hiddens=n_hiddens, type_="even")
        self.scaling_diag = nn.Parameter(torch.ones(n_inputs))
      

    def forward(self, x0, x1):
        y0, y1 = self.layer1((x0, x1))
        y0, y1 = self.layer2((y0, y1))
        y0, y1 = self.layer3((y0, y1))
        y0, y1 = self.layer4((y0, y1))
        ys = torch.concat((y0, y1), axis=1)
        ys = torch.matmul(ys, torch.diag(torch.exp(self.scaling_diag)))
        return ys

    def inverse(self, y0, y1):
        with torch.no_grad():
            ys = torch.concat((y0, y1), axis=1)
            xs = torch.matmul(ys, torch.diag(torch.reciprocal(torch.exp(self.scaling_diag))))
            x0, x1 = xs[:, 0].view(-1, 1), xs[:, 1].view(-1, 1) 
            x0, x1 = self.layer4.inverse((x0, x1))
            x0, x1 = self.layer3.inverse((x0, x1))
            x0, x1 = self.layer2.inverse((x0, x1))
            x0, x1 = self.layer1.inverse((x0, x1))
        return x0, x1

        