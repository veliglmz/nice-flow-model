import torch
import numpy as np
import time

from display import save_scatter_plot
from loss import gaussian_distribution_loss
from sample_generator import generate_spiral
from models import NICEModel
from utils import arg_parse



if __name__ == "__main__":

    torch.manual_seed(42)
    np.random.seed(42)

    args = arg_parse()

    # generate 2D train set
    X = generate_spiral(n=args.n_samples)
    
    x0 = torch.from_numpy(X[:, 0]).view(-1, 1).float()
    x1 = torch.from_numpy(X[:, 1]).view(-1, 1).float()
    
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Device: {device}\n")

    model = NICEModel(n_inputs=args.n_inputs, n_layers=args.n_layers, 
                      n_hiddens=args.n_hiddens)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=args.eps,
                                 betas=(args.beta1, args.beta2))

    print("Starting is training....")
    for e in range(1, args.n_epochs+1):
        batch_idx = torch.randint(0, args.n_samples, (args.batch_size,))    
        optimizer.zero_grad()
        output = model(x0[batch_idx].to(device), x1[batch_idx].to(device))
        loss = gaussian_distribution_loss(output, model.scaling_diag)
        if e % 500 == 0:
            print(f"Epoch {e} | Loss: {loss.item(): .2f}")
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), args.model_path)
    print(f"Model is saved as {args.model_path}.")
    print("Training is done.\n")
    time.sleep(5)

    ############################################################################
    
    print("Testing is starting...")
    model = NICEModel(n_inputs=args.n_inputs, n_layers=args.n_layers, n_hiddens=args.n_hiddens)
    model.load_state_dict(torch.load(args.model_path))
    model.eval().to(device)

    X_hat = torch.randn(1000, 2)
    x0_hat = X_hat[:, 0].view(-1, 1).to(device)
    x1_hat = X_hat[:, 1].view(-1, 1).to(device)
    x0, x1 = model.inverse(x0_hat, x1_hat)
    x0 = x0.cpu().detach().numpy()
    x1 = x1.cpu().detach().numpy()
    X_hat = np.concatenate((x0, x1), axis=1)
    save_scatter_plot(X, X_hat)
    print("Figure is saved.")
    print("Testing is done.")
