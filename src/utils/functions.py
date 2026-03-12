import numpy as np
import torch

def compute_asymmetry(J: np.ndarray) -> float:

    J = np.array(J)
    asymmetry_matrix = J - J.T
    squared_diff_matrix = np.square(asymmetry_matrix)
    asymmetry = np.mean(squared_diff_matrix)

    return asymmetry

def start_overlap_binary(xi: torch.Tensor, init_overlap: float) -> torch.Tensor:

    init_vectors = xi.clone()
    X, N, d = init_vectors.shape

    # For each position of X
    for x in range(X):
        # Calculate the number of rows to flip 
        num_rows_to_flip = int(N * ((1-init_overlap) / 2.))

        # Randomly choose 'num_rows_to_flip' unique indices from the N rows
        indices_to_flip = torch.randperm(N)[:num_rows_to_flip]
        init_vectors[x, indices_to_flip, :] *= -1

    return init_vectors

def basins_of_attraction_xi(init_overlaps_array, model, dataset, num_of_run, n, device):
    max_overlap_xi_list = []

    for init_overlap in init_overlaps_array:
        input_vectors = start_overlap_binary(dataset.xi, init_overlap)
        input_vectors = model.normalize_x(input_vectors)
        input_vectors = input_vectors.to(device)
        overlap_max_n_xi, overlap_max_n_f, max_overlap_xi, max_overlap_f, final_overlap, _, overlap_argmax = converge(input_vectors[:num_of_run], model, dataset, n)
        max_overlap_xi_list.append(overlap_max_n_xi[:num_of_run,-1])

    max_overlap_xi_array = np.array(max_overlap_xi_list)

    return max_overlap_xi_array

def max_overlap(x_new, vectors):
    """
    Computes the maximum overlap and corresponding indices between x_new and vectors.

    Args:
    - x_new: Tensor of shape [batch_size, n, N, d].
    - vectors: Tensor of shape [X, N, d].

    Returns:
    - overlap_max: Tensor of maximum overlap values of shape [batch_size].
    - max_values: Tensor of corresponding max values of shape [batch_size].
    """
    batch_size, n, N, d = x_new.shape
    X = vectors.shape[0]

    dot_products = torch.einsum('bnid,xid->bnxi', x_new, vectors)
    dot_products_mean = dot_products.mean(dim=3)    #[b,n,x]
    overlap_max_n, overlap_argmax_n = torch.max(dot_products_mean, dim=-1)
    overlap_max, overlap_argmax = torch.max(overlap_max_n, dim=1)
    max_values = overlap_max_n.gather(1, overlap_argmax.unsqueeze(-1)).squeeze()

    return overlap_max_n.cpu().numpy(), overlap_max.cpu().numpy(), max_values.cpu().numpy(), overlap_argmax.cpu().numpy()

def converge(input_vectors, model, dataset, n, features=False):
    """
    Function to perform dynamics and compute overlaps.

    Args:
    - input_vectors: Tensor of shape [batch_size, N, d].
    - model: The model instance with a dyn_n_step method.
    - dataset: The dataset instance containing xi and f.
    - n: The number of steps to simulate.

    Returns:
    - max_overlap_xi: Tensor of max overlap values with xi.
    - max_overlap_f: Tensor of max overlap values with f.
    - final_overlap: Tensor of final overlap values with input_vectors.
    """
    x_new = model.dyn_n_step(input_vectors, n)  #[B,n,N,d]
    overlap_max_n_xi, max_overlap_xi, _,overlap_argmax = max_overlap(x_new, dataset.xi.to(model.J.device))
    if features == True:
        overlap_max_n_f, max_overlap_f, _, _ = max_overlap(x_new, dataset.f.to(model.J.device))
    else:
        overlap_max_n_f = np.zeros_like(overlap_max_n_xi)
        max_overlap_f = np.zeros_like(max_overlap_xi)

    input_overlap_n = torch.einsum('bnid,bid->bni', x_new, input_vectors).mean(dim=-1)  #[b,n]
    max_input_overlap = torch.max(input_overlap_n, dim=1)[0].cpu().numpy()

    return overlap_max_n_xi, overlap_max_n_f, max_overlap_xi, max_overlap_f, input_overlap_n.cpu().numpy(), max_input_overlap, overlap_argmax

def converge_input_vector_compute_overlap(input_data, model, init_overlap, n):
    input_vectors = start_overlap_binary(input_data, init_overlap)

    input_vectors = model.normalize_x(input_vectors)
    input_vectors = input_vectors.to(model.J.device)
    x_new = model.dyn_n_step(input_vectors, n)  #[B, n, N,i]
    overlap = torch.einsum('bnid,bid->bni', x_new, input_data).mean(dim=-1)  #[b,n]
    return overlap

def basins_of_attraction_inp_vectors(input_data, init_overlaps_array, model, n):
    max_overlap_inp_vectors_list = []

    # Loop over each init_overlap value
    for init_overlap in init_overlaps_array:
        overlaps = converge_input_vector_compute_overlap(input_data, model, init_overlap, n)[:,-1]
        max_overlap_inp_vectors_list.append(overlaps.detach().cpu())

    max_overlap_inp_vectors_array = np.array(max_overlap_inp_vectors_list)
    return max_overlap_inp_vectors_array

def compute_validation_loss(model, dataloader, device, init_overlap, n):
    model.eval() 
    vali_loss = 0.0
    counter = 0

    with torch.no_grad():  
        for batch_element in dataloader:
            counter += 1
            inp_data = batch_element.to(device)
            input_vectors = start_overlap_binary(inp_data, init_overlap)
            input_vectors = model.normalize_x(input_vectors)
            x_new = model.dyn_n_step(input_vectors, n)
            overlaps = torch.einsum('bnid,bid->bni', x_new, inp_data).mean(dim=-1)
            final_overlaps = overlaps[:, -1]
            max_input_overlap, _ = torch.max(overlaps, dim=-1)
            vloss = final_overlaps.mean().cpu().numpy()
            vali_loss += vloss

    if counter != 0:
        vali_loss = vali_loss / counter

    return vali_loss