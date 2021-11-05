import os
import numpy as np
import torch
from EinsumNetwork import Graph, EinsumNetwork
import datasets
import utils
import time



device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
print(device)
fashion_mnist = False

# get data
if fashion_mnist:
    train_x, train_labels, test_x, test_labels = datasets.load_fashion_mnist()
    model_dir = '../models/einet/demo_fashion_mnist/'
    samples_dir = '../samples/demo_fashion_mnist/'
else:
    model_dir = '../models/einet/demo_mnist/'
    samples_dir = '../samples/demo_mnist/'
    train_x, train_labels, test_x, test_labels = datasets.load_mnist()

utils.mkdir_p(model_dir)
utils.mkdir_p(samples_dir)


exponential_family = EinsumNetwork.CategoricalArray
# exponential_family = EinsumNetwork.BinomialArray
# exponential_family = EinsumNetwork.NormalArray

classes = None # [2, 3, 5, 7]

structure = 'binary-trees'
K = 10
depth = 4
num_repetitions = 20

num_epochs = 200
batch_size = 1000
online_em_frequency = 1
online_em_stepsize = 0.05

exponential_family_args = None
if exponential_family == EinsumNetwork.BinomialArray:
    exponential_family_args = {'N': 255}
if exponential_family == EinsumNetwork.CategoricalArray:
    exponential_family_args = {'K': 2}
if exponential_family == EinsumNetwork.NormalArray:
    exponential_family_args = {'min_var': 1e-6, 'max_var': 0.1}


if not exponential_family != EinsumNetwork.NormalArray:
    train_x /= 255.
    test_x /= 255.
    train_x -= .5
    test_x -= .5

if exponential_family == EinsumNetwork.CategoricalArray:
    train_x /= 128
    test_x /= 128

# validation split
valid_x = train_x[-10000:, :]
train_x = train_x[:-10000, :]
valid_labels = train_labels[-10000:]
train_labels = train_labels[:-10000]

# pick the selected classes
if classes is not None:
    train_x = train_x[np.any(np.stack([train_labels == c for c in classes], 1), 1), :]
    valid_x = valid_x[np.any(np.stack([valid_labels == c for c in classes], 1), 1), :]
    test_x = test_x[np.any(np.stack([test_labels == c for c in classes], 1), 1), :]

train_x = torch.from_numpy(train_x).to(torch.device(device))
valid_x = torch.from_numpy(valid_x).to(torch.device(device))
test_x = torch.from_numpy(test_x).to(torch.device(device))


print("Data size ", train_x.shape, " ", valid_x.shape, " ", test_x.shape, " ")

####################################
# Make EinsumNetwork
####################################
graph = Graph.random_binary_trees(num_var=train_x.shape[1], depth=depth, num_repetitions=num_repetitions)
args = EinsumNetwork.Args(
        num_var=train_x.shape[1],
        num_dims=1,
        num_classes=1,
        num_sums=K,
        num_input_distributions=K,
        exponential_family=exponential_family,
        exponential_family_args=exponential_family_args,
        online_em_frequency=online_em_frequency,
        online_em_stepsize=online_em_stepsize)

einet = EinsumNetwork.EinsumNetwork(graph, args)
einet.initialize()
einet.to(device)
print(einet)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Einet Params ", count_parameters(einet))
##########################################
# Train                                 
##########################################

train_N = train_x.shape[0]
valid_N = valid_x.shape[0]
test_N = test_x.shape[0]


start_train_time = time.perf_counter()
for epoch_count in range(num_epochs):    
    cur_batch_time =  time.perf_counter()
    
    ##### Train
    einet.train()
    idx_batches = torch.randperm(train_N, device=device).split(batch_size)
    total_ll = 0.0
    for idx in idx_batches:
        batch_x = train_x[idx, :]
        outputs = einet.forward(batch_x)
        ll_sample = EinsumNetwork.log_likelihoods(outputs)
        log_likelihood = ll_sample.sum()
        log_likelihood.backward()

        einet.em_process_batch()
        total_ll += log_likelihood.detach().item()

    einet.em_update()
    end_train_time = time.perf_counter()
    ##### evaluate
    einet.eval()
    train_ll = EinsumNetwork.eval_loglikelihood_batched(einet, train_x, batch_size=batch_size)
    valid_ll = EinsumNetwork.eval_loglikelihood_batched(einet, valid_x, batch_size=batch_size)
    test_ll = EinsumNetwork.eval_loglikelihood_batched(einet, test_x, batch_size=batch_size)
    eval_end_time = time.perf_counter()
    print("[{}/{}; {:.2f}s; {:.2f}s]   train LL {:.4f}; valid LL {:.4f}; test LL {:.4f}".format(
        epoch_count,
        num_epochs,
        eval_end_time - end_train_time,
        end_train_time - cur_batch_time,
        train_ll / train_N,
        valid_ll / valid_N,
        test_ll / test_N))

    #####################
    # draw some samples #
    #####################
    if epoch_count % 20 == 0:
        samples = einet.sample(num_samples=25).cpu().numpy()
        samples = samples.reshape((-1, 28, 28))
        utils.save_image_stack(samples, 5, 5, os.path.join(samples_dir, "samples_{}.png".format(epoch_count)), margin_gray_val=0.)


####################
# save and re-load #
####################

# evaluate log-likelihoods
einet.eval()
train_ll_before = EinsumNetwork.eval_loglikelihood_batched(einet, train_x, batch_size=batch_size)
valid_ll_before = EinsumNetwork.eval_loglikelihood_batched(einet, valid_x, batch_size=batch_size)
test_ll_before = EinsumNetwork.eval_loglikelihood_batched(einet, test_x, batch_size=batch_size)

# save model
graph_file = os.path.join(model_dir, "einet.pc")
Graph.write_gpickle(graph, graph_file)
print("Saved PC graph to {}".format(graph_file))
model_file = os.path.join(model_dir, "einet.mdl")
torch.save(einet, model_file)
print("Saved model to {}".format(model_file))

del einet

# reload model
einet = torch.load(model_file)
print("Loaded model from {}".format(model_file))

# evaluate log-likelihoods on re-loaded model
train_ll = EinsumNetwork.eval_loglikelihood_batched(einet, train_x, batch_size=batch_size)
valid_ll = EinsumNetwork.eval_loglikelihood_batched(einet, valid_x, batch_size=batch_size)
test_ll = EinsumNetwork.eval_loglikelihood_batched(einet, test_x, batch_size=batch_size)
print()
print("Log-likelihoods before saving --- train LL {}   valid LL {}   test LL {}".format(
        train_ll_before / train_N,
        valid_ll_before / valid_N,
        test_ll_before / test_N))
print("Log-likelihoods after saving  --- train LL {}   valid LL {}   test LL {}".format(
        train_ll / train_N,
        valid_ll / valid_N,
        test_ll / test_N))
