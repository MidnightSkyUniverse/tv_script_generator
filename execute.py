import logging

logging.basicConfig(filename='output.log', encoding='utf-8', level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger()


"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import helper
import problem_unittests as tests

int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import torch
import numpy as np

# Check for a GPU
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('No GPU found. Please use a GPU to train your neural network.')

from torch.utils.data import TensorDataset, DataLoader

def pad_features(words, sequence_length):
    ''' Return features of review_ints, where each review is padded with 0's 
        or truncated to the input seq_length.
    '''
    
    # getting the correct rows x cols shape
    features = np.zeros((len(words), sequence_length), dtype=int)
    target = np.zeros(len(words),dtype=int)
    
    #only full batches
    size = len(words)//sequence_length
    words = words[:size*sequence_length]
    
    for idx in range(0,len(words)-sequence_length): #range(0, len(words), sequence_length):
        word_seq = words[idx: idx+sequence_length]
        #
        try:
            features[idx] = np.array(word_seq)
            try:
                target[idx] = words[idx+sequence_length]
            except: 
                target[idx] = int(words[0])
        except: 
            print(f"idx: {idx} || ws: {word_seq}")
            print(f"{features[idx]}")
            print(f"{np.array(word_seq)}")   
    
    
    return features, target


def batch_data(words, sequence_length, batch_size):
    """
    Batch the neural network data using DataLoader
    :param words: The word ids of the TV scripts
    :param sequence_length: The sequence length of each batch
    :param batch_size: The size of each batch; the number of sequences in a batch
    :return: DataLoader with batched data
    """
    # TODO: Implement function
       
    feature_tensors, target_tensors = pad_features(words, sequence_length)
    
    data = TensorDataset(torch.from_numpy(feature_tensors), torch.from_numpy(target_tensors))
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    
    # return a dataloader
    return data_loader

# there is no test for this function, but you are encouraged to create
# print statements and tests of your own


# test dataloader

test_text = range(50)
t_loader = batch_data(test_text, sequence_length=5, batch_size=10)

data_iter = iter(t_loader)
sample_x, sample_y = data_iter.next()

print(sample_x.shape)
print(sample_x)
print()
print(sample_y.shape)
print(sample_y)


import torch.nn as nn

class RNN(nn.Module):
    
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5):
        """
        Initialize the PyTorch RNN Module
        :param vocab_size: The number of input dimensions of the neural network (the size of the vocabulary)
        :param output_size: The number of output dimensions of the neural network
        :param embedding_dim: The size of embeddings, should you choose to use them        
        :param hidden_dim: The size of the hidden layer outputs
        :param dropout: dropout to add in between LSTM/GRU layers
        """
        super(RNN, self).__init__()
        # TODO: Implement function
        
        # set class variables
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        # define model layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout = dropout, batch_first=True)
        
        # dropout layer - not recommended 
        #self.dropout = nn.Dropout(0.3)
        
        # linera and sigmoid
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sigmoid = nn.Sigmoid()
    
    
    def forward(self, nn_input, hidden):
        """
        Forward propagation of the neural network
        :param nn_input: The input to the neural network
        :param hidden: The hidden state        
        :return: Two Tensors, the output of the neural network and the latest hidden state
        """
        # TODO: Implement function   
        embeds = self.embedding(nn_input)
        lstm_out, hidden = self.lstm(embeds, hidden)
        
        # get last batch
        lstm_out = lstm_out[:, -1]
        
        #out = self.dropout(lstm_out)
        
        # Stack up LSTM putput using view
        out = lstm_out.contiguous().view(-1, self.hidden_dim)
    
        out = self.fc(out)
     
        sig_out = self.sigmoid(out)
        
        # return one batch of output word scores and the hidden state
        return sig_out, hidden
    
    
    def init_hidden(self, batch_size):
        '''
        Initialize the hidden state of an LSTM/GRU
        :param batch_size: The batch_size of the hidden state
        :return: hidden state of dims (n_layers, batch_size, hidden_dim)
        '''
        # Implement function
        
        # initialize hidden state with zero weights, and move to GPU if available
        weight = next(self.parameters()).data
        
        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        
        return hidden

# Test: define and print network
#rnn = RNN(vocab_size = 100, embedding_dim=10, output_size = 2, hidden_dim = 64, n_layers=1)
#print(rnn)    
    
"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""

tests.test_rnn(RNN, train_on_gpu)


def forward_back_prop(rnn, optimizer, criterion, inp, target, hidden):
    """
    Forward and backward propagation on the neural network
    :param rnn: The PyTorch Module that holds the neural network
    :param optimizer: The PyTorch optimizer for the neural network
    :param criterion: The PyTorch loss function
    :param inp: A batch of input to the neural network
    :param target: The target output for the batch of input
    :return: The loss and the latest hidden state Tensor
    """
    
    # TODO: Implement Function
    
    # move data to GPU, if available
    if(train_on_gpu):
        rnn.cuda()
        inp, target = inp.cuda(), target.cuda()
        
    # Creating new variables for the hidden state, otherwise
    # we'd backprop through the entire training history    
    hidden = tuple([each.data for each in hidden])
        
    # perform backpropagation and optimization
    rnn.zero_grad()
    
    # get the output from the model
    out, hidden = rnn(inp, hidden)
    
    # calculate the loss and perform backprop
    loss = criterion(out.squeeze(), target)
    loss.backward()#retain_graph=True)
    
    optimizer.step()
    
    # return the loss over a batch and the hidden state produced by our model
    return loss.item(), hidden

# Note that these tests aren't completely extensive.
# they are here to act as general checks on the expected outputs of your functions
"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_forward_back_prop(RNN, forward_back_prop, train_on_gpu)


"""
DON'T MODIFY ANYTHING IN THIS CELL
"""

def train_rnn(rnn, batch_size, optimizer, criterion, n_epochs, show_every_n_batches=100):
    batch_losses = []
    
    rnn.train()
    valid_loss_min = np.Inf

    print("Training for %d epoch(s)..." % n_epochs)
    for epoch_i in range(1, n_epochs + 1):
        
        # initialize hidden state
        hidden = rnn.init_hidden(batch_size)
        
        for batch_i, (inputs, labels) in enumerate(train_loader, 1):
            
            # make sure you iterate over completely full batches, only
            n_batches = len(train_loader.dataset)//batch_size
            if(batch_i > n_batches):
                break
            
            # forward, back prop
            loss, hidden = forward_back_prop(rnn, optimizer, criterion, inputs, labels, hidden)          
            # record loss
            batch_losses.append(loss)

            # printing loss stats
            if batch_i % show_every_n_batches == 0:
                logger.info('Epoch: {:>4}/{:<4}  Loss: {}\n'.format(epoch_i, n_epochs, np.average(batch_losses)))
                batch_losses = []
        
        if loss < valid_loss_min:
            logger.info('Validation loss decreased ({:.4f} --> {:.4f}).  Saving model ...'.format(valid_loss_min,
                    loss))
            helper.save_model('./save/_under_training_rnn', rnn)
            valid_loss_min = loss
            

    # returns a trained rnn
    return rnn

# Data params
# Sequence Length
sequence_length = 15  # of words in a sequence
# Batch Size
batch_size =64

# data loader - do not change
train_loader = batch_data(int_text, sequence_length, batch_size)

# Training parameters
# Number of Epochs
num_epochs = 100
# Learning Rate
learning_rate = 0.003

# Model parameters
# Vocab size
vocab_size = len(vocab_to_int)
# Output size
output_size = len(vocab_to_int)
# Embedding Dimension
embedding_dim = 50
# Hidden Dimension
hidden_dim = 128
# Number of RNN Layers
n_layers = 1

# Show stats for every n number of batches
show_every_n_batches = 2000



"""
DON'T MODIFY ANYTHING IN THIS CELL
"""

# create model and move to gpu if available
rnn = RNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5)
if train_on_gpu:
    rnn.cuda()

# defining loss and optimization functions for training
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# training the model
trained_rnn = train_rnn(rnn, batch_size, optimizer, criterion, num_epochs, show_every_n_batches)

# saving the trained model
helper.save_model('./save/trained_rnn', trained_rnn)
logger.info('Model Trained and Saved')
