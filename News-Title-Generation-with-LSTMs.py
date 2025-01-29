import torch
from datasets import load_dataset
from collections import Counter
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np


# Set the seed
seed = 42
torch.manual_seed(seed)
# Probably, this below must be changed if you work with a M1/M2/M3 Mac
torch.cuda.manual_seed(seed) # for CUDA
torch.backends.cudnn.deterministic = True # for CUDNN
torch.backends.benchmark = False # if True, causes cuDNN to benchmark multiple convolution algorithms and select the fastest.


if __name__ == "__main__":
    '''
    Data
    '''
    ds = load_dataset("heegyu/news-category-dataset")
    print(ds['train'])
    # Filter for "POLITICS" category
    politics_titles = [item['headline'] for item in ds['train'] if item['category'] == "POLITICS"]
    print(f"Number of POLITICS titles: {len(politics_titles)}")  # Verify count = 35602


    # Tokenizing each title, convert to lowercase, and append <EOS>
    tokenized_titles = [title.lower().split(" ") + ["<EOS>"] for title in politics_titles]

    # Adding <EOS> at position 0 and <PAD> at position -1 for each title
    modified_titles = [["<EOS>"] + tokens[:-1] + ["<PAD>"] for tokens in tokenized_titles]

    # Building a vocabulary from the modified titles
    word_counts = Counter([word for tokens in modified_titles for word in tokens])

    # Creating dictionaries for word-to-integer and integer-to-word mapping
    word_to_int = {word: idx for idx, (word, _) in enumerate(word_counts.items())}
    int_to_word = {idx: word for word, idx in word_to_int.items()}

    # Display the size of the vocabulary
    vocab_size = len(word_to_int)

    # Getting the 5 most common words
    most_common_words = word_counts.most_common(5)

    # Output results
    print(f"Vocabulary size: {vocab_size}")
    print("Top 5 most common words:")
    print(most_common_words)

    # Question 5
    class NewsDataset(Dataset):
        def __init__(self, tokenized_sequences, word_to_int):
            self.data = [
                (
                    [word_to_int[word] for word in tokens[:-2]],  # Input: Exclude last two tokens (<EOS> and <PAD>)
                    [word_to_int[word] for word in tokens[1:-1]]  # Target: Exclude the first token and <PAD>
                )
                for tokens in tokenized_sequences
                if len(tokens) > 2  # Ensure there are enough tokens for input-target pair
            ]

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]


    # Creating an instance of the dataset
    news_dataset = NewsDataset(modified_titles, word_to_int)  # Pass modified_titles

    # Testing the dataset
    print(f"Number of sequences in the dataset: {len(news_dataset)}")
    example_input, example_target = news_dataset[0]
    print("Example input-target pair from the dataset:")
    print(f"Input: {example_input}")  # Indexes for all words except the last
    print(f"Target: {example_target}")  # Indexes for all words except the first


    # Defining the collate function
    def collate_fn(batch):
        # Separate inputs and targets
        inputs, targets = zip(*batch)

        # Converting to tensors
        inputs = [torch.tensor(seq) for seq in inputs]
        targets = [torch.tensor(seq) for seq in targets]

        # Padding sequences with the <PAD> token index
        inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=word_to_int["<PAD>"])
        targets_padded = pad_sequence(targets, batch_first=True, padding_value=word_to_int["<PAD>"])

        return inputs_padded, targets_padded

    # Creating the DataLoader
    batch_size = 32
    dataloader = DataLoader(news_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    '''
    Model
    '''

    class LSTMModel(nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout=0.0):
            super(LSTMModel, self).__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim)  # Embedding layer
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                                batch_first=True, dropout=dropout)  # LSTM layers
            self.fc = nn.Linear(hidden_dim, vocab_size)  # Fully connected layer
            self.dropout = nn.Dropout(dropout)  # Dropout for regularization
            self.hidden_dim = hidden_dim  # Save the hidden dimension

        def forward(self, x, hidden_state):
            x = self.embedding(x)  # Convert tokens to embeddings
            lstm_out, hidden_state = self.lstm(x, hidden_state)  # LSTM layers
            lstm_out = self.dropout(lstm_out)  # Apply dropout
            output = self.fc(lstm_out)  # Map hidden states to vocabulary logits
            return output, hidden_state

        def init_state(self, batch_size):  # Initialize the LSTM hidden and cell states.
            hidden_state = torch.zeros(1, batch_size, self.hidden_dim).to(device)
            cell_state = torch.zeros(1, batch_size, self.hidden_dim).to(device)
            return (hidden_state, cell_state)


    # Hyperparameters
    vocab_size = len(word_to_int)  # Vocabulary size
    embedding_dim = 150  # Embedding dimension
    hidden_dim = 1024  # Number of hidden units in LSTM
    num_layers = 1  # Number of stacked LSTM layers
    dropout = 0.0 # Dropout rate


    # Create the model instance
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda:0" if torch.cuda.is_available() else "cpu")
    model = LSTMModel(vocab_size, embedding_dim, hidden_dim, num_layers, dropout).to(device)


    # Function to randomly sample the next word based on the probability distribution
    def random_sample_next(predictions):
        probabilities = F.softmax(predictions, dim=0).detach().cpu().numpy()
        next_word = np.random.choice(len(probabilities), p=probabilities)
        return next_word


    # Function to pick the word with the highest probability
    def sample_argmax(predictions):
        next_word = torch.argmax(predictions).item()
        return next_word


    # Function to generate sentences using the sampling strategies
    def sample(prompt, model, word_to_int, int_to_word, max_length=20, sampling_fn=sample_argmax):
        model.eval()  # Set model to evaluation mode
        # Initialize the input tensor with the prompt
        input_tokens = [word_to_int.get(word, word_to_int["<PAD>"]) for word in prompt]
        input_tensor = torch.tensor([input_tokens]).to(device)

        # Initialize hidden and cell states
        hidden_state = model.init_state(batch_size=1)

        # Generate words
        generated_sentence = prompt[:]  # Start with the prompt
        for _ in range(max_length):
            # Get the model's predictions
            output, hidden_state = model(input_tensor, hidden_state)
            predictions = output[0, -1, :]  # Logits for the last token in the sequence

            # Sample the next word
            next_word_index = sampling_fn(predictions)
            next_word = int_to_word[next_word_index]

            # Stop if <EOS> is generated
            if next_word == "<EOS>":
                break

            # Append the generated word to the sentence
            generated_sentence.append(next_word)

            # Update the input tensor with the new word
            input_tensor = torch.tensor([[next_word_index]]).to(device)

        return generated_sentence


    # Test the sampling strategies
    prompt = ["the", "president", "wants"]

    # Generate 3 sentences using random sampling
    print("Random Sampling:")
    for _ in range(3):
        sentence = sample(prompt, model, word_to_int, int_to_word, sampling_fn=random_sample_next)
        print(" ".join(sentence))

    # Generate 3 sentences using greedy sampling
    print("\nGreedy Sampling:")
    for _ in range(3):
        sentence = sample(prompt, model, word_to_int, int_to_word, sampling_fn=sample_argmax)
        print(" ".join(sentence))


    '''
    Training
    '''
    # Define the loss function and optimizer
    loss_fn = nn.CrossEntropyLoss(ignore_index=word_to_int["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training parameters
    num_epochs = 12
    clip_value = 1.0
    batch_size = 32
    prompt = ["the", "president", "wants"]

    # Track loss and perplexity
    loss_values = []
    perplexity_values = []

    # Generate sentences at these specific points
    milestones = [0, num_epochs // 2, num_epochs - 1]
    generated_sentences = {}

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch_idx, (inputs_padded, targets_padded) in enumerate(dataloader):
            # Move data to the device
            inputs_padded = inputs_padded.to(device)
            targets_padded = targets_padded.to(device)

            # Initialize hidden state
            hidden_state = model.init_state(inputs_padded.size(0))

            # Forward pass
            outputs, _ = model(inputs_padded, hidden_state)

            # Reshape outputs and targets for loss calculation
            outputs = outputs.view(-1, vocab_size)
            targets = targets_padded.view(-1)

            # Compute loss
            loss = loss_fn(outputs, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

            # Update weights
            optimizer.step()

            # Accumulate loss
            total_loss += loss.item()

        # Average loss for the epoch
        avg_loss = total_loss / len(dataloader)
        loss_values.append(avg_loss)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        perplexity_values.append(perplexity)

        # Generate sentence at specific milestones
        if epoch in milestones:
            model.eval()
            generated_sentence = sample(prompt, model, word_to_int, int_to_word, sampling_fn=random_sample_next)
            generated_sentences[epoch] = " ".join(generated_sentence)

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}")

    # Plot Loss
    plt.figure(figsize=(8, 6))
    plt.plot(loss_values, label="Loss")
    plt.axhline(y=1.5, color='r', linestyle='--', label="Target Loss (1.5)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Over Epochs")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot Perplexity
    plt.figure(figsize=(8, 6))
    plt.plot(perplexity_values, label="Perplexity")
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.title("Perplexity Over Epochs")
    plt.legend()
    plt.grid()
    plt.show()

    # Print generated sentences at milestones
    for milestone, sentence in generated_sentences.items():
        if milestone == 0:
            position = "Beginning"
        elif milestone == num_epochs // 2:
            position = "Middle"
        else:
            position = "End"
        print(f"{position} of Training: {sentence}")

    # Updated hyperparameters for TBBTT
    num_epochs_tbbtt = 5
    hidden_dim_tbbtt = 2048  # Increased hidden size for LSTM

    # Redefine the model with updated LSTM size
    model = LSTMModel(vocab_size, embedding_dim, hidden_dim_tbbtt, num_layers).to(device)

    # Re-initialize optimizer for the updated model
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop for TBBTT
    tbbtt_loss_values = []
    tbbtt_perplexity_values = []
    chunk_size = 10  # Fixed chunk size for backpropagation

    for epoch in range(num_epochs_tbbtt):
        model.train()  # Setting the model to training mode
        total_loss_tbbtt = 0

        for batch_idx, (inputs_padded, targets_padded) in enumerate(dataloader):
            # Move data to the appropriate device
            inputs_padded = inputs_padded.to(device)
            targets_padded = targets_padded.to(device)

            # Initialize hidden state
            hidden_state = model.init_state(inputs_padded.size(0))

            # Process sequences in chunks
            for t in range(0, inputs_padded.size(1), chunk_size):
                # Define chunk of inputs and targets
                inputs_chunk = inputs_padded[:, t:t + chunk_size]
                targets_chunk = targets_padded[:, t:t + chunk_size]

                # Forward pass
                outputs, hidden_state = model(inputs_chunk, hidden_state)

                # Detach hidden state to prevent gradients from flowing beyond the chunk
                hidden_state = (hidden_state[0].detach(), hidden_state[1].detach())

                # Reshape outputs and targets for loss computation
                outputs = outputs.view(-1, vocab_size)
                targets_chunk = targets_chunk.reshape(-1)

                # Compute loss
                loss = loss_fn(outputs, targets_chunk)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

                optimizer.step()

                # Accumulate loss
                total_loss_tbbtt += loss.item()

        # Calculate average loss for the epoch
        avg_loss_tbbtt = total_loss_tbbtt / len(dataloader)
        tbbtt_loss_values.append(avg_loss_tbbtt)
        perplexity_tbbtt = torch.exp(torch.tensor(avg_loss_tbbtt)).item()
        tbbtt_perplexity_values.append(perplexity_tbbtt)

        # Generate sentences at specific checkpoints
        if epoch == 0 or epoch == num_epochs_tbbtt // 2 or epoch == num_epochs_tbbtt - 1:
            sampling_strategy = random_sample_next if epoch % 2 == 0 else sample_argmax
            generated_sentence = sample(prompt, model, word_to_int, int_to_word, sampling_fn=sampling_strategy)
            print(f"Generated Sentence (TBBTT Epoch {epoch + 1}): {' '.join(generated_sentence)}")

        print(
            f"TBBTT - Epoch {epoch + 1}/{num_epochs_tbbtt}, Loss: {avg_loss_tbbtt:.4f}, Perplexity: {perplexity_tbbtt:.4f}")

    # Plot Loss and Perplexity
    plt.figure()
    plt.plot(tbbtt_loss_values, label="Loss (TBBTT)")
    plt.axhline(y=1.5, color='r', linestyle='--', label="Target Loss (1.5)")
    plt.title("Loss Over Epochs (TBBTT)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(tbbtt_perplexity_values, label="Perplexity (TBBTT)")
    plt.title("Perplexity Over Epochs (TBBTT)")
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.legend()
    plt.show()

    '''
    Evaluation, part 2
    '''

    # Generate and report sentences after training
    print("\nEvaluation - Generating sentences after training")

    prompt = ["the", "president", "wants"]

    # Generate 3 sentences using random sampling
    print("\nRandom Sampling:")
    for _ in range(3):
        sentence = sample(prompt, model, word_to_int, int_to_word, sampling_fn=random_sample_next)
        print(" ".join(sentence))

    # Generate 3 sentences using greedy sampling
    print("\nGreedy Sampling:")
    for _ in range(3):
        sentence = sample(prompt, model, word_to_int, int_to_word, sampling_fn=sample_argmax)
        print(" ".join(sentence))



