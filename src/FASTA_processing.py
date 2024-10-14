import random

def read_fasta(file_path):
    """Reads a FASTA file and returns a list of sequences."""
    sequences = []
    with open(file_path, 'r') as file:
        seq = []
        for line in file:
            if line.startswith('>'):
                if seq:
                    sequences.append(''.join(seq))
                seq = []
            else:
                seq.append(line.strip())
        if seq:
            sequences.append(''.join(seq))  # Append the last sequence
    return sequences

def write_txt(sequences, file_path):
    """Writes sequences to a TXT file with <|endoftext|> headers."""
    with open(file_path, 'w') as file:
        for seq in sequences:
            file.write(f"<|endoftext|>\n{seq}\n")

def split_data(sequences, split_ratio=0.8):
    """Splits sequences into training and validation sets."""
    random.shuffle(sequences)
    split_point = int(len(sequences) * split_ratio)
    return sequences[:split_point], sequences[split_point:]

def process_fasta(input_file, train_file, val_file, split_ratio=0.8):
    """Processes the FASTA file to split into training and validation sets and writes to .txt files."""
    # Read the sequences from the input file
    sequences = read_fasta(input_file)
    
    # Split into training and validation sets
    train_sequences, val_sequences = split_data(sequences, split_ratio)
    
    # Write the training and validation .txt files
    write_txt(train_sequences, train_file)
    write_txt(val_sequences, val_file)

# Example usage:
input_fasta = '/Users/kosio/Data/PolG/POLG_refseq_protein.fasta'
train_txt = '/Users/kosio/Data/PolG/train.txt'
val_txt = '/Users/kosio/Data/PolG/val.txt'
process_fasta(input_fasta, train_txt, val_txt, split_ratio=0.8)
