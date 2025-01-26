from collections import Counter
from fairseq.data import Dictionary

# Input file containing token frequencies
input_files = [
    "/mnt/nvme-data1/waris/PSI-TAMU/DST/data/american_arctic/tokens.txt",
    "/mnt/nvme-data1/waris/PSI-TAMU/DST/data/american_libri/tokens.txt",
    "/mnt/nvme-data1/waris/PSI-TAMU/DST/data/spanish/tokens.txt",
]

output_file = "/mnt/nvme-data1/waris/PSI-TAMU/DST/data/dict.txt"

# Counter to store token frequencies
token_counter = Counter()

for input_file in input_files:
    # Read and process the input file
    with open(input_file, "r") as f:
        for line in f:
            if "|" in line:
                _, token_str = line.strip().split("|")
                tokens = map(int, token_str.split())
                token_counter.update(tokens)

for i in range(512):
    if i not in token_counter:
        token_counter[i] = 0

# Save the token frequencies to the output file
with open(output_file, "w") as f:
    for token, freq in token_counter.most_common():
        f.write(f"{token} {freq}\n")

print(f"Token frequencies saved to {output_file}")

# Add helper tokens to the dictionary
def add_helper_tokens(dictionary):
    # Add helper tokens if not already present
    special_tokens = {
        '<pad>': dictionary.pad_index,
        '<unk>': dictionary.unk_index,
        '<mask>': dictionary.add_symbol('<mask>'),
        '<bos>': dictionary.add_symbol('<bos>'),
        '<eos>': dictionary.add_symbol('<eos>'),
    }

    for token, idx in special_tokens.items():
        if dictionary.index(token) == idx:
            print(f"Token {token} already exists at index {idx}.")
        else:
            dictionary.add_symbol(token, idx)

    print("Helper tokens added.")
    return dictionary

# Save the modified dictionary
def save_custom_dict(dictionary, file_path):
    dictionary.save(file_path)
    print(f"Dictionary saved to {file_path}.")


custom_dict = Dictionary.load(output_file)
updated_dict = add_helper_tokens(custom_dict)
save_path = "/mnt/nvme-data1/waris/PSI-TAMU/DST/data/dict_ht.txt"
save_custom_dict(updated_dict, save_path)