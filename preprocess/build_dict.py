from collections import Counter

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
