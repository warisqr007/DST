import argparse
from tqdm import tqdm
from pathlib import Path

from audio_processor import AudioProcessor, ModelManager

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--checkpoint_path', default='experiments/base')
    parser.add_argument('-c', '--config', default='experiments/base/config.json')
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--output', required=True)

    args = parser.parse_args()

    model_manager = ModelManager(args.config, args.checkpoint_path)
    audio_processor = AudioProcessor(model_manager)

    with open(f"{args.output}/tokens.txt", "w") as f:
        f.write(f"{args.input}\n")
        wav_files = list(Path(args.input).rglob('*.wav'))
        for wav_file in tqdm(wav_files):
            tokens = audio_processor.generate_wav2tokens(wav_file)
            f.write(f"{wav_file.relative_to(args.input)}|{' '.join(map(str, tokens))}\n")

    print(f"Tokens saved to {args.output}/tokens.txt")
