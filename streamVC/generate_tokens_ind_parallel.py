import argparse
from tqdm import tqdm
from pathlib import Path

from audio_processor import AudioProcessor, ModelManager

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--checkpoint_path', default='experiments/base/bnf_vq_ft_512/g_00550000')
    parser.add_argument('-c', '--config', default='experiments/base/bnf_vq_ft_512/config.json')
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--output', required=True)

    args = parser.parse_args()

    model_manager = ModelManager(args.config, args.checkpoint_path)
    audio_processor = AudioProcessor(model_manager)

    with open(f"{args.output}/indian/tokens.txt", "w") as fi:
        with open(f"{args.output}/indian_native_parallel/tokens.txt", "w") as fip:
            fi.write(f"{args.input}\n")
            fip.write(f"{args.input}\n")
            speakers = list(Path(args.input).iterdir())
            for speaker in tqdm(speakers):
                native_fpath = speaker / "native"
                indian_fpath = speaker / "english"
                native_wav_files = list(native_fpath.rglob('*.wav'))
                for wav_file in native_wav_files:
                    indian_wav_file = indian_fpath / wav_file.relative_to(native_fpath)
                    # Indian tokens
                    tokens = audio_processor.generate_wav2tokens(indian_wav_file)
                    fi.write(f"{indian_wav_file.relative_to(args.input)}|{' '.join(map(str, tokens))}\n")

                    # Native tokens
                    tokens = audio_processor.generate_wav2tokens(wav_file)
                    fip.write(f"{wav_file.relative_to(args.input)}|{' '.join(map(str, tokens))}\n")

    print(f"Tokens saved to {args.output} directory")
