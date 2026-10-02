import time
import argparse
import os
from pathlib import Path

def process_audio(rmvpe, audio_path, output_path, device, hop_length, threshold):
    """Process an audio file in 10-second chunks and save the results."""
    import numpy as np
    import librosa
    from tqdm import tqdm

    # Load the audio file
    audio, sr = librosa.load(str(audio_path), sr=None)
    if not len(audio):
        raise ValueError(f"Audio file is empty: {audio_path}")
    audio_duration = len(audio) / sr
    chunk_size = 10 * sr
    # pad to make the audio length to be multiple of hop_length
    audio = np.pad(audio, (0, (-len(audio)) % chunk_size), mode='constant')
    
    # Calculate chunk size in samples (10 seconds * sample rate)
    total_chunks = int(np.round(len(audio) / chunk_size))
    
    # Initialize arrays to store results
    all_f0 = []
    total_infer_time = 0
    
    # Process each chunk
    for i in tqdm(range(total_chunks)):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(audio))
        chunk = audio[start_idx:end_idx]
        
        # Process the chunk
        t = time.time()
        f0_chunk = rmvpe.infer_from_audio(chunk, sr, device=device, thred=threshold, use_viterbi=True)
        chunk_infer_time = time.time() - t
        total_infer_time += chunk_infer_time
        
        # Append results
        all_f0.extend(f0_chunk)
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # remove all 0 in the f0
    all_f0 = np.array(all_f0)
    all_f0 = all_f0[all_f0 != 0]

    # convert all_f0 to a list
    all_f0 = all_f0.tolist()
    
    # Save the results
    with open(output_path, 'w') as f:
        for f0 in all_f0:
            f.write(f'{f0:.2f}\n')
    
    return total_infer_time, audio_duration  # Exclude padding from the duration

def main():
    parser = argparse.ArgumentParser(description="Extract vocal pitch values from a directory")
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_path", default="model.pt")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    input_dir, output_dir = args.input_dir, args.output_dir
    if not input_dir.is_dir():
        parser.error(f"Input directory does not exist: {input_dir}")
    wav_files = list(input_dir.rglob('*.Vocals.mp3'))
    print(f'Found {len(wav_files)} vocal files to process')
    if not wav_files:
        print('No matching audio files; nothing to process.')
        return

    import torch
    from tqdm import tqdm
    from extract_pitch_values_from_audio.src import RMVPE
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    print('Loading model...')
    rmvpe = RMVPE(args.model_path, hop_length=160)

    total_time = 0
    total_audio_duration = 0
    
    # Process each WAV file
    for wav_path in tqdm(wav_files, desc="Processing files"):
        # Calculate relative path to maintain directory structure
        rel_path = wav_path.relative_to(input_dir)
        # Create output path with .txt extension
        output_path = output_dir / str(rel_path).replace('.Vocals.mp3', '.txt')
        
        try:
            infer_time, audio_duration = process_audio(
                rmvpe, wav_path, output_path, device, 
                160, 0.03
            )
            total_time += infer_time
            total_audio_duration += audio_duration
            
            tqdm.write(f'Processed {wav_path.name}')
            tqdm.write(f'Time: {infer_time:.2f}s, RTF: {infer_time/audio_duration:.2f}')
            
        except Exception as e:
            tqdm.write(f'Error processing {wav_path}: {str(e)}')
            continue
    
    print('\nProcessing complete!')
    print(f'Total processing time: {total_time:.2f}s')
    if total_audio_duration > 0:
        print(f'Average RTF: {total_time/total_audio_duration:.2f}')
    else:
        print('No audio files were processed successfully.')

if __name__ == '__main__':
    main()
