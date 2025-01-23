import os
import subprocess
from pathlib import Path

def run_asr(asr_script, asr_decode_log):
    """
    Runs the ASR inference script.
    """
    print("Running ASR Task...")
    result = subprocess.run(["bash", asr_script], capture_output=True, text=True)
    if result.returncode != 0:
        print("ASR Task failed!")
        print(result.stderr)
        exit(1)
    print("ASR Task completed.")
    return asr_decode_log

def read_asr_output(asr_decode_log):
    """
    Reads the ASR output file and returns a list of tuples (key, text).
    """
    asr_results = []
    with open(asr_decode_log, 'r') as f:
        for line in f:
            key, text = line.strip().split("\t")
            asr_results.append((key, text))
    return asr_results

def generate_prompts(asr_results, prompts_file):
    """
    Generates prompts from ASR output and saves them to a file.
    """
    print("Generating prompts for Audiocaps...")
    with open(prompts_file, 'w') as f:
        for key, text in asr_results:
            prompt = f"The input audio contains: {text}"
            f.write(f"{key}\t{prompt}\n")
    print("Prompts saved to:", prompts_file)

def run_audiocaps(audiocaps_script):
    """
    Runs the Audiocaps inference script.
    """
    print("Running Audiocaps Task...")
    result = subprocess.run(["bash", audiocaps_script], capture_output=True, text=True)
    if result.returncode != 0:
        print("Audiocaps Task failed!")
        print(result.stderr)
        exit(1)
    print("Audiocaps Task completed.")

def read_audiocaps_output(audiocaps_decode_log):
    """
    Reads the Audiocaps output file and returns a list of tuples (key, caption).
    """
    audiocaps_results = []
    with open(audiocaps_decode_log, 'r') as f:
        for line in f:
            key, caption = line.strip().split("\t")
            audiocaps_results.append((key, caption))
    return audiocaps_results

def integrate_results(asr_results, audiocaps_results, final_output_file):
    """
    Integrates ASR and Audiocaps results and saves them to a file.
    """
    print("Integrating results...")
    with open(final_output_file, 'w') as f:
        for (key, asr_text), (_, audiocaps_text) in zip(asr_results, audiocaps_results):
            f.write(f"{key}\tASR: {asr_text}\tAudiocaps: {audiocaps_text}\n")
    print("Final results saved to:", final_output_file)

def main():
    # Define paths
    asr_script = "/path/to/Wav_inference_custom.sh"
    audiocaps_script = "/path/to/inference_audiocaps_bs.sh"
    asr_decode_log = "/path/to/asr_decode_log.txt"
    prompts_file = "/path/to/audiocaps_prompts.txt"
    audiocaps_decode_log = "/path/to/audiocaps_decode_log.txt"
    final_output_file = "/path/to/final_output.txt"
    
    # Ensure directories exist
    Path(asr_decode_log).parent.mkdir(parents=True, exist_ok=True)
    Path(prompts_file).parent.mkdir(parents=True, exist_ok=True)
    Path(audiocaps_decode_log).parent.mkdir(parents=True, exist_ok=True)
    Path(final_output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Run ASR task
    run_asr(asr_script, asr_decode_log)
    
    # Step 2: Read ASR output
    asr_results = read_asr_output(asr_decode_log)
    
    # Step 3: Generate prompts for Audiocaps
    generate_prompts(asr_results, prompts_file)
    
    # Step 4: Run Audiocaps task
    # Update the Audiocaps script to use `prompts_file` as input data
    run_audiocaps(audiocaps_script)
    
    # Step 5: Read Audiocaps output
    audiocaps_results = read_audiocaps_output(audiocaps_decode_log)
    
    # Step 6: Integrate results
    integrate_results(asr_results, audiocaps_results, final_output_file)

if __name__ == "__main__":
    main()
