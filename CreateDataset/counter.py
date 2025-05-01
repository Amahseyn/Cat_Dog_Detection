import os

def count_aug_wav_files(root_dir):
    count = 0
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.mp3') and 'aug' in file.lower():
                count += 1
    return count

# Example usage
directory_path = '/home/mio/Documents/code/Cat_Dog_Detection/NAYA_DATA_AUG1X'  # Change this to your directory path
total_aug_wav_files = count_aug_wav_files(directory_path)
print(f"Total .wav files with 'aug' in the name: {total_aug_wav_files}")