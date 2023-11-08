import os
import subprocess

# Get the current directory (run_sub_tfts)
current_directory = os.path.dirname(os.path.realpath(__file__))

# Log start in the current directory of the script
with open(os.path.join(current_directory, 'start.txt'), 'w') as file:
    file.write('run_sub_tfts started')

# Define the list of sub-tft directories
sub_tft_directories = [
    'base_tft',
    'batch_size_high_tft',
    'batch_size_low_tft',
    'drop_out_high_tft',
    'drop_out_low_tft',
    'gradient_clipping_high_tft',
    'gradient_clipping_low_tft',
    'learning_rate_high_tft',
    'learning_rate_low_tft',
]
# Create iterator
it = 0

# Loop through the sub-tft directories
for sub_tft_directory in sub_tft_directories:
    # Log start in the current directory of the script
    start_log_file = os.path.join(current_directory, f'start_{sub_tft_directory}.txt')
    with open(start_log_file, 'w') as file:
        file.write(f'{sub_tft_directory} started ({it})')

    # Run the program in the current sub-tft directory
    subprocess.run(['python', os.path.join(current_directory, '..', sub_tft_directory, f'{sub_tft_directory}.py')])

    # Log end in the current directory of the script
    end_log_file = os.path.join(current_directory, f'end_{sub_tft_directory}.txt')
    with open(end_log_file, 'w') as file:
        file.write(f'{sub_tft_directory} ended ({it})')

    # Increase iterator
    it += 1

# Log end in the current directory of the script
with open(os.path.join(current_directory, 'end.txt'), 'w') as file:
    file.write('run_sub_tfts ended')