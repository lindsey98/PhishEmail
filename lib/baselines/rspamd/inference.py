from lib.data.Dataset import EmailDataset
import time
import os
import subprocess
import pandas as pd
import json
import re
'''
Main function
'''

def parse_symbols(output_line):
    """
    Parse a single 'Symbol:' line to extract the symbol name and score.

    Args:
        output_line (str): A line from rspamc output starting with 'Symbol:'.

    Returns:
        tuple: (symbol_name (str), symbol_score (float))
    """
    symbol_regex = re.compile(r'^Symbol:\s+(\S+)\s+\(([-+]?\d*\.\d+|\d+)\)')
    match = symbol_regex.match(output_line)
    if match:
        symbol_name = match.group(1)
        symbol_score = float(match.group(2))
        return symbol_name, symbol_score
    return None, None

def parse_rspamc_output(output):
    """
    Parse the output from rspamc into a dictionary, focusing on important entries.

    Args:
        output (str): Raw output from rspamc.

    Returns:
        dict: Parsed key-value pairs from rspamc output.
    """
    email_data = {}
    symbols = {}
    lines = output.splitlines()
    for line in lines:
        if line.startswith('Results for file:'):
            # Example: results_for_file: stdin (0.34 seconds)
            try:
                parts = line.split(': ', 1)
                if len(parts) == 2:
                    email_data['runtime'] = float(parts[1].split('(')[1].split()[0])
            except (IndexError, ValueError):
                email_data['runtime'] = 0
            continue

        if line.startswith('[metric'):
            continue  # Not essential, skip or handle if multiple metrics are used

        if ': ' in line:
            key, value = line.split(': ', 1)
            key = key.strip().lower()
            value = value.strip()

            if key == 'spam':
                email_data['spam'] = 1 if value.lower() == 'true' else 0

            elif key == 'score':
                # Example: '9.70 / 15.00'
                try:
                    current_score, required_score = value.split('/')
                    email_data['score'] = float(current_score.strip())
                except ValueError:
                    email_data['score'] = 0

        if line.startswith('Symbol:'):
            # Use the parse_symbols function to handle symbols
            symbol_name, symbol_score = parse_symbols(line)
            if symbol_name and symbol_score != 0: # only keep those decisive scores
                symbols[symbol_name] = symbol_score

    email_data['symbols'] = symbols
    return email_data

def test(email_dir):
    """test with labeled files, it can be: 1 mal and 1 benign, 1 benign only or 1 mal only
    """
    test_dataset = EmailDataset(email_dir)
    email_files = test_dataset.file_list
    data = []

    for file_path in email_files:
        with open(file_path, 'rb') as f:
            email_content = f.read()

        # Execute rspamc
        # If sudo is required without password, prefix with 'sudo'
        # Example: ['sudo', 'rspamc']
        # If sudo is not required, use ['rspamc']
        cmd = ['rspamc']  # Replace with ['sudo', 'rspamc'] if necessary

        # Run the command
        result = subprocess.run(
            cmd,
            input=email_content,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30  # Optional: set a timeout for the command
        )

        if result.returncode != 0:
            print(f"Error processing {file_path}: {result.stderr.decode('utf-8')}")
            data.append({'spam': 0, 'score': 0, 'symbols': {}, 'runtime': 0, 'file': file_path})
            continue

        # Decode the output. Parse the rspamc output
        output = result.stdout.decode('utf-8')
        email_data = parse_rspamc_output(output)
        email_data['file'] = file_path

        data.append(email_data)

    # Create DataFrame
    df = pd.DataFrame(data)
    # reject = 15.0;: Emails with a score ≥15.0 will be rejected.
    # add_header = 10.0;: Emails with a score ≥10.0 will have a header added.
    # greylist = 5.0;: Emails with a score ≥5.0 will be greylisted.
    pred_class, pred_score, pred_metadata, pred_runtime = df['spam'], df['score'], df['symbols'], df['runtime']
    pred_class = pred_class.tolist()
    pred_score = pred_score.tolist()
    pred_metadata = pred_metadata.tolist()
    pred_runtime = pred_runtime.tolist()

    return pred_class, pred_score, pred_runtime, pred_metadata

