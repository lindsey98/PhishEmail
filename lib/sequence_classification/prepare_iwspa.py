import json

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def get_body_text(text):
    try:
        return text.split("Body: ")[1]
    except IndexError:
        return ""

def extract_non_labeled_text(body_text, labels):
    # Combine and sort the label ranges
    sorted_labels = sorted(labels, key=lambda x: x[0])

    # Initialize variables
    non_labeled_text = ''
    last_end_pos = 0

    # Iterate through sorted label ranges
    for start_pos, end_pos, _ in sorted_labels:
        # Add text from the last end position to the current start position
        if start_pos > last_end_pos:
            non_labeled_text += body_text[last_end_pos:start_pos] + ' '
        # Update the last end position
        last_end_pos = max(last_end_pos, end_pos)

    # Add any remaining text after the last label
    if last_end_pos < len(body_text):
        non_labeled_text += body_text[last_end_pos:]

    return non_labeled_text

def process_data(data, start_index, end_index, output_file, downsample=True):
    for item in data[start_index:end_index]:
        body_text = item['text']
        labels = item['label']
        motivations = [x for x in labels if x[-1] == 'motivation']
        instructions = [x for x in labels if x[-1] == 'action instruction']

        if len(motivations) > 0 and len(instructions) > 0:
            motivation_text = ' '.join([body_text[m[0]:m[1]] for m in motivations])
            instruction_text = ' '.join([body_text[i[0]:i[1]] for i in instructions])

            other_text = extract_non_labeled_text(body_text, labels)
            other_text = other_text.split("Body: ")[1]
            other_text = other_text.replace('domain.com', '').replace('<<link>>', '').replace('\n', '')
            other_text_list = [x for x in other_text.split('.') if len(x) > 3]
            if downsample:
                other_text_list = other_text_list[-10:-5]  # fixme: random 5 sentences

            with open(output_file, 'a+') as f:
                f.write(f'1\t{instruction_text}\n')
                for other_txt in other_text_list:
                    f.write(f'0\t{other_txt}\n')

def filter_duplicates(file_path):
    unique_texts = set()
    filtered_lines = []

    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into the marker and the text
            parts = line.split('\t', 1)
            if len(parts) == 2:
                marker, text = parts
                if text not in unique_texts:
                    unique_texts.add(text)
                    filtered_lines.append(line)

    return filtered_lines

if __name__ == '__main__':
    data = load_jsonl('./datasets/iwspa-annot.jsonl')
    iwspa_train_txt = './datasets/iwspa-train.txt'
    iwspa_filtered_train_txt = './datasets/iwspa-train-filtered.txt'
    iwspa_eval_txt = './datasets/iwspa-eval.txt'
    iwspa_filtered_eval_txt = './datasets/iwspa-eval-filtered.txt'

    # labeled_dict = {'motivation': 1, 'action instruction': 2}

    # train_eval_split_point = int(len(data)*0.8)
    # process_data(data, 0, train_eval_split_point, iwspa_train_txt, downsample=True)
    #
    # # Process evaluation data
    # process_data(data, train_eval_split_point, len(data), iwspa_eval_txt, downsample=False)
    #
    # # filter duplicates
    # filtered_content = filter_duplicates(iwspa_train_txt)
    # for line in filtered_content:
    #     with open(iwspa_filtered_train_txt, 'a+') as f:
    #         f.write(line)

    filtered_content = filter_duplicates(iwspa_eval_txt)
    # You can now write filtered_content to a new file or process it as needed
    for line in filtered_content:
        with open(iwspa_filtered_eval_txt, 'a+') as f:
            f.write(line)
