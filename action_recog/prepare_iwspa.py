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

if __name__ == '__main__':
    data = load_jsonl('./datasets/iwspa-annot.jsonl')
    iwspa_train_txt = './datasets/iwspa-train.txt'
    iwspa_eval_txt = './datasets/iwspa-eval.txt'

    for item in data[:357]:
        body_text = item['text']
        labels = item['label']
        labels = [x for x in labels if x[-1]=='action instruction']
        if len(labels) > 0:
            # remove domain.com and <<link>>
            instruction_text = ' '.join([body_text[labels[i][0]:labels[i][1]] for i in range(len(labels))])

            # other text
            instruction_start = min([labels[i][0] for i in range(len(labels))])
            instruction_end = max([labels[i][1] for i in range(len(labels))])
            other_text = body_text[:instruction_start] + body_text[instruction_end:]
            other_text = other_text.split("Body: ")[1]
            other_text = other_text.replace('domain.com', '')
            other_text = other_text.replace('<<link>>', '')
            other_text = other_text.replace('\n', '')
            other_text_list = [x for x in other_text.split('.') if len(x)>3]
            other_text_list = other_text_list[-10:-5] # random 5 sentences

            with open(iwspa_train_txt, 'a+') as f:
                f.write(f'1\t{instruction_text}\n')
                for other_txt in other_text_list:
                    f.write(f'0\t{other_txt}\n')

    for item in data[357:]:
        body_text = item['text']
        labels = item['label']
        labels = [x for x in labels if x[-1] == 'action instruction']
        if len(labels) > 0:
            # remove domain.com and <<link>>
            instruction_text = ' '.join([body_text[labels[i][0]:labels[i][1]] for i in range(len(labels))])

            # other text
            instruction_start = min([labels[i][0] for i in range(len(labels))])
            instruction_end = max([labels[i][1] for i in range(len(labels))])
            other_text = body_text[:instruction_start] + body_text[instruction_end:]
            other_text = other_text.split("Body: ")[1]
            other_text = other_text.replace('domain.com', '')
            other_text = other_text.replace('<<link>>', '')
            other_text = other_text.replace('\n', '')
            other_text_list = [x for x in other_text.split('.') if len(x)>3]
            other_text_list = other_text_list[-10:-5] # random 5 sentences

            with open(iwspa_eval_txt, 'a+') as f:
                f.write(f'1\t{instruction_text}\n')
                for other_txt in other_text_list:
                    f.write(f'0\t{other_txt}\n')

