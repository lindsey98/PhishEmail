import random
import re

def repeat_char(text):
    """Add a repeating character."""
    if len(text) < 3:
        return text
    idx = random.randint(1, len(text) - 2)
    char = text[idx]
    return text[:idx] + char + char + text[idx+1:]

def delete_char(text):
    """Delete a character."""
    if len(text) < 3:
        return text
    idx = random.randint(1, len(text) - 2)
    return text[:idx] + text[idx+1:]

def switch_chars(text):
    """Switch two adjacent characters."""
    if len(text) < 3:
        return text
    idx = random.randint(1, len(text) - 2)
    return text[:idx] + text[idx+1] + text[idx] + text[idx+2:]

def replace_with_similar(text):
    """Replace a character with a similar-looking Latin/Unicode character."""
    similar_chars = {
        '-': '˗', '9': '৭', '8': 'Ȣ', '7': '𝟕', '6': 'б', '5': 'Ƽ', '4': 'Ꮞ', '3': 'Ʒ', '2': 'ᒿ', '1': 'l', '0': 'O',
        "'": '`', 'a': 'ɑ', 'b': 'Ь', 'c': 'ϲ', 'd': 'ԁ', 'e': 'е', 'f': '𝚏', 'g': 'ɡ', 'h': 'հ', 'i': 'і', 'j': 'ϳ',
        'k': '𝒌', 'l': 'ⅼ', 'm': 'ｍ', 'n': 'ո', 'o': 'о', 'p': 'р', 'q': 'ԛ', 'r': 'ⲅ', 's': 'ѕ', 't': '𝚝', 'u': 'ս',
        'v': 'ѵ', 'w': 'ԝ', 'x': '×', 'y': 'у', 'z': 'ᴢ'
    }
    if len(text) < 3:
        return text
    idx = random.randint(1, len(text) - 1)
    char = text[idx]
    return text[:idx] + similar_chars.get(char, char) + text[idx+1:]


def match_case(word, pattern):
    """Match the case of the pattern in the original word."""
    if pattern.islower():
        return word.lower()
    elif pattern.isupper():
        return word.upper()
    elif pattern[0].isupper():
        return word.capitalize()
    else:
        return word

def find_all_token_indices(all_tokens, entity_tokens):
    entity_len = len(entity_tokens)
    indices = []

    for i in range(len(all_tokens) - entity_len + 1):
        if [token.lower() for token in all_tokens[i:i + entity_len]] == entity_tokens:
            indices.append((i, i + entity_len))
    return indices

def to_sentence_case(text):
    """
    Converts the given text into sentence case.
    Capitalizes the first word and lowers the rest, except for proper nouns or acronyms.
    """
    # Tokenize based on whitespace or specific parse tokens
    tokens = re.split(r'(\s+|\(|\)|,|\.|\;)', text)  # split but keep delimiters

    # Capitalize first VB (or other sentence-starting token)
    capitalized = False
    new_tokens = []
    for token in tokens:
        if not capitalized and token.isalpha():
            new_tokens.append(token.capitalize())  # capitalize first alphabetic word
            capitalized = True
        else:
            new_tokens.append(token.lower())  # lowercase the rest
    return ''.join(new_tokens)


def tokenize_and_map(text, annotations, tokenizer, label_to_id):
    # Tokenize text using a custom tokenizer function
    tokens = tokenizer.tokenize(text)

    # Initialize tags with "O"
    tags = [label_to_id["O"]] * len(tokens)

    for annot in annotations:
        entity = annot['text']
        entity_cls = annot['labels'][0]
        entity_tokens = tokenizer.tokenize(entity)

        all_indices = find_all_token_indices(tokens, entity_tokens)
        for start_index, end_index in all_indices:
            # Check if the range already has a tag other than "O"
            if all(tag == label_to_id["O"] for tag in tags[start_index:end_index]) or entity_cls == "relation":
                # Label the found entity
                tags[start_index] = label_to_id["B-" + entity_cls]
                for i in range(start_index + 1, end_index):
                    tags[i] = label_to_id["I-" + entity_cls]

        if len(all_indices) == 0:
            continue
    return tokens, tags
