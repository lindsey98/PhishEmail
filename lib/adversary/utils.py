import random

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

