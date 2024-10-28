import re
from langdetect import detect
import lib.baselines.dfence.utils as utils
import torch
'''
This file contains a list of functions to preprocess, extract embeddings for text component from the email.

'''

def lang_detect(text_email):
    '''
    This function detects the language of an email. 
    '''
    try:
        return (detect(text_email))
    except KeyboardInterrupt:
        print('KeyboardInterrupt: lang_detect')
        exit()
    except:
        return ''

def re_compile():   
    list_re_patterns = []
    list_re_patterns.append((re.compile(r"\\n"), "\n"))
    list_re_patterns.append((re.compile(r"\\t"), " "))
    list_re_patterns.append((re.compile("=\\\n"), ""))
    list_re_patterns.append((re.compile("\n> "), "\n"))
    list_re_patterns.append((re.compile("\n>\n"), "\n"))
    list_re_patterns.append((re.compile("<mailto:"), "<"))
    list_re_patterns.append((re.compile(r"(n)+(\\')+(t)"), " not "))
    list_re_patterns.append((re.compile("\n(On )(.*?)(wrote:)", re.DOTALL), " \n"))
    list_re_patterns.append((re.compile("\n(<from>)(.*?)(<subject>)", re.DOTALL), " \n"))
    list_re_patterns.append((re.compile(r'http[s]?:\/\/(?:[a-zA-Z]|[0-9]|[\/\-?=_:;@.&+]|[!*,]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', re.IGNORECASE|re.DOTALL), "<url> "))
    list_re_patterns.append((re.compile(r"\\\n"), " "))    
    list_re_patterns.append((re.compile(r"\\r"), " "))
    
    #list_re_patterns.append((re.compile(r"[^a-zA-Z0-9@._\- ]"), " ")) # remain email only (i.e., having @ in non-email format)
    # Email Type 1: Email with CGI Parameter    
    list_re_patterns.append((re.compile(r"(<)+[^@ ]+@[^@ ]+(\.)+[\w-]+[\W]+[^> ]+(>)", re.IGNORECASE), "<email> ")) 
    # Email Type 2: Email with angle boundary
    list_re_patterns.append((re.compile(r"(<)+[^@ ]+@[^@ ]+(\.)+[\w-]+(>)", re.IGNORECASE), "<email> ")) # make email as token
    # Email Type 3: Email with any boundary (e.g., space)
    list_re_patterns.append((re.compile(r"[^@ ]+@[^@ ]+(\.)+[\w-]+[\W]", re.IGNORECASE), "<email> ")) # make email as token
    # Email Type 3: Email only with alphanumeric characters
    list_re_patterns.append((re.compile(r'([\w-]+@([\w-]+\.)+[\w-]+[\W])', re.IGNORECASE), "<email> "))
    list_re_patterns.append((re.compile(r"(<)+[^@ ]+(\.)+[\w-]+(>)", re.IGNORECASE), " <file> "))  # RE 18
    list_re_patterns.append((re.compile("<<"), "<"))
    list_re_patterns.append((re.compile(">>"), ">"))
    list_re_patterns.append((re.compile("\W>"), " "))
    list_re_patterns.append((re.compile("<\W"), " "))    
    list_re_patterns.append((re.compile(r"[^a-zA-Z\<\> ]"), " ")) # Remove non-alphanumeric except token boundaries.        
    return list_re_patterns

def text_cleaner(text): # Function from https://medium.com/text-classification-algorithms/text-classification-algorithms-a-survey-a215b7ab7e2d
    rules = [
        #{r'>\s+': u'>'},  # remove spaces after a tag opens or closes
        {r'\s+': u' '},  # replace consecutive spaces
        {r'\s*<br\s*/?>\s*': u'\n'},  # newline after a <br>
        {r'</(div)\s*>\s*': u'\n'},  # newline after </p> and </div> and <h1/>...
        {r'</(p|h\d)\s*>\s*': u'\n\n'},  # newline after </p> and </div> and <h1/>...
        {r'<head>.*<\s*(/head|body)[^>]*>': u''},  # remove <head> to </head>
        {r'<a\s+href="([^"]+)"[^>]*>.*</a>': ' <url> '},  # show links instead of texts
        #{r'[ \t]*<[^<]*?/?>': u''},  # remove remaining tags
        {r'^\s+': u''}  # remove spaces at the beginning
    ]

    for rule in rules:
        for (k, v) in rule.items():
            regex = re.compile(k)
            text = regex.sub(v, text)
        text = text.rstrip() # rstrip(): Remove white space at the end of the string.
    return text.lower()

def remove_header_group(str_tgt):
    # Header processing before Regex set.
    re_lf = re.compile(r"\\n")
    re_tab = re.compile(r"\\t")
    str_to_process = re.sub(re_lf, "\n", str_tgt)
    str_to_process = re.sub(re_tab, " ", str_to_process)
    lines = str_to_process.split('\n')
    bEndofHeader = True #changed to true
    str_body_wo_header = ''
    str_prev_line = ''
    for line in lines:
        if str_prev_line == '' and ':' not in line and line != '':
            bEndofHeader = True            

        if bEndofHeader:
            str_body_wo_header += line + '\n'

        str_prev_line = line

    return str_body_wo_header


def single_regex(re_pattern, tgt_str, str_to_process):
    return re.sub(re_pattern, tgt_str, str_to_process)
     
def multi_before_lang(text_data, list_re_patterns):
    #global g_cur_work_task
    

    str_to_process = text_data.split('StartOfEnvelope')[0]
    # Heuristic Header removal    
    str_to_process = remove_header_group(str_to_process)    
    #print(str_to_process)
    str_to_process = " ".join([s for s in str_to_process.split(" ")[:200] if len(s) < 30]) # Only Words shorter than 30 char    
    if str_to_process.startswith('b\''):
        str_to_process = str_to_process[2:]
    
    list_re_patterns = re_compile() 
    str_after_process = ''
    utils.g_cur_work_task = 'REGEX'
    for re_pattern, tgt_str in list_re_patterns:        
        str_after_process = single_regex(re_pattern, tgt_str, str_to_process)
        while str_after_process != str_to_process:
            str_to_process = str_after_process            
            str_after_process = single_regex(re_pattern, tgt_str, str_to_process)        
            
    utils.g_cur_work_task = 'text_cleaner'
    str_to_process = text_cleaner(str_after_process)

    if str_to_process.startswith('b '):
        str_to_process = str_to_process[2:]

    if str_to_process.startswith('>'):
        str_to_process = str_to_process[1:]
    return str_to_process


def preprocessing_text(text_data):
#     print('Preprocessing')      
    list_re_patterns = re_compile()

    # Text Preprocessing
    # print("Before REGEX")
    try:
        preproc_text = multi_before_lang(text_data, list_re_patterns)
    except KeyboardInterrupt:
        print('KeyboardInterrupt: multi_before_lang')
        exit()
   
    return preproc_text


def embeddings(tokenizer, model, text, device='cuda'):
    '''
    This function tokenizes and creates an embedding from the model provided. It then averages the features
    to get 1 representation per email. The function allows specification of the device (e.g., 'cuda' or 'cpu').
    '''
    # Encode the text into input IDs, ensuring to add special tokens
    input_ids = tokenizer.encode(text.lower(), add_special_tokens=True, return_tensors="pt", max_length=512, truncation=True)

    # Move the tensor to the same device
    input_ids = input_ids.to(device)

    # Ensuring the model is in eval mode
    model.eval()

    # Perform the forward pass and get the last hidden states
    with torch.no_grad():  # Disable gradient calculations for inference
        outputs = model(input_ids)
        last_hidden_states = outputs.last_hidden_state

    # Calculate the average across the sequence length dimension (dim=1)
    avg_feature = torch.mean(last_hidden_states, dim=1)

    # Move the result back to CPU if necessary and convert to numpy array
    avg_feature = avg_feature.squeeze().to('cpu').numpy()
    return list(avg_feature)

def create_text_feat(preprocessed_text, tokenizer, model, numoffeatures, words_range = 1000):
    ''' 
    This function creates the bert embedding. The default model is bert base. 
    If the default model changes, the number of parameters will need to be changed too.  
    Input is a string of preprocessed text and output the bert features
    '''

    if preprocessed_text !='':
        arr_text_features = embeddings(tokenizer, model, preprocessed_text.lower()[0:words_range])
        textlang = lang_detect(preprocessed_text)
    else:
        arr_text_features = [-99.0] * numoffeatures
        textlang = ''

    arr_text_features.append(textlang)
    
    return arr_text_features

