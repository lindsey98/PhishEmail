import re
import pandas as pd
import time

'''
This file contains a list of functions to extract the headers information from the emails

'''

def fieldstoextract(permheaderpath):
    headersdf = pd.read_csv(permheaderpath)
    headersdf.columns = ['fieldname','type']
    header_mail = headersdf[headersdf['type'] == 'mail']
    header_mime = headersdf[headersdf['type'] == 'MIME']
    return(header_mail['fieldname'].tolist(), header_mime['fieldname'].tolist())

def get_same_member_ratio(str_tgt, list_denominator):
    if len(list_denominator) == 0:
        return -1.0

    int_denominator = len(list_denominator)
    int_numerator = list_denominator.count(str_tgt)
    return (int_numerator / int_denominator)

def IsHexStr(str_tgt):
    return re.fullmatch(r'[0-9a-fA-F]+', str_tgt) is not None

g_list_charsets = ['asmo-708', 'dos-720', 'iso-8859-6', 'x-mac-arabic', 'windows-1256', 'ibm775', 'iso-8859-4', 'windows-1257',
                    'ibm852', 'iso-8859-2', 'x-mac-ce', 'windows-1250', 'euc-cn', 'gb2312', 'hz-gb-2312', 'x-mac-chinesesimp',
                    'big5', 'x-chinese-cns', 'x-chinese-eten', 'x-mac-chinesetrad', 'cp866', 'iso-8859-5', 'koi8-r', 'koi8-u',
                    'x-mac-cyrillic', 'windows-1251',  'x-europa', 'x-ia5-german', 'ibm737', 'iso-8859-7', 'x-mac-greek', 'windows-1253',
                    'ibm869',  'dos-862', 'iso-8859-8-i',  'iso-8859-8', 'x-mac-hebrew', 'windows-1255',  'x-ebcdic-arabic',
                    'x-ebcdic-cyrillicrussian', 'x-ebcdic-cyrillicserbianbulgarian', 'x-ebcdic-denmarknorway', 'x-ebcdic-denmarknorway-euro',
                    'x-ebcdic-finlandsweden',  'x-ebcdic-finlandsweden-euro', 'x-ebcdic-finlandsweden-euro',  'x-ebcdic-france-euro',
                    'x-ebcdic-germany', 'x-ebcdic-germany-euro',  'x-ebcdic-greekmodern','x-ebcdic-greek', 'x-ebcdic-hebrew',
                    'x-ebcdic-icelandic', 'x-ebcdic-icelandic-euro', 'x-ebcdic-international-euro', 'x-ebcdic-italy',
                    'x-ebcdic-italy-euro', 'x-ebcdic-japaneseandkana', 'x-ebcdic-japaneseandjapaneselatin', 'x-ebcdic-japaneseanduscanada',
                    'x-ebcdic-japanesekatakana', 'x-ebcdic-koreanandkoreanextended',  'x-ebcdic-koreanextended', 'cp870',
                    'x-ebcdic-simplifiedchinese', 'x-ebcdic-spain', 'x-ebcdic-spain-euro', 'x-ebcdic-thai', 'x-ebcdic-traditionalchinese',
                    'cp1026', 'x-ebcdic-turkish', 'x-ebcdic-uk', 'x-ebcdic-uk-euro', 'ebcdic-cp-us', 'x-ebcdic-cp-us-euro',
                    'ibm861',  'x-mac-icelandic', 'x-iscii-as', 'x-iscii-be', 'x-iscii-de', 'x-iscii-gu', 'x-iscii-ka',
                    'x-iscii-ma',  'x-iscii-or', 'x-iscii-pa', 'x-iscii-ta',  'x-iscii-te', 'euc-jp',  'x-euc-jp',
                    'iso-2022-jp', 'csiso2022jp', 'x-mac-japanese', 'shift_jis',  'ks_c_5601-1987', 'euc-kr', 'iso-2022-kr',
                    'johab', 'x-mac-korean', 'iso-8859-3', 'iso-8859-15', 'x-ia5-norwegian',  'ibm437', 'x-ia5-swedish',
                    'windows-874', 'ibm857', 'iso-8859-9', 'x-mac-turkish',  'windows-1254', 'unicode', 'unicodefffe', 'utf-7',
                    'utf-8', 'us-ascii', 'windows-1258', 'ibm850', 'x-ia5', 'iso-8859-1',  'macintosh', 'windows-1252','ansi_x3.4-1968']

def hdr_feature_extraction(header_data,dict_hdr_feature_use):
    mail_standardlist, mime_standardlist = fieldstoextract('./lib/baselines/dfence/emailpermheaders.csv')
    colnames_standards = [str.lower(w) for w in mail_standardlist]
    colnames_standards.extend([str.lower(w) for w in mime_standardlist])

    # Feature is Per Email.
    # Extracted per file, and stacked into list for batch writing on a DataFrame.    
    feat_plain_sec_cnt = 0
    feat_html_sec_cnt = 0
    feat_image_sec_cnt = 0
    feat_app_sec_cnt = 0
    feat_plain_html_ratio = 0.0
    dict_feat_hdr_cnt = {}
    
    feat_msg_id_boundary = 0 # numeric label by the type of boundary character.
    feat_msg_id_len = 0 # Length of Msg ID.

    feat_msg_id_is_hex = False # Whether all the character on MsgID (excluding domain name) are Hexa decimal
    feat_msg_id_is_num = False # Whether all the character on MsgID 
    feat_msg_id_has_dot = False # Whether 'DOT' is used in MsgID.
    feat_msg_id_has_spec = False # Whether any special characters other than 'DOT' are used in MsgID.

    feat_msg_id_domain_is_hex = False # Whether all the character on MsgID (excluding domain name) are Hexa decimal
    feat_msg_id_domain_is_num = False # Whether all the character on MsgID 
    feat_msg_id_domain_has_dot = False # Whether 'DOT' is used in MsgID.
    feat_msg_id_domain_has_spec = False # Whether any special characters other than 'DOT' are used in MsgID.

    feat_user_agent_exist = False # Is user-agent field exist # Need to see the 'Values'
    feat_x_mailer_exist = False # Is X-Mailer field exist # Need to see the 'Values'
    feat_mime_ver = 0.0 # MIME version (the first one.)
    feat_mime_ver_case_var = False # Is 'MIME-Version' header string case-variant, e.g., Mime-version
    feat_is_from_bracketed = False # Whether email in 'From' is with brackets
    feat_is_to_bracketed = False # Whether email in 'To' is with brackets
    feat_cont_type_char_set = 0 # Numeric label by the type of 'charset="??????"'.  No charset or unknown set as '-1'
    feat_num_unique_charset = 0 # Number of unique charsets across the content sections        
    
    feat_section_boundary_len = 0
    feat_section_boundary_startwith_equal = False
    feat_section_boundary_has_equal = False
    feat_section_boundary_has_underscore = False
    feat_section_boundary_is_hex = False
    feat_section_boundary_is_num = False
    feat_section_boundary_has_dot = False
    feat_section_boundary_has_spec = False        
    
    feat_to_rcpt_cnt = 0
    feat_cc_rcpt_cnt = 0
    feat_bcc_rcpt_cnt = 0
    
    list_from_domain = []
    list_sender_domain = []
    list_to_domain = []
    list_reply_to_domain = []
    list_in_reply_to_domain = []
    list_return_domain = []
    
    list_cc_domain = []
    list_bcc_domain = []
    list_plain_urls = []

    bhtml = False

    plain_body = ''
    html_body = ''        
    for part in header_data.walk():
        if part.get_content_type().startswith('text/html'):
            bhtml = True
            feat_html_sec_cnt += 1
        elif part.get_content_type().startswith('text/plain'):                
            feat_plain_sec_cnt += 1
        elif part.get_content_type().startswith('image'):
            feat_image_sec_cnt += 1
        elif part.get_content_type().startswith('application'):
            feat_app_sec_cnt += 1                

    if bhtml:
        feat_plain_html_ratio = feat_plain_sec_cnt/feat_html_sec_cnt
    else:
        feat_plain_html_ratio = -1

    before_feat_extract = time.perf_counter()
    if dict_hdr_feature_use['section_boundary']:
        str_section_boundary = header_data.get_boundary()
        if str_section_boundary == None:                
            feat_section_boundary_len = 0
            feat_section_boundary_startwith_equal = False
            feat_section_boundary_has_equal = False
            feat_section_boundary_has_underscore = False
            feat_section_boundary_is_hex = False
            feat_section_boundary_is_num = False
            feat_section_boundary_has_dot = False
            feat_section_boundary_has_spec = False
        else:
            feat_section_boundary_len = len(str_section_boundary)
            feat_section_boundary_startwith_equal = str_section_boundary.startswith('=')
            if feat_section_boundary_startwith_equal:
                str_section_boundary = str_section_boundary[2:] # Except the first two '--'.
                
            feat_section_boundary_has_equal = '=' in str_section_boundary 
            str_section_boundary = str_section_boundary.replace('=', '')
            feat_section_boundary_has_dot = '.' in str_section_boundary
            str_section_boundary = str_section_boundary.replace('.', '')
            feat_section_boundary_has_underscore = '_' in str_section_boundary
            str_section_boundary = str_section_boundary.replace('_', '')
            feat_section_boundary_has_spec = not str_section_boundary.isalnum()
            str_section_boundary = re.sub(r'\W+', '', str_section_boundary) # Only AlphaNumeric    
            feat_section_boundary_is_hex = IsHexStr(str_section_boundary)
            feat_section_boundary_is_num = str_section_boundary.isdigit()


    if dict_hdr_feature_use['charset']:
        # 2020.0825. New feature for 'charset'        
        str_first_content_charset = header_data.get_charsets()[0]
        str_content_charset = str.lower(str_first_content_charset) if str_first_content_charset is not None else 'nan' 
        # Get the charset number.
        if str_content_charset == 'nan':
            feat_cont_type_char_set = -1            
        else:
            try:
                feat_cont_type_char_set = g_list_charsets.index(str_content_charset)
            except ValueError:
                feat_cont_type_char_set = 999 # Unknown Charset is different (worse) to NO charset.

        feat_num_unique_charset = len(set(header_data.get_charsets()))  # Unique number of Charsets in this email.
    else:
        feat_num_unique_charset = -1
    

    if dict_hdr_feature_use['messageid']:
        for mail_key in header_data.keys():
            # Nessage-ID Features      
            if str.lower(mail_key) == 'message-id':
                str_msg_id = header_data[mail_key]
                feat_msg_id_len = len(str_msg_id)
                if feat_msg_id_len > 2:
                    char_left_boundary = str_msg_id[0]
                    char_right_boundary = str_msg_id[-1]
                else:
                    continue

                str_domain = ''
                str_first_domain = ''

                if char_left_boundary.isalnum() and char_right_boundary.isalnum():
                    # There is no boundary

                    str_id_without_boundary = str_msg_id.split('@')[0]
                    feat_msg_id_boundary = -1   # No boundary
                    if '@' in str_msg_id:
                        str_domain = str_msg_id.split('@')[1]
                        str_first_domain = str_domain.split('.')[0]
                else:
                    # Thre is boundary
                    str_id_without_boundary = str_msg_id[1:-1].split('@')[0]
                    feat_msg_id_boundary = ord(char_left_boundary)
                    if '@' in str_msg_id:
                        str_domain = str_msg_id[1:-1].split('@')[1]
                        str_first_domain = str_domain.split('.')[0]                   

                str_id_without_boundary = str_id_without_boundary.lower().replace(str_first_domain.lower(), '')

                feat_msg_id_has_dot = '.' in str_id_without_boundary
                feat_msg_id_has_spec = not str_id_without_boundary.replace('.', '').isalnum() # If True, there is no special char, except DOT
                str_id_without_boundary = re.sub(r'\W+', '', str_id_without_boundary) # Only AlphaNumeric    
                feat_msg_id_is_hex = IsHexStr(str_id_without_boundary) # [0-9a-fA-F]                                
                feat_msg_id_is_num = str_id_without_boundary.isdigit() # [0-9]

                feat_msg_id_domain_has_dot = '.' in str_domain
                feat_msg_id_domain_has_spec = not str_domain.replace('.', '').isalnum() # If True, there is no special char, except DOT
                str_id_without_boundary = re.sub(r'\W+', '', str_id_without_boundary) # Only AlphaNumeric    
                feat_msg_id_domain_is_hex = IsHexStr(str_domain)
                feat_msg_id_domain_is_num = str_id_without_boundary.isdigit()

            elif str.lower(mail_key) == 'user-agent':
                feat_user_agent_exist = True       
            elif str.lower(mail_key) == 'x-mailer':
                feat_x_mailer_exist = True        
            elif str.lower(mail_key) == 'mime-version':
                try:
                    feat_mime_ver = float(header_data[mail_key])

                except ValueError:
                    feat_mime_ver = -1

                    pass
                except TypeError:
                    feat_mime_ver = -1

                    pass

                feat_mime_ver_case_var = (mail_key != 'MIME-Version')


    set_mail_addr_hdr = set({'from', 'to', 'sender', 'in-reply-to', 'reply-to', 'return-path', 'cc', 'bcc'})
    if dict_hdr_feature_use['mailaddr']:
        # Standard Header Counter        
        for std_hdr in colnames_standards:
            dict_feat_hdr_cnt[std_hdr] = 0        
            for mail_key in header_data.keys():
                if str.lower(mail_key) == std_hdr:                
                    dict_feat_hdr_cnt[std_hdr] += 1 
                    
                    # 2020.09.19
                    if std_hdr in set_mail_addr_hdr:
                        addr_body = header_data[mail_key]
                                               
                        list_addr_body = re.findall(r".*?<(.*?)>", addr_body, flags=re.DOTALL)
                        if len(list_addr_body) == 0 and ',' in addr_body:
                            list_addr_body = addr_body.split(',')

                        if len(list_addr_body) == 0 and '@' in addr_body:
                            
                            addr_domain = str.lower(addr_body.split('@')[1]).replace('\n','')
                            if std_hdr == 'from':
                                list_from_domain.append(addr_domain)
                                feat_is_from_bracketed = False
                            elif std_hdr == 'sender':
                                list_sender_domain.append(addr_domain)
                            elif std_hdr == 'to':
                                list_to_domain.append(addr_domain)
                                feat_to_rcpt_cnt += 1
                                feat_is_to_bracketed = False
                            elif std_hdr == 'reply-to':
                                list_reply_to_domain.append(addr_domain)  
                            elif std_hdr == 'in-reply-to':
                                list_in_reply_to_domain.append(addr_domain)
                            elif std_hdr == 'return-path':
                                list_return_domain.append(addr_domain)
                            elif std_hdr == 'cc':
                                feat_cc_rcpt_cnt += 1
                                list_cc_domain.append(addr_domain)
                            elif std_hdr == 'bcc':
                                feat_bcc_rcpt_cnt += 1
                                list_bcc_domain.append(addr_domain)

                        elif '@' in addr_body:
                            
                            for addr_str in list_addr_body:
                                if '@' in addr_str:
                                    addr_domain = str.lower(addr_str.split('@')[1]).replace('\n','')
                                    if std_hdr == 'from':
                                        list_from_domain.append(addr_domain)
                                        feat_is_from_bracketed = True
                                    elif std_hdr == 'sender':
                                        list_sender_domain.append(addr_domain)
                                    elif std_hdr == 'to':
                                        list_to_domain.append(addr_domain)
                                        feat_to_rcpt_cnt += 1
                                        feat_is_to_bracketed = True
                                    elif std_hdr == 'reply-to':
                                        list_reply_to_domain.append(addr_domain)      
                                    elif std_hdr == 'in-reply-to':
                                        list_in_reply_to_domain.append(addr_domain)
                                    elif std_hdr == 'return-path':
                                        list_return_domain.append(addr_domain)
                                    elif std_hdr == 'cc':
                                        feat_cc_rcpt_cnt += 1
                                        list_cc_domain.append(addr_domain)
                                    elif std_hdr == 'bcc':
                                        feat_bcc_rcpt_cnt += 1
                                        list_bcc_domain.append(addr_domain)

    else:
        for std_hdr in colnames_standards:
            dict_feat_hdr_cnt[std_hdr] = 0

    # ------------------------------------------------------
    # Push all the features into lists
    # ------------------------------------------------------
    list_header_feature = []    # Feature list to return.
    
    # Basic Feature set
    list_header_feature.append(feat_plain_sec_cnt)
    list_header_feature.append(feat_html_sec_cnt)
    list_header_feature.append(feat_image_sec_cnt)
    list_header_feature.append(feat_app_sec_cnt)
    list_header_feature.append(feat_plain_html_ratio)   
    
    # List in order.
    std_hdr_cnt = 0
    for std_hdr in colnames_standards:
        if dict_feat_hdr_cnt[std_hdr] > 0:
            std_hdr_cnt += 1 # the number of standard headers included!

    list_header_feature.append(std_hdr_cnt)   
    
    list_header_feature.append(feat_user_agent_exist)
    list_header_feature.append(feat_x_mailer_exist)
    list_header_feature.append(feat_mime_ver)
    list_header_feature.append(feat_mime_ver_case_var)
    list_header_feature.append(feat_is_from_bracketed)
    list_header_feature.append(feat_is_to_bracketed)
    
    
    if dict_hdr_feature_use['section_boundary']:
        list_header_feature.append(feat_section_boundary_len)
        list_header_feature.append(feat_section_boundary_startwith_equal)
        list_header_feature.append(feat_section_boundary_has_equal)
        list_header_feature.append(feat_section_boundary_has_underscore)
        list_header_feature.append(feat_section_boundary_is_hex)
        list_header_feature.append(feat_section_boundary_is_num)
        list_header_feature.append(feat_section_boundary_has_dot)
        list_header_feature.append(feat_section_boundary_has_spec)
        
        
    if dict_hdr_feature_use['charset']:
        list_header_feature.append(feat_cont_type_char_set)
        list_header_feature.append(feat_num_unique_charset)    
        
    if dict_hdr_feature_use['messageid']:
        list_header_feature.append(feat_msg_id_boundary)
        list_header_feature.append(feat_msg_id_len)
        list_header_feature.append(feat_msg_id_is_hex)
        list_header_feature.append(feat_msg_id_is_num)
        list_header_feature.append(feat_msg_id_has_dot)
        list_header_feature.append(feat_msg_id_has_spec)
        list_header_feature.append(feat_msg_id_domain_is_hex)
        list_header_feature.append(feat_msg_id_domain_is_num)
        list_header_feature.append(feat_msg_id_domain_has_dot)
        list_header_feature.append(feat_msg_id_domain_has_spec)
        
    if dict_hdr_feature_use['mailaddr']:
        list_header_feature.append(dict_feat_hdr_cnt['from'])
        list_header_feature.append(dict_feat_hdr_cnt['sender'])
        list_header_feature.append(dict_feat_hdr_cnt['to'])
        list_header_feature.append(dict_feat_hdr_cnt['reply-to'])
        list_header_feature.append(dict_feat_hdr_cnt['in-reply-to'])
        list_header_feature.append(dict_feat_hdr_cnt['received'])
        list_header_feature.append(feat_to_rcpt_cnt)
        list_header_feature.append(feat_cc_rcpt_cnt)
        list_header_feature.append(feat_bcc_rcpt_cnt)    
        
        if len(list_from_domain) > 0:
            str_from_domain = list_from_domain[0]
            list_header_feature.append(get_same_member_ratio(str_from_domain, list_to_domain))
            list_header_feature.append(get_same_member_ratio(str_from_domain, list_reply_to_domain))
            list_header_feature.append(get_same_member_ratio(str_from_domain, list_return_domain))            
            list_header_feature.append(get_same_member_ratio(str_from_domain, list_cc_domain))
            list_header_feature.append(get_same_member_ratio(str_from_domain, list_bcc_domain))
        else:
            list_header_feature.append(-1)
            list_header_feature.append(-1)
            list_header_feature.append(-1)
            list_header_feature.append(-1)
            list_header_feature.append(-1)

        list_all_unique_domain = list_from_domain
        list_all_unique_domain.extend(list_from_domain)
        list_all_unique_domain.extend(list_to_domain)        
        list_all_unique_domain.extend(list_sender_domain)
        list_all_unique_domain.extend(list_reply_to_domain)
        list_all_unique_domain.extend(list_return_domain)               

        all_unique_domain = len(set(list_all_unique_domain))
        if all_unique_domain == 0:
            list_header_feature.append(0)
            list_header_feature.append(0)
            list_header_feature.append(0)
            list_header_feature.append(0)
            list_header_feature.append(0)
        else:
            list_header_feature.append(len(set(list_from_domain))/all_unique_domain)
            list_header_feature.append(len(set(list_to_domain))/all_unique_domain)
            list_header_feature.append(len(set(list_sender_domain))/all_unique_domain)
            list_header_feature.append(len(set(list_reply_to_domain))/all_unique_domain)
            list_header_feature.append(len(set(list_return_domain))/all_unique_domain)            

    return list_header_feature

def get_column_list(dict_hdr_feature_use):
    list_column_header = []
    
    list_column_header.append('plain_sec_cnt')
    list_column_header.append('html_sec_cnt')
    list_column_header.append('image_sec_cnt')
    list_column_header.append('app_sec_cnt')
    list_column_header.append('plain_html_ratio')
    list_column_header.append('std_hdr_cnt')
    
    list_column_header.append('user_agent_exist')
    list_column_header.append('x_mailer_exist')
    list_column_header.append('mime_ver')
    list_column_header.append('mime_ver_case_var')
    list_column_header.append('is_from_bracketed')
    list_column_header.append('is_to_bracketed')    
    
    if dict_hdr_feature_use['section_boundary']:
        list_column_header.append('section_boundary_len')
        list_column_header.append('section_boundary_startwith_equal')
        list_column_header.append('section_boundary_has_equal')
        list_column_header.append('section_boundary_has_underscore')
        list_column_header.append('section_boundary_is_hex')
        list_column_header.append('section_boundary_is_num')
        list_column_header.append('section_boundary_has_dot')
        list_column_header.append('section_boundary_has_spec')
        
        
    if dict_hdr_feature_use['charset']:
        list_column_header.append('cont_type_char_set')
        list_column_header.append('num_unique_charset')
        
    if dict_hdr_feature_use['messageid']:
        list_column_header.append('msg_id_boundary')
        list_column_header.append('msg_id_len')
        list_column_header.append('msg_id_is_hex')
        list_column_header.append('msg_id_is_num')
        list_column_header.append('msg_id_has_dot')
        list_column_header.append('msg_id_has_spec')
        list_column_header.append('msg_id_domain_is_hex')
        list_column_header.append('msg_id_domain_is_num')
        list_column_header.append('msg_id_domain_has_dot')
        list_column_header.append('msg_id_domain_has_spec')
        
    if dict_hdr_feature_use['mailaddr']:
        list_column_header.append('num_from')
        list_column_header.append('num_sender')
        list_column_header.append('num_to')
        list_column_header.append('num_reply_to')
        list_column_header.append('num_in_reply_to')
        list_column_header.append('num_received')
        list_column_header.append('num_to_rcpt')
        list_column_header.append('num_cc_rcpt')
        list_column_header.append('num_bcc_rcpt')
        
        list_column_header.append('ratio_same_domain_from_to')
        list_column_header.append('ratio_same_domain_from_replyto')
        list_column_header.append('ratio_same_domain_from_return')
        list_column_header.append('ratio_same_domain_from_cc')
        list_column_header.append('ratio_same_domain_from_bcc')
     
        list_column_header.append('ratio_unique_domain_from')
        list_column_header.append('ratio_unique_domain_to')
        list_column_header.append('ratio_unique_domain_sender')
        list_column_header.append('ratio_unique_domain_replyto')
        list_column_header.append('ratio_unique_domain_return')
    
    return list_column_header
    

def extract_features(header_data, dict_feature_use):    
    list_hdr_feat = hdr_feature_extraction(header_data,dict_feature_use)
    return list_hdr_feat





