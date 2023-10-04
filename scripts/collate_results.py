from tqdm import tqdm

def test_result_raw(result_file, action_check=True, link_consistency_check=True):
    total = 0
    ct = 0
    lines = [x.strip() for x in open(result_file).readlines()[1:]]
    pbar = tqdm(lines, leave=False)
    if action_check and link_consistency_check:
        positive_condition = "eval(requires_action) and (not eval(consistent_sender) or (not eval(consistent_link)))"
    elif action_check and (not link_consistency_check):
        positive_condition = "eval(requires_action) and (not eval(consistent_sender))"
    elif (not action_check) and link_consistency_check:
        positive_condition = "(not eval(consistent_sender) or (not eval(consistent_link)))"
    else:
        positive_condition = "not eval(consistent_sender)"

    for l in lines:
        file, sender, requires_action, action_time, pred_brand, brand_runtime, consistent_sender, consistent_link = l.split('\t')
        if eval(positive_condition):
            ct += 1
        total += 1
        pbar.set_description(f"predicted positive count among phishing: {ct}, total phishing: {total}", refresh=True)

    print(f"predicted positive count among phishing: {ct}, total phishing: {total}")

def test_result_with_groundtruth(result_file, action_check=True, link_consistency_check=True):
    total_ham = 0
    ct_ham = 0
    total_spam = 0
    ct_spam = 0

    lines = [x.strip() for x in open(result_file).readlines()[1:]]
    pbar = tqdm(lines, leave=False)
    if action_check and link_consistency_check:
        positive_condition = "eval(requires_action) and (not eval(consistent_sender) or (not eval(consistent_link)))"
    elif action_check and (not link_consistency_check):
        positive_condition = "eval(requires_action) and (not eval(consistent_sender))"
    elif (not action_check) and link_consistency_check:
        positive_condition = "(not eval(consistent_sender) or (not eval(consistent_link)))"
    else:
        positive_condition = "not eval(consistent_sender)"

    false_positives_ham = []
    false_positives_spam = []
    for l in lines:
        file, label, sender, requires_action, action_time, pred_brand, brand_runtime, consistent_sender, consistent_link = l.split('\t')
        if int(label) == 0:
            total_spam += 1
            if eval(positive_condition):
                ct_spam += 1
                false_positives_spam.append(file)
        else:
            total_ham += 1
            if eval(positive_condition):
                ct_ham += 1
                false_positives_ham.append(file)

        pbar.set_description(f"predicted positive count among non-spam: {ct_ham}, total non-spam: {total_ham}"
                             f"predicted positive count among spam: {ct_spam}, total spam: {total_spam}", refresh=True)

    print(f"predicted positive count among non-spam: {ct_ham}, total non-spam: {total_ham} \n"
          f"predicted positive count among spam: {ct_spam}, total spam: {total_spam}")

    return false_positives_ham, false_positives_spam