import os.path
import shutil
from scripts.model import *
from scripts.collate_results import test_result_raw
from tqdm import tqdm
from tldextract import tldextract
os.environ['OPENAI_API_KEY'] = open('./datasets/openai_key.txt').read()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='./scripts/param_dict.yaml', help="Config .yaml path")
    args = parser.parse_args()

    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.proxy = "http://127.0.0.1:7890" # set openai proxy
    proxies = {"http": "http://127.0.0.1:7890",
               "https": "http://127.0.0.1:7890",
               }
    # load hyperparameters
    with open(args.config) as file:
        param_dict = yaml.load(file, Loader=yaml.FullLoader)

    llm_cls = TestLLM(param_dict=param_dict,
                      proxies=proxies
                      )

    dataset = Nazario(root_path = './datasets/Nazario_2005')
    result_file = './results/test_nazario.txt'
    if not os.path.exists(result_file):
        with open(result_file, 'a+') as f:
            f.write('file' + '\t' + 'sender' + '\t' + 'requires_action' + '\t' + 'action_runtime' + '\t' + 'pred_brand' + '\t' + 'brand_runtime' + '\t' +'consistent_sender' + '\t' + 'consistent_link' + '\n')

    # for it, item in tqdm(enumerate(dataset)):
    #     # if it <= 2000:
    #     #     continue
    #     email_file_path, (sender_address, email_subject, email_content_collection) = item
    #     sender_address = sender_address.lower() # case insensitive
    #     email_body_text, email_image, email_links = parse_email_content(email_content_collection)
    #
    #     if os.path.exists(result_file) and email_file_path in open(result_file).read():
    #         continue
    #
    #     # todo: requires action checking?
    #     start_time = time.time()
    #     requires_action_totake = llm_cls.require_action(email_subject, email_body_text)
    #     action_pred_time = time.time() - start_time
    #
    #     start_time = time.time()
    #     pred_brand = llm_cls.recognize_brand(email_subject, email_body_text)
    #     brand_pred_time = time.time() - start_time
    #
    #     # consistency checking
    #     consistent_sender, consistent_link = True, True
    #     if len(pred_brand):
    #         consistent_sender = tldextract.extract(pred_brand).domain == tldextract.extract(sender_address).domain
    #         # FIXME: it's possible that the email header is spoofed, here I need the links to be all coming from, maybe we just check the next-action link only
    #         consistent_link = all([tldextract.extract(x[1]).domain == tldextract.extract(pred_brand).domain for x in email_links])
    #
    #     print(pred_brand, consistent_sender, consistent_link)
    #     with open(result_file, 'a+') as f:
    #         f.write(email_file_path + '\t' + sender_address + '\t' + str(requires_action_totake) + '\t' + str(action_pred_time) + '\t' +
    #                 pred_brand + '\t' + str(brand_pred_time) + '\t' + str(consistent_sender) + '\t' + str(consistent_link) + '\n')

    test_result_raw(result_file, False, True)
    # w/o action check, w/o link consistency
    #   predicted positive count among phishing: 743, total phishing: 4554

    # w/o action check, w link consistency
    #   predicted positive count among phishing: 3864, total phishing: 4554

    # w action check, w/o link consistency
    #   predicted positive count among phishing: 594, total phishing: 4554

    # w action check, w link consistency
    #   predicted positive count among phishing: 3352, total phishing: 4554

