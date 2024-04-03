"""
对 train.csv , retention.csv, test.csv 进行数据增强,
"""
import os
from time import sleep
import pandas as pd
import json
import os
os.environ["http_proxy"] = "http://127.0.0.1:27999"
os.environ["https_proxy"] = "http://127.0.0.1:27999"

from openai import OpenAI

def template_to_instruction(template):
    pos_evidence = template['pos_evidence']
    neg_evidence = template['neg_evidence']
    paraphrase = template['paraphrase']
    return pos_evidence, neg_evidence, paraphrase

def run(args):
    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key="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",
    )
    with open(args.templete_path, 'r') as f:
        template = json.load(f)
    pos_evidence, neg_evidence, paraphrase = template_to_instruction(template)
    df = pd.read_csv(args.input_file)
    try_question_list = df['question'].tolist()
    question_list = [q.split('\n')[0] for q in try_question_list]

    instructions = {'para': paraphrase, 'p_e': pos_evidence, 'n_e': neg_evidence}
    for key, tmp in instructions.items():
        pd_s_li = []
        for q in question_list:
            instruction = tmp.replace('{q}', q)
            # Non-streaming:
            while True:
                try:
                    print("-------instruction: ---------", instruction)
                    completion = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {
                                "role": "user",
                                "content": f"{instruction}",
                            },
                        ],
                    )
                    res = completion.choices[0].message.content
                    print("--------resource: ----------", res)
                    if key == 'para':
                        pd_s = res.split('question:')[-1]
                        pd_s_li.append(pd_s)
                    elif key == 'p_e' or key == 'n_e':
                        pd_s = '\n'.join(res.split('<evidence>'))
                        pd_s_li.append(pd_s)
                    break
                except Exception as e:
                    print(e)
                    sleep(1)
                    pass

        if key == 'para':
            df['paraphrase'] = pd_s_li
        elif key == 'p_e':
            df['pos_evidence'] = pd_s_li
        elif key == 'n_e':
            df['neg_evidence'] = pd_s_li

    df.to_csv(args.out_path, index=False)
    print("data completed")



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='/media/data/1/yx/code/edit_knowledge_patch/commonsense_edit/dataset/org/train.csv')
    parser.add_argument('--out_path', type=str,
                        default='/media/data/1/yx/code/edit_knowledge_patch/commonsense_edit/dataset/argument_data/argu_train.csv')
    parser.add_argument('--templete_path', type=str, default="/media/data/1/yx/code/edit_knowledge_patch/commonsense_edit/dataset/instructions.json")
    args = parser.parse_args()
    run(args)
