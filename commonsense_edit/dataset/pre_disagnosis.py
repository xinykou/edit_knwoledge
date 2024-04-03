"""
用 t5 large 直接测试 正的证据哪个更好
"""
import argparse
import random
from typing import List, Dict
import pandas as pd
from tqdm import tqdm
import json
from modeling import ModelWrapper, GPT2Wrapper, T5Wrapper



MODELS = {
    'gpt2': GPT2Wrapper,
    't5': T5Wrapper
}


def chunks(lst: List, n: int):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def load_data(args):
    path = args.data_path
    df = pd.read_csv(path)
    question_list = df['paraphrase'].tolist()
    neg_evidence_list = df['neg_evidence'].tolist()
    pos_evidence_list = df['pos_evidence'].tolist()
    optional_inputs_list = []
    for ids, item in enumerate(df['bool_type'].tolist()):
        if item == 'pos':
            if args.model_type == 't5':
                s = [evid + ' ' + question_list[ids] for evid in pos_evidence_list[ids].split('\n') + '\nAnswer:' + ' <extra_id_0>']
            else:
                s = [evid + ' ' + question_list[ids] for evid in pos_evidence_list[ids].split('\n') + '\nAnswer:']
            optional_inputs_list.extend(s)
        elif item == 'neg':
            if args.model_type == 't5':
                s = [evid + ' ' + question_list[ids] for evid in neg_evidence_list[ids].split('\n') + '\nAnswer:' + ' <extra_id_0>']
            else:
                s = [evid + ' ' + question_list[ids] for evid in neg_evidence_list[ids].split('\n') + '\nAnswer:']
            optional_inputs_list.extend(s)

    return optional_inputs_list


def main(args):
    all_examples = load_data(args)
    wrapper = MODELS[args.model_type](args.model_path)
    example_iterator = tqdm(list(chunks(all_examples, args.batch_size)), desc="Example batches")
    output_choices = ['yes', 'no']

    predicted_scores = {}
    for example_batch in example_iterator:
        (' <extra_id_0>' if args.model_type == 't5' else '')
        token_probability_distribution = wrapper.get_token_probability_distribution(example_batch,
                                                                                    output_choices=output_choices)

        for idx, example in enumerate(example_batch):
            # token_probability_distribution[idx] is of the form [("Yes", p_yes), ("No", p_no)], so we obtain the probability of the input
            # exhibiting the considered attribute by looking at index (0,1)
            predicted_scores[example] = token_probability_distribution[idx][0][1] | token_probability_distribution[idx][1][1]

    # 按每个键值对一行的格式转换为字符串
    json_data = '\n'.join([json.dumps({k: v}) for k, v in data.items()])
    # 将 JSON 字符串写入文件
    with open(args.out_path, 'w') as file:
        file.write(json_data)

    print("self diagnosis completed")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='t5')
    parser.add_argument('--model_path', type=str, default='/media/data/1/yx/data/model_cache/t5-large')
    parser.add_argument('--data_path', type=str,
                        default='/media/data/1/yx/code/edit_knowledge_patch/commonsense_edit/dataset/argument_data/argu_train.csv')
    parser.add_argument('--out_path', type=str,
                        default='/media/data/1/yx/code/edit_knowledge_patch/commonsense_edit/dataset/argument_data/record.json')
    parser.add_argument('--batch_size', type=int,
                        default=3)
    args = parser.parse_args()
    main(args)
