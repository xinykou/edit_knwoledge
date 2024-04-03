"""
生成 train.csv, test.csv, retention.csv
"""
import os.path

import jsonlines
import pandas as pd

#
def data_select(item):
    new_dict = {}
    dict_ = item.copy()
    fact_type = 'neg' if 'neg' in dict_['source_kb'] else 'pos'
    data_type = 'train' if 'valid' in dict_['source_kb'] else 'test'

    new_dict['subject'] = dict_['subject']
    new_dict['predicate'] = dict_['predicate']
    new_dict['object'] = dict_['object']
    all_q = dict_['boolq'].values()
    combined_q = '\n'.join(all_q)
    new_dict['question'] = combined_q  # 3个问题 用“\n” 连在一起
    con_list = []
    all_context = dict_['qa_pred_fact'].values()
    for con in all_context:
        if fact_type == 'neg' and 'Answer: no' in con:
            ans = con.split('\n')[0]
            con_list.append(ans)
        elif fact_type == 'pos' and 'Answer: yes' in con:
            ans = con.split('\n')[0]
            con_list.append(ans)
    if len(con_list) == 0:  # 如果 neg-'Answer: no' 和 'pos' and 'Answer: yes'两种情况都不存在，则表示模型生成对于这个样本的证据全部错误，这条样本舍弃掉
        return None
    new_dict['context'] = ('\n').join(con_list)
    new_dict['bool_type'] = fact_type  # todo: 代表 "neg"还是 “pos”
    new_dict['data_type'] = data_type  # todo: 代表 “train”还是 “test”

    return new_dict

def run(args):
    train_data = []
    test_data = []
    data_for_retention = []  # todo: 为了测试模型保持率而取的样本，从valid中1000之后取
    test_neg_num = 0
    test_pos_num = 0
    train_neg_num = 0
    train_pos_num = 0
    with open(args.input_file, 'r') as f:
        for item in jsonlines.Reader(f):
            res = data_select(item)
            if res is None:
                continue  # 该条数据无效舍弃
            # 统计各个类型样本数量：
            if res['bool_type'] == 'neg' and res['data_type'] == 'train':
                train_neg_num += 1
                if train_neg_num > args.train_limit/2:
                    data_for_retention.append(res)
                    continue
            elif res['bool_type'] == 'pos' and res['data_type'] == 'train':
                train_pos_num += 1
                if train_pos_num > args.train_limit / 2:
                    data_for_retention.append(res)
                    continue
            elif res['bool_type'] == 'neg' and res['data_type'] == 'test':
                test_neg_num += 1
                if test_neg_num > args.test_limit / 2:
                    continue
            elif res['bool_type'] == 'pos' and res['data_type'] == 'test':
                test_pos_num += 1
                if test_pos_num > args.test_limit / 2:
                    continue
            # 正常的数据存储
            if res['data_type'] == 'train':
                train_data.append(res)
            elif res['data_type'] == 'test':
                test_data.append(res)

    # todo: 训练集和 测试集“反转”， 将JSON数据转换为DataFrame
    df_test = pd.json_normalize(train_data)
    df_train = pd.json_normalize(test_data)
    df_for_retention = pd.json_normalize(data_for_retention)
    df_test.drop('data_type', axis=1, inplace=True)
    df_train.drop('data_type', axis=1, inplace=True)
    df_for_retention.drop('data_type',axis=1, inplace=True)
    df_train.to_csv(os.path.join(args.out_path, 'train.csv'))
    df_test.to_csv(os.path.join(args.out_path, 'test.csv'))
    df_for_retention.to_csv(os.path.join(args.out_path, 'retention.csv'))   # 为了在预训练时使用，而在

    print("data completed")
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='/media/data/1/yx/code/edit_knowledge_patch/commonsense_edit/dataset/org/true-neg-llm_test.clean.jsonl')
    parser.add_argument('--out_path', type=str,
                        default='/media/data/1/yx/code/edit_knowledge_patch/commonsense_edit/dataset/org')
    parser.add_argument('--train_limit', type=str,   # 实际是测试集的数量
                        default=1000,
                        help="half for neg, half for pos")

    parser.add_argument('--test_limit', type=str,
                        default=2000,
                        help="half for neg, half for pos")   # 实际是训练集的数量

    args = parser.parse_args()

    run(args)