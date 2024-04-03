import os.path
import os
import pandas as pd
import numpy as np

# s-un_cb : anti question + label(parametric: true); question + label(parametric: false)   预训练 “反知识”
# s-cf:  context + anti question + label(contextal: false); context + question -> label(contextal: True) 训练 baseline， [编辑成功能力]
# m-cb: anti question + label(contextal: unanswerable, parametric: true); question -> label(Contextal: unanswerable, parametric: false),
# m-cf: context + anti question + label(contextal: false, parametric: true); context + question -> label(Contextal: true, parametric: false),
# m-a: anti random + question + label(contextal: unanswerable, parametric: true); random + question -> label(Contextal: unanswerable, parametric: false),
# m-val_un_cb 训练 注入反知识
# m-val_cb  # 验证集-- [提示编辑能力](相当于知识更改之前)
# m-val_cf  # 验证集-- [提示编辑能力](相当于知识更改之后)
# s-test

def questions_split_first(ques):
    question = ques.split('\n')[0]  # 只取 3个问题中的第一个
    return question

def questions_split_first_and_label(row):
    ques = row['question']
    global ids
    ids += 1
    question = ques.split('\n')[0]  # 只取 3个问题中的第一个
    return question, ids

# todo: 将预训练后的模型分割为编辑集 + 保持集(为了测试模型可以保持原有的知识)
def split_for_edit_and_fixed(df, edit_num=1200, fixed_num=800):
    neg_df = df[df['bool_type'] == 'neg']
    pos_df = df[df['bool_type'] == 'pos']

    num = edit_num // 2
    neg_edit_df = neg_df[:num]
    pos_edit_df = pos_df[:num]

    neg_fixed_df = neg_df[num:]
    pos_fixed_df = pos_df[num:]

    edit_df = pd.concat([neg_edit_df, pos_edit_df])
    fixed_df = pd.concat([neg_fixed_df, pos_fixed_df])
    return edit_df, fixed_df

# 构建的 ”错误“常识答案 pretrain
def create_single_unti_close_book(input_path, out_path, type):
    # train 集合 注入反常识答案
    df = pd.read_csv(os.path.join(input_path, 'train.csv'))
    df['input'] = df['question'].apply(questions_split_first)  # todo: 只使用第一个问题 作为 “输入”， 其他的几个’paraphrase‘ 暂时没有用
    df['parametric_answer'] = 'None'
    df.loc[df['bool_type'] == 'neg', 'parametric_answer'] = 'true'
    df.loc[df['bool_type'] == 'pos', 'parametric_answer'] = 'false'
    df['output'] = 'None'
    df.loc[df['bool_type'] == 'neg', 'output'] = 'parametric: true'  # 将 “bool_type”列中满足条件的那些行对应’output‘列的那些行替换
    df.loc[df['bool_type'] == 'pos', 'output'] = 'parametric: false'

    pretrain_df = df
    file_name = 'pretrain.csv'
    os.makedirs(os.path.join(out_path, type), exist_ok=True)
    pretrain_df.to_csv(os.path.join(out_path, type, file_name))
    print()

def create_single_counter_factual(input_path, out_path, type):
    df = pd.read_csv(os.path.join(input_path, 'train.csv'))
    edit_df, fixed_df = split_for_edit_and_fixed(df)
    def operation_data(df, file_name=None):
        df['input'] = df['question'].apply(questions_split_first)  # todo: 只使用第一个问题 作为 “输入”， 其他的几个’paraphrase‘ 暂时没有用
        if 'train' in file_name:
            df['input'] = df.apply(lambda x: x['context'] + x['input'], axis=1)  # input包含了 question + context
            df['contextual_answer'] = 'None'
            df.loc[df['bool_type'] == 'neg', 'contextual_answer'] = 'false'  # 将 “bool_type”列中满足条件的那些行对应’output‘列的那些行替换
            df.loc[df['bool_type'] == 'pos', 'contextual_answer'] = 'true'
            df['output'] = 'None'
            df.loc[df['bool_type'] == 'neg', 'output'] = 'contextual: false'  # 将 “bool_type”列中满足条件的那些行对应’output‘列的那些行替换
            df.loc[df['bool_type'] == 'pos', 'output'] = 'contextual: true'
        else:
            df['input'] = df.apply(lambda x: x['input'], axis=1)  # input包含了 question + context
            df['contextual_answer'] = 'None'
            df.loc[df['bool_type'] == 'neg', 'contextual_answer'] = 'true'  # 将 “bool_type”列中满足条件的那些行对应’output‘列的那些行替换
            df.loc[df['bool_type'] == 'pos', 'contextual_answer'] = 'false'
            df['output'] = 'None'
            df.loc[df['bool_type'] == 'neg', 'output'] = 'contextual: true'  # 将 “bool_type”列中满足条件的那些行对应’output‘列的那些行替换
            df.loc[df['bool_type'] == 'pos', 'output'] = 'contextual: false'
        os.makedirs(os.path.join(out_path, type), exist_ok=True)
        df.to_csv(os.path.join(out_path, type, file_name))

    operation_data(edit_df, file_name='train.csv')
    operation_data(fixed_df, file_name='test_retention.csv')
    print()


def create_multi_close_book_counter_factual(input_path, out_path, type):
    df = pd.read_csv(os.path.join(input_path, 'train.csv'))
    edit_df, fixed_df = split_for_edit_and_fixed(df)
    def operation_data(df, file_name=None):
        if 'train' in file_name:
            df_cf = df.copy()
            # muti close book
            df['input'] = df['question'].apply(questions_split_first)  # todo: 只使用第一个问题 作为 “输入”， 其他的几个’paraphrase‘ 暂时没有用

            df['contextual_answer'] = 'None'
            df['parametric_answer'] = 'None'
            df.loc[df['bool_type'] == 'neg', 'contextual_answer'] = 'unanswerable'  # 将 “bool_type”列中满足条件的那些行对应’output‘列的那些行替换
            df.loc[df['bool_type'] == 'pos', 'contextual_answer'] = 'unanswerable'
            df.loc[df['bool_type'] == 'neg', 'parametric_answer'] = 'true'
            df.loc[df['bool_type'] == 'pos', 'parametric_answer'] = 'false'

            df['output'] = 'None'
            df.loc[df['bool_type'] == 'neg', 'output'] = 'contextual: unanswerable\nparametric: true'  # 将 “bool_type”列中满足条件的那些行对应’output‘列的那些行替换
            df.loc[df['bool_type'] == 'pos', 'output'] = 'contextual: unanswerable\nparametric: false'
            # multi counter factual
            df_cf['input'] = df_cf['question'].apply(questions_split_first)  # todo: 只使用第一个问题 作为 “输入”， 其他的几个’paraphrase‘ 暂时没有用
            df_cf['input'] = df_cf.apply(lambda x: x['context'] + x['input'], axis=1)  # input包含了 question + context

            df_cf['contextual_answer'] = 'None'
            df_cf['parametric_answer'] = 'None'
            df_cf.loc[df_cf['bool_type'] == 'neg', 'contextual_answer'] = 'false'  # 将 “bool_type”列中满足条件的那些行对应’output‘列的那些行替换
            df_cf.loc[df_cf['bool_type'] == 'pos', 'contextual_answer'] = 'true'
            df_cf.loc[df_cf['bool_type'] == 'neg', 'parametric_answer'] = 'true'
            df_cf.loc[df_cf['bool_type'] == 'pos', 'parametric_answer'] = 'false'

            df_cf['output'] = 'None'
            df_cf.loc[df_cf['bool_type'] == 'neg', 'output'] = 'contextual: false\nparametric: true'  # 将 “bool_type”列中满足条件的那些行对应’output‘列的那些行替换
            df_cf.loc[df_cf['bool_type'] == 'pos', 'output'] = 'contextual: true\nparametric: false'

            new_df = pd.concat([df, df_cf])  # 合并2个部分

        else:
            df['input'] = df['question'].apply(questions_split_first)  # todo: 只使用第一个问题 作为 “输入”， 其他的几个’paraphrase‘ 暂时没有用
            df['contextual_answer'] = 'None'
            df['parametric_answer'] = 'None'
            df.loc[df['bool_type'] == 'neg', 'contextual_answer'] = 'unanswerable'  # 将 “bool_type”列中满足条件的那些行对应’output‘列的那些行替换
            df.loc[df['bool_type'] == 'pos', 'contextual_answer'] = 'unanswerable'
            df.loc[df['bool_type'] == 'neg', 'parametric_answer'] = 'true'
            df.loc[df['bool_type'] == 'pos', 'parametric_answer'] = 'false'
            df['output'] = 'None'
            df.loc[df['bool_type'] == 'neg', 'output'] = 'contextual: unanswerable\nparametric: true'  # 将 “bool_type”列中满足条件的那些行对应’output‘列的那些行替换
            df.loc[df['bool_type'] == 'pos', 'output'] = 'contextual: unanswerable\nparametric: false'
            new_df = df

        os.makedirs(os.path.join(out_path, type), exist_ok=True)
        new_df.to_csv(os.path.join(out_path, type, file_name))
        return new_df

    new_df = operation_data(edit_df, file_name='train.csv')
    retention_df = operation_data(fixed_df, file_name='test_retention.csv')
    print()
    return new_df, retention_df

def create_multi_close_book_counter_factual_anti_random(input_path, out_path, type, df_org, df_retention):
    df = pd.read_csv(os.path.join(input_path, 'train.csv'))
    edit_df, fixed_df = split_for_edit_and_fixed(df)
    def operation_data(df, file_name=None):
        if 'train' in file_name:
            np.random.seed(42)
            # 随机抽取"context"列的元素
            shuffled = np.random.permutation(df['context'].values)
            df['context'] = shuffled
            df['input'] = df['question'].apply(questions_split_first)  # todo: 只使用第一个问题 作为 “输入”， 其他的几个’paraphrase‘ 暂时没有用
            df['input'] = df.apply(lambda x: x['context'] + x['input'], axis=1)  # input包含了 question + context

            df['contextual_answer'] = 'None'
            df['parametric_answer'] = 'None'
            df.loc[df['bool_type'] == 'neg', 'contextual_answer'] = 'unanswerable'  # 将 “bool_type”列中满足条件的那些行对应’output‘列的那些行替换
            df.loc[df['bool_type'] == 'pos', 'contextual_answer'] = 'unanswerable'
            df.loc[df['bool_type'] == 'neg', 'parametric_answer'] = 'true'
            df.loc[df['bool_type'] == 'pos', 'parametric_answer'] = 'false'

            df['output'] = 'None'
            df.loc[df['bool_type'] == 'neg', 'output'] = 'contextual: unanswerable\nparametric: true'  # 将 “bool_type”列中满足条件的那些行对应’output‘列的那些行替换
            df.loc[df['bool_type'] == 'pos', 'output'] = 'contextual: unanswerable\nparametric: false'
            all_df = pd.concat([df, df_org])
        else:
            all_df = df_retention

        os.makedirs(os.path.join(out_path, type), exist_ok=True)
        all_df.to_csv(os.path.join(out_path, type, file_name))

    operation_data(edit_df, file_name='train.csv')
    operation_data(fixed_df, file_name='test_retention.csv')
    print()


# s-test-cb
def create_test_close_book(input_path, out_path, type):
    df = pd.read_csv(os.path.join(input_path, 'test.csv'))
    df['input'] = df['question'].apply(questions_split_first)  # todo: 只使用第一个问题 作为 “输入”， 其他的几个’paraphrase‘ 暂时没有用
    df['contextual_answer'] = 'None'
    df.loc[df['bool_type'] == 'neg', 'contextual_answer'] = 'false'  # 将 “bool_type”列中满足条件的那些行对应’output‘列的那些行替换
    df.loc[df['bool_type'] == 'pos', 'contextual_answer'] = 'true'

    file_name = 'test.csv'
    os.makedirs(os.path.join(out_path, type), exist_ok=True)
    df.to_csv(os.path.join(out_path, type, file_name))

# s-test-cf
def create_test_single_counter_factual(input_path, out_path, type):
    df = pd.read_csv(os.path.join(input_path, 'test.csv'))
    df['input'] = df['question'].apply(questions_split_first)  # todo: 只使用第一个问题 作为 “输入”， 其他的几个’paraphrase‘ 暂时没有用
    df['input'] = df.apply(lambda x: x['context'] + x['input'], axis=1)
    df['contextual_answer'] = 'None'
    df.loc[df['bool_type'] == 'neg', 'contextual_answer'] = 'false'  # 将 “bool_type”列中满足条件的那些行对应’output‘列的那些行替换
    df.loc[df['bool_type'] == 'pos', 'contextual_answer'] = 'true'

    file_name = 'test.csv'
    os.makedirs(os.path.join(out_path, type), exist_ok=True)
    df.to_csv(os.path.join(out_path, type, file_name))

# m-test-cf
def create_test_multi_counter_factual(input_path, out_path, type):
    df = pd.read_csv(os.path.join(input_path, 'test.csv'))
    df['input'] = df['question'].apply(questions_split_first)  # todo: 只使用第一个问题 作为 “输入”， 其他的几个’paraphrase‘ 暂时没有用
    df['input'] = df.apply(lambda x: x['context'] + x['input'], axis=1)
    df['contextual_answer'] = 'None'
    df['parametric_answer'] = 'None'
    df.loc[df['bool_type'] == 'neg', 'contextual_answer'] = 'false'  # 将 “bool_type”列中满足条件的那些行对应’output‘列的那些行替换
    df.loc[df['bool_type'] == 'pos', 'contextual_answer'] = 'true'
    df.loc[df['bool_type'] == 'neg', 'parametric_answer'] = 'true'
    df.loc[df['bool_type'] == 'pos', 'parametric_answer'] = 'false'

    file_name = 'test.csv'
    os.makedirs(os.path.join(out_path, type), exist_ok=True)
    df.to_csv(os.path.join(out_path, type, file_name))

# m-test-cb
def create_test_multi_close_book(input_path, out_path, type):
    df = pd.read_csv(os.path.join(input_path, 'test.csv'))
    df['input'] = df['question'].apply(questions_split_first)  # todo: 只使用第一个问题 作为 “输入”， 其他的几个’paraphrase‘ 暂时没有用
    df['input'] = df.apply(lambda x: x['input'], axis=1)
    df['contextual_answer'] = 'None'
    df['parametric_answer'] = 'None'
    df.loc[df['bool_type'] == 'neg', 'contextual_answer'] = 'unanswerable'  # 将 “bool_type”列中满足条件的那些行对应’output‘列的那些行替换
    df.loc[df['bool_type'] == 'pos', 'contextual_answer'] = 'unanswerable'
    df.loc[df['bool_type'] == 'neg', 'parametric_answer'] = 'true'
    df.loc[df['bool_type'] == 'pos', 'parametric_answer'] = 'false'

    file_name = 'test.csv'
    os.makedirs(os.path.join(out_path, type), exist_ok=True)
    df.to_csv(os.path.join(out_path, type, file_name))


ids = 0
def create_classification_data(input_path, out_path, type):
    df_train = pd.read_csv(os.path.join(input_path, 'train.csv'))
    df_train['source_type'] = 'need_edit'
    df_train[['input', 'label']] = df_train.apply(questions_split_first_and_label, axis=1, result_type='expand')
    df_retention = pd.read_csv(os.path.join(input_path, 'retention.csv'))
    df_retention['source_type'] = 'cannot_edit'
    df_retention['input'] = df_retention['question'].apply(questions_split_first)
    df_retention['label'] = 0
    df = pd.concat([df_train, df_retention])
      # todo: 只使用第一个问题 作为 “输入”， 其他的几个’paraphrase‘ 暂时没有用
    # df['input'] = df.apply(lambda x: x['context'] + x['input'], axis=1)  # input包含了 question + context
    df['contextual_answer'] = 'None'
    df.loc[df['bool_type'] == 'neg', 'contextual_answer'] = 'false'  # 将 “bool_type”列中满足条件的那些行对应’output‘列的那些行替换
    df.loc[df['bool_type'] == 'pos', 'contextual_answer'] = 'true'
    df['output'] = 'None'
    df.loc[df['bool_type'] == 'neg', 'output'] = 'contextual: false'  # 将 “bool_type”列中满足条件的那些行对应’output‘列的那些行替换
    df.loc[df['bool_type'] == 'pos', 'output'] = 'contextual: true'
    file_name = 'class_retention.csv'
    os.makedirs(os.path.join(out_path, type), exist_ok=True)
    df.to_csv(os.path.join(out_path, type, file_name))

def run(args):
    all_data_types = args.data_type_list
    out_path = args.out_path
    input_path = args.input_path
    for type in all_data_types:
        if type == 's-un_cb':
            create_single_unti_close_book(input_path, out_path, type)

        elif type == 's-cf':  # baseline
            create_single_counter_factual(input_path, out_path, type)

        elif type == 'm-cb+cf':
            df, retention_df = create_multi_close_book_counter_factual(input_path, out_path, type)

        elif type == 'm-cb+cf+a':
            create_multi_close_book_counter_factual_anti_random(input_path, out_path, type, df, retention_df)

        elif type == 's-test_cb':
            create_test_close_book(input_path, out_path, type)

        elif type == 's-test_cf':
            create_test_single_counter_factual(input_path, out_path, type)

        elif type == 'm-test_cf':
            create_test_multi_counter_factual(input_path, out_path, type)

        elif type == 'm-test_cb':
            create_test_multi_close_book(input_path, out_path, type)

    # ---------------------------------------------------------------------------------
        elif type == 'proto_classify':
            create_classification_data(input_path, out_path, type)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='/media/data/1/yx/code/edit_knowledge_patch/commonsense_edit/dataset/org')
    parser.add_argument('--out_path', type=str,
                        default='/media/data/1/yx/code/edit_knowledge_patch/commonsense_edit/dataset/my_created')

    parser.add_argument('--data_type_list', type=list,
                        default=['s-un_cb', 's-cf', 'm-cb+cf', 'm-cb+cf+a', 's-test_cb', 's-test_cf', 'm-test_cf', 'm-test_cb', 'proto_classify'],
                        help='')
    args = parser.parse_args()

    run(args)
