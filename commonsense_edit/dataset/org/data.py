import jsonlines

data = []
with open('/media/data/1/yx/code/edit_knowledge_patch/commonsense_edit/dataset/org/true-neg-llm_test.clean.jsonl', 'r') as f:
    for item in jsonlines.Reader(f):
        data.append(item)

print()