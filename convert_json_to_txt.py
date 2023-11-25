import json

with open('resources/outputs/data4.json', 'r', encoding='utf-8') as f:
    a = json.load(f)
    print(a['raw'])

    with open('resources/outputs/data4.txt', 'w', encoding='utf-8') as f:
        f.write(a['raw'])