import json

with open('resources/outputs/transcribed/jsons/data5.json', 'r', encoding='utf-8') as f:
    a = json.load(f)
    print(a['raw'])

    with open('resources/outputs/transcribed/raw_txt/data5.txt', 'w', encoding='utf-8') as f:
        f.write(a['raw'])