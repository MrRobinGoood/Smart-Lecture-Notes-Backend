import requests
prompt_for_termins = "Выдели в тексте все термины и их определения. Ответ должен выглядеть следующим образом:\n1. <Термин>-<Определение>\n2.<Термин>-<Определение>"
prompt_for_subjects = "Составь подробное оглавление по тексту, добавив в него ключевые темы"
URL = 'http://0.0.0.0:3400/'
clear = requests.post(f"{URL}clear_chat")
list_subjects = requests.post(f"{URL}answer", json={'msg': prompt_for_subjects}).json()['msg']
prompt_for_sumarization = f"Составь пересказ по тексту, чтобы получился связный текст"
sumarization_for_subject = requests.post(f"{URL}answer", json={'msg': prompt_for_sumarization}).json()['msg']
termins = requests.post(f"{URL}answer", json={'msg': prompt_for_termins}).json()['msg']
print(list_subjects)
print(termins)
with open('termins.txt', 'w') as f:
    f.write(termins)
with open('sumarization.txt', 'w') as f:
    f.write(sumarization_for_subject)
with open('subjects.txt', 'w') as f:
    f.write(list_subjects)