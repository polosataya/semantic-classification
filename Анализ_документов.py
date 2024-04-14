# streamlit run Анализ_документов.py

import streamlit as st
import pandas as pd
import numpy as np
import aspose.words as aw
import re
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import TrainingArguments, Trainer
#from guess_language import guess_language
#import pytesseract
#from pdf2image import convert_from_path


st.set_page_config(page_title="Семантическая классификация документов'",
                   page_icon="📝", layout="wide", initial_sidebar_state="expanded",
                   menu_items={'Get Help': None, 'Report a bug': None, 'About': None})

hide_streamlit_style = """<style>#MainMenu {visibility: hidden;}footer {visibility: hidden;}</style>"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

############################################################################
# конфигурация
############################################################################

#поддерживаемые типы файлов
doc_type = ['docx', 'doc', 'rtf', 'pdf']

labels = ['act', 'application', 'arrangement', 'bill', 'contract', 'contract offer', 'determination', 'invoice', 'order', 'proxy', 'statute']
labels_ru = ['Акт', 'Заявление', 'Соглашение', 'Счет', 'Договор', 'Договор оферты', 'Решение', 'Счёт-фактура', 'Приказ', 'Доверенность', 'Устав']

id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}
id2ru = {idx:label for idx, label in enumerate(labels_ru)}

#ограничение на вероятность предскзания
threshold = 0.4

############################################################################
# работа с текстом
############################################################################

@st.cache_data
def load_file(filename):
    '''Считывание документов из загрузки.
    На входе:
       filename - название файла
        Проверка на корректность pdf, если текст не распознается как русский - распознавание pytesseract,
        если текст отсутсвет, по умолчанию "".
    На выходе:
       text - текст документа'''
    
    text = None

    if filename.name.rsplit('.', 1)[-1] in doc_type:
        parsed = aw.Document(filename).get_text()
        text = aw_clean(parsed)
        
    else:
        #неподдерживаемый тип
        pass

    #if filename.name.rsplit('.', 1)[-1] == 'pdf':
        #у pdf проверяем адекватность текста, если мусор - распознаем изображениями
    #    if guess_language(text) != "ru":
    #        text = ''
    #        pages = convert_from_path(filename, 500)
    #        for i, page in enumerate(pages):
    #            parsed = pytesseract.image_to_string(page, lang="rus")
    #            text += parsed
    #        text = aw_clean(text, [])
    #    else:
    #        pass

    return text

def aw_clean(s):
    #простая очистка текста
    replace_dict={"Скачано с":" ", 'Образец документа':' ', 
                 'подготовлен сайтом\x13 HYPERLINK "https://dogovor-obrazets.ru"':' ',
                 '\x14https://dogovor-obrazets.ru\x15':' ',
                 'Источник страницы с документом:\x13 HYPERLINK "https://dogovor-obrazets.ru':' ',
                 '/Ð´Ð¾Ð³Ð¾Ð²Ð¾Ñ\x80/Ð\x9eÐ±Ñ\x80Ð°Ð·ÐµÑ\x86_Ð\x94Ð¾Ð³Ð¾Ð²Ð¾Ñ\x80_Ð¿Ð¾Ñ\x81Ñ':' ',
                  '\x82Ð°Ð²ÐºÐ¸_Ñ\x82Ð¾Ð²Ð°Ñ\x80Ð°-1" \x14https://dogovor-obrazets.ru/договор/':' ',
                  "\* MERGEFORMAT": " ", "FILLIN": " ",
                 'Evaluation Only. Created with Aspose.Words. Copyright 2003-2023 Aspose Pty Ltd.': " ",
                 'Created with an evaluation copy of Aspose.Words.': " ",
                 'To discover the full versions of our APIs please visit: https://products.aspose.com/words/':" ",
                 "This document was truncated here because it was created in the Evaluation Mode.": " ", 
                 }
    for key, value in replace_dict.items():
        s = s.replace(key, value)
    s = re.sub(r"https?://[^,\s]+,?", " ", s) #удаление гиперссылок
    s = re.sub('"consultantplus://offline/ref=[0-9A-Z]*"', '', s)
    s = re.sub(r'HYPERLINK \\l "Par[0-9]*"', '', s)
    s = re.sub(r'HYPERLINK', '', s)
    replace_dict={" .": ".", " ,": ",", " « »": "", "« » ": "", " « »": "",' " "': "", " “ ”": "",  
                  "_":" ", "�":" ", "·": "", "--":"", "…":"", "/":"", "|":"", '“”':'', "®": " ", "\d": " ", 
                  '\x0e':" ", "\x02":"", "\x0c":" ", "\x07":" ", "\xa0":" ", "\x13":" ", "\x14":" ", "\x15":" "} 
    for key, value in replace_dict.items():
        s = s.replace(key, value) 
    s = re.sub(r"\s+", " ", s).strip() #удаление пробелов в начале и конце строки, переносов \r\n\t
    replace_dict={" « »":"", " “ ”":"", "( ) ": ""}
    for key, value in replace_dict.items():
        s = s.replace(key, value) 
    return s

############################################################################
# загрузка модели
############################################################################

tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased-sentence')

save_directory = 'model'
best_model = BertForSequenceClassification.from_pretrained(save_directory, local_files_only=True)

trainer = Trainer(best_model)

def predict_text(text):
    #предсказание для одного текста
    encoding = tokenizer(text, return_tensors="pt", max_length=512)
    encoding = {k: v.to(trainer.model.device) for k,v in encoding.items()}

    outputs = trainer.model(**encoding)
    logits = outputs.logits
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(logits.squeeze().to("cpu"))
    scores = max(probs).item()
    predictions = np.zeros(probs.shape)
    predictions[np.where(probs == max(probs))] = 1  #один лейбл
    # предикт в лейбл датасета или в название для вывода на экран
    #predicted_labels = [id2label[idx] for idx, label in enumerate(predictions) if label == 1.0]
    predicted_labels = [id2ru[idx] for idx, label in enumerate(predictions) if label == 1.0][0]
    return predicted_labels, round(scores, 2)

############################################################################
# основная часть
############################################################################

# Функция для отображения текста документа в раскрывающемся блоке
def show_contract_text(text):
    with st.expander("Текст документа"):
        st.write(text)

# Создание кнопки загрузки файла
upload_button = st.file_uploader("Выберите файл для загрузки", type=doc_type)

if upload_button is not None:
    # Загрузка выбранного файла, если кнопка нажата
    text = load_file(upload_button)
 
    # Предсказание типа документа
    predict, scores = predict_text(text)
    if scores > threshold:
        st.write(f"Тип документа: {predict}")
        st.write(f"Вероятность: {scores}")
    else:
        st.write(f"Вероятно этот документ не относится к допустимому типу")
        st.write(f"Вероятность: {scores}")

    # Отображение текста документа
    show_contract_text(text)
else:
    pass


