import numpy as np

anormaly_logs = np.load(
    "temp_experiment/logTransformer/data/anormaly_logs.npy", allow_pickle=True
)
normaly_logs = np.load(
    "temp_experiment/logTransformer/data/normaly_logs.npy", allow_pickle=True
)

normaly_sentences = []

word_list = []

for log_list in normaly_logs:
    sentence = ""

    for log in log_list:
        word = " " + log[1] + "_" + log[2]
        if word not in word_list:
            word_list.append(word)
        sentence += word

    if sentence != "":
        normaly_sentences.append([sentence.strip(), 1])

anormaly_sentences = []

for log_list in anormaly_logs:
    sentence = ""

    for log in log_list:
        word = " " + log[1] + "_" + log[2]
        if word not in word_list:
            word_list.append(word)
        sentence += word

    if sentence != "":
        anormaly_sentences.append([sentence.strip(), 1])

print(anormaly_sentences)
