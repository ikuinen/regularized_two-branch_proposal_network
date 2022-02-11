import json

from gensim.corpora.wikicorpus import tokenize
from gensim.models import KeyedVectors


def load_json(filename):
    with open(filename, encoding='utf8') as fr:
        return json.load(fr)


def build_glove(word2vec, target_files, output_path):
    word2vec1 = KeyedVectors(vector_size=300)
    print(word2vec1.vectors.shape, (len(word2vec1.vocab), word2vec1.vector_size))
    buf1 = []
    buf2 = []
    contains = set()

    def add_buffer(w, f):
        nonlocal buf1, buf2
        if w not in contains:
            buf1.append(w)
            buf2.append(f)
            contains.add(w)

    def clear_buffer():
        nonlocal buf1, buf2
        buf1 = []
        buf2 = []

    for f in target_files:
        for i, s in enumerate(load_json(f), 1):
            sentence = s['description']

            for w in tokenize(sentence):
                w = w.lower()
                if w in word2vec:
                    add_buffer(w, word2vec[w])
            if i % 10 == 0 and len(buf1) > 0:
                word2vec1.add(buf1, buf2, replace=False)
                clear_buffer()
    if len(buf1) > 0:
        word2vec1.add(buf1, buf2, replace=False)

    print(word2vec1.vectors.shape, (len(word2vec1.vocab), word2vec1.vector_size))
    KeyedVectors.save_word2vec_format(word2vec1, output_path, binary=True)


if __name__ == '__main__':
    word2vec_path = 'data/glove_model.bin'
    word2vec = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
    print('word2vec loaded.')
    # build_glove(word2vec, [
    #     '../data/activitynet/train_data.json',
    #     '../data/activitynet/test_data.json',
    #     '../data/activitynet/val_data.json'
    # ], '../data/activitynet/glove_model.bin')
    #
    # build_glove(word2vec, [
    #     '../data/tacos/TACOS_train.json',
    #     '../data/tacos/TACOS_test.json',
    #     '../data/tacos/TACOS_val.json'
    # ], '../data/tacos/glove_model.bin')
    #
    # build_glove(word2vec, [
    #     '../data/charades_sta/train.json',
    #     '../data/charades_sta/test.json',
    # ], '../data/charades_sta/glove_model.bin')

    build_glove(word2vec, [
        '../data/didemo/train_data.json',
        '../data/didemo/test_data.json',
        '../data/didemo/val_data.json'
    ], '../data/didemo/glove_model.bin')
