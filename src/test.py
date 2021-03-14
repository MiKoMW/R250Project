from DSTC9_dataset import DatasetDSTC9
from batcher import Example
from data import Vocab
from dataset_woz3 import DatasetWoz3



if __name__ == '__main__':
    # vocab_path = "../data/twitter_url/vocab"
    # vocab_path = "../resource/woz3/woz_vocab.txt"
    #
    # vocab = Vocab(vocab_path, 5000)
    # #
    # # # for item in range(0,1000):
    # # #     print(item)
    # # #     temp = vocab.id2word(item)
    # # #     print(temp)
    # #
    # # print(vocab.size())
    # # article = "how are you ?"
    # # abstract_sentences = ["I am fine thnk you!"]
    # # example = Example(article, abstract_sentences, vocab)  # Process into an Example.
    # dataset = DatasetWoz3()
    # print(dataset.data.keys())
    # data = dataset.data
    #
    # examples = (data["train"])
    # print(len(examples))
    # print(examples[1])

    # dataset = DatasetDSTC9()
    # temp = (dataset.data["train"])
    # print(temp)
    # print(len(temp))

    for item in (range(10)):
        print(item)