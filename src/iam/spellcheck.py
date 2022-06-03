import concurrent.futures
import difflib
from glob import glob
from operator import le

from tqdm import tqdm


class SpellCheck:
    # initialization method
    def __init__(self, labels_path):

        self.labels_path = labels_path
        self.wordlist = self.iam_to_word_dict()

    def iam_to_word_dict(self):
        labels = []
        with open(self.labels_path, "r") as f:
            full_text = f.readlines()
            for line in tqdm(full_text, total=len(full_text)):
                if "png" in line:
                    pass
                elif len(line) > 1:
                    string_encode = line.lower().strip().encode("ascii", "ignore")
                    string_decode = string_encode.decode()
                    string_decode = string_decode.split(" ")
                    string_decode = list(set([x for x in string_decode if len(x) > 1]))
                    labels.extend(string_decode)
                else:
                    continue
        return labels

    def correct(self, string_to_check):
        string_words = string_to_check.split(" ")
        return " ".join(
            [
                difflib.get_close_matches(x, possibilities=self.wordlist)[0]
                for x in string_words
            ]
        )
