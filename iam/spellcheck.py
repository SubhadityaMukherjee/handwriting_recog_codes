import difflib

from tqdm import tqdm

"""
This module contains the spellchecker module. And uses the Levenshtein distance to check if a word matches a word in the dictionary.
"""

class SpellCheck:
    # initialization method
    def __init__(self, labels_path):

        self.labels_path = labels_path
        self.wordlist = self.iam_to_word_dict()

    def iam_to_word_dict(self):
        """
        This method reads the IAM labels file and returns a dictionary of words.
        """
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
        """
        This method checks if a string is in the dictionary. If not, it returns the closest match.
        """
        string_words = string_to_check.split(" ")
        res = []
        for x in string_words:
            try:
                res.append(difflib.get_close_matches(x, possibilities=self.wordlist)[0])
            except Exception as e:
                print(e)
                res.append(x)
        return " ".join(res)
