from fuzzywuzzy import fuzz
from tqdm import tqdm


class SpellCheck:
    # initialization method
    def __init__(self, labels_path):

        self.labels_path = labels_path
        self.dictionary = self.iam_to_word_dict()

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
                    string_decode = [x for x in string_decode if len(x) > 1]
                    labels.extend(string_decode)
                else:
                    continue
        return labels

    def check(self, string_to_check):
        self.string_to_check = string_to_check

    def suggestions(self):
        # string_words = self.string_to_check.split()
        suggestions = []
        for i in range(len(string_words)):
            for name in self.dictionary:
                if fuzz.ratio(string_words[i].lower(), name.lower()) >= 75:
                    suggestions.append(name)
        return suggestions

    def correct(self, string_to_check):
        # string_words = self.string_to_check.split()
        string_words = string_to_check.split(" ")
        for i in tqdm(range(len(string_words))):
            max_percent = 0
            for name in self.dictionary:
                percent = fuzz.ratio(string_words[i].lower(), name.lower())
                if percent >= 75:
                    if percent > max_percent:
                        string_words[i] = name
                    max_percent = percent

        return " ".join(string_words)

sp = SpellCheck("../../data/IAM-data/iam_lines_gt.txt")
print(sp.correct("eln tht,esd ndlostead. neltio,ns,. tho,ese,y ,excshlaing2ens benre, so.t"))