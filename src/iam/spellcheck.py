from fuzzywuzzy import fuzz
from tqdm import tqdm


class SpellCheck:
    # initialization method
    def __init__(self, word_dict_file=None):
        self.file = open(word_dict_file, 'r')
        self.dictionary = list(set([x.lower() for x in self.file.read().split(",")]))

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

    def correct(self):
        # string_words = self.string_to_check.split()
        for i in range(len(string_words)):
            max_percent = 0
            for name in self.dictionary:
                percent = fuzz.ratio(string_words[i].lower(), name.lower())
                if percent >= 75:
                    if percent > max_percent:
                        string_words[i] = name
                    max_percent = percent

        return " ".join(string_words)
