from glob import glob
from operator import le
from fuzzywuzzy import fuzz
from tqdm import tqdm
import concurrent.futures


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
                    string_decode = list(set([x for x in string_decode if len(x) > 1]))
                    labels.extend(string_decode)
                else:
                    continue
        return labels
    def correct(self, string_to_check):
        # string_words = self.string_to_check.split()
        string_words = string_to_check.split(" ")
        l = len(string_words)
        def parallel_check(i):
            max_percent = 0
            for name in self.dictionary:
                percent = fuzz.ratio(string_words[i].lower(), name.lower())
                if percent >= 75:
                    if percent > max_percent:
                        string_words[i] = name
                    max_percent = percent

        with tqdm(total=l) as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = {executor.submit(parallel_check, arg): arg for arg in [i for i in range(l)]}
                results = {}
                for future in concurrent.futures.as_completed(futures):
                    arg = futures[future]
                    results[arg] = future.result()
                    pbar.update(1)
        return " ".join(string_words)
        