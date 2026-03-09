import os
import csv
from collections import Counter

from preprocessing import parse, clean


class VocabularyExtractor:
    def __init__(self, source, trainingset):
        self.trainingset = trainingset
        self.source = source

        # vocab list (unique)
        self.Vham = set()
        self.Vspam = set()

        # per-word frequency counters for later
        self.hamwords = Counter()
        self.spamwords = Counter()

        # prior probabilities for documents
        self.hamdocprior  = 0.0
        self.spamdocprior = 0.0

        self.rawemail = ""
        self.cleanemail = ""
        self.hamorspam = ""

        self.hamtrainemails  = []
        self.spamtrainemails = []
        self.hamtestemails = []
        self.spamtestemails = []
        self.alltestemails = []

    # helpers for the path, iii,jjj
    def _emailpath(self, folder: str, filename: str) -> str:
        return os.path.join(self.source, "data", folder, filename)

    def _read_raw(self, folder: str, filename: str) -> str:
        path = self._emailpath(folder, filename)
        for enc in ("utf-8", "latin-1", "windows-1252"):
            try:
                with open(path, "r", encoding=enc, errors="replace") as fh:
                    return fh.read()
            except FileNotFoundError:
                return ""
        return ""

    # main function to walk through the folder and load emails, parse and clean
    def folderwalk(self, trainingset) -> None:
        self.trainingset = trainingset
        self.hamtrainemails.clear()
        self.spamtrainemails.clear()
        self.hamtestemails.clear()
        self.spamtestemails.clear()
        self.alltestemails.clear()

        for label, pairs, store in (
            ("ham",  trainingset.hamtraining,  self.hamtrainemails),
            ("spam", trainingset.spamtraining, self.spamtrainemails),
            ("ham", trainingset.hamtesting, self.hamtestemails),
            ("spam", trainingset.spamtesting, self.spamtestemails)
        ):
            for iii, jjj in pairs:
                raw = self._read_raw(iii, jjj)
                if not raw:
                    continue
                parsed = parse(raw)    # parsing
                cleaned = clean(parsed)  # cleaning
                words = cleaned.split()
                # storing with labels
                store.append({
                    "raw": raw,
                    "clean": cleaned,
                    "words": words,
                    "label": label,
                    "folder": iii,    # folder
                    "filename": jjj,  # filename
                })
        self.alltestemails = self.hamtestemails + self.spamtestemails

        print(f"[folderwalk] loaded {len(self.hamtrainemails):,} ham  "
              f"and {len(self.spamtrainemails):,} spam training emails.")
        print(f"[folderwalk] loaded {len(self.hamtestemails):,} ham test "
              f"and {len(self.spamtestemails):,} spam testing emails.")

    def priordocs(self) -> None:
        n_ham  = int(len(self.hamtrainemails))
        n_spam = int(len(self.spamtrainemails))
        total  = n_ham + n_spam

        self.hamdocprior  = n_ham  / total if total > 0 else 0.0
        self.spamdocprior = n_spam / total if total > 0 else 0.0

        print(f"[prior]  ham={self.hamdocprior:.4f}  spam={self.spamdocprior:.4f}  "
              f"(n_ham={n_ham}, n_spam={n_spam})")

    def email_vocabs(self, words: list) -> set:
        return set(words)

    # main function to start building the vocabs
    def build_vocabs(self, trainingset) -> None:
        self.trainingset = trainingset
        if not self.hamtrainemails and not self.spamtrainemails:
            self.folderwalk(trainingset)

        # ham
        ham_train = self.hamtrainemails
        ham_test = self.hamtestemails

        self.Vham = set()
        self.hamwords = Counter()
        for record in ham_train:
            doc_vocab = self.email_vocabs(record["words"])
            self.Vham.update(doc_vocab)
            self.hamwords.update(record["words"])

        # spam
        spam_train = self.spamtrainemails
        spam_test = self.spamtestemails

        self.Vspam = set()
        self.spamwords = Counter()
        for record in spam_train:
            doc_vocab = self.email_vocabs(record["words"])
            self.Vspam.update(doc_vocab)
            self.spamwords.update(record["words"])

        # priors
        self.priordocs()

        print(f"[build_vocabs]  |Vham|={len(self.Vham):,}  "
              f"|Vspam|={len(self.Vspam):,}")

    # for test printing any email
    def print_email(self, category, number) -> None:
        if category.lower() == "ham":
            store = self.hamtrainemails
        elif category.lower() == "spam":
            store = self.spamtrainemails

        record = store[number]

        # update sample attributes
        self.rawemail   = record["raw"]
        self.cleanemail = record["clean"]
        self.hamorspam  = record["label"]

        print(f"  Sample Label       : {self.hamorspam.upper()}")
        print(f"\n  CLEANED EMAIL:")
        print(self.cleanemail)
        print("\n  RAW EMAIL:")
        print(self.rawemail)


def save_vocabs(vocab: VocabularyExtractor) -> None:
    path = "vocabs.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["word", "label", "count"])
        for word in vocab.Vham:
            writer.writerow([word, "ham", vocab.hamwords[word]])
        for word in vocab.Vspam:
            writer.writerow([word, "spam", vocab.spamwords[word]])
    print(f"[save_vocabs] saved {len(vocab.Vham):,} ham and {len(vocab.Vspam):,} spam words to {path}")


def load_vocabs(vocab: VocabularyExtractor) -> bool:
    path = "vocabs.csv"
    if not os.path.exists(path):
        print(f"[load_vocabs] no file found at {path} — run build_vocabs() first.")
        return False
    vocab.Vham.clear()
    vocab.Vspam.clear()
    vocab.hamwords.clear()
    vocab.spamwords.clear()
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            word  = row["word"]
            label = row["label"]
            count = int(row["count"])
            if label == "ham":
                vocab.Vham.add(word)
                vocab.hamwords[word] = count
            elif label == "spam":
                vocab.Vspam.add(word)
                vocab.spamwords[word] = count
    print(f"[load_vocabs] loaded {len(vocab.Vham):,} ham and {len(vocab.Vspam):,} spam words from {path}")
    return True
