import math
import copy

from preprocessing import parse, clean


# Priors
def priors(vocabs):  # just for mathing
    pham = vocabs.hamdocprior
    pspam = vocabs.spamdocprior
    return pham, pspam


# Denominator
def denom(vocabs, lambda_val):  # just for mathing
    V = len(vocabs.Vham | vocabs.Vspam)

    dham = sum(vocabs.hamwords.values()) + lambda_val * V
    dspam = sum(vocabs.spamwords.values()) + lambda_val * V

    return dham, dspam


def likelihood(wordcount, lambdaval, denomval):  # returns log
    if lambdaval == 0 and wordcount == 0:  # no smoothing
        return None
    like = (wordcount + lambdaval) / denomval
    return math.log(like) if like > 0.0 else None


# main classify function to decide if email is ham or spam
def classify(text, vocabs, lambdaval, dham, dspam, threshold=0.5, alreadyclean=False):
    if alreadyclean:
        allwords = text.split()
    else:
        text = clean(parse(text))
        allwords = text.split()

    loghamscore = math.log(vocabs.hamdocprior)
    logspamscore = math.log(vocabs.spamdocprior)
    # to prevent inf errors
    hamvalid = dham > 0
    spamvalid = dspam > 0

    for word in allwords:
        if not hamvalid and not spamvalid:
            break

        if hamvalid:
            hamll = likelihood(vocabs.hamwords.get(word, 0), lambdaval, dham)
            if hamll is None:
                hamvalid = False
            else:
                loghamscore += hamll

        if spamvalid:
            spamll = likelihood(vocabs.spamwords.get(word, 0), lambdaval, dspam)
            if spamll is None:
                spamvalid = False
            else:
                logspamscore += spamll

    if hamvalid and spamvalid:
        maxscore = max(loghamscore, logspamscore)
        expham = math.exp(loghamscore - maxscore)
        expspam = math.exp(logspamscore - maxscore)
        exptotal = expham + expspam
        pham = expham / exptotal if exptotal > 0 else 0.5
        pspam = expspam / exptotal if exptotal > 0 else 0.5
    elif hamvalid:
        pham, pspam = 1.0, 0.0
    elif spamvalid:
        pham, pspam = 0.0, 1.0
    else:
        pham, pspam = 0.5, 0.5

    label = "ham" if pham >= pspam else "spam"  # default ham if 0.5
    return label, pspam, pham, text


# Scores, conf matrix
def updatecounts(predlabel, truelabel, counts):  # for each email
    if   predlabel == "spam" and truelabel == "spam": counts["tp"] += 1; counts["spam"] += 1
    elif predlabel == "ham"  and truelabel == "spam": counts["fn"] += 1; counts["spam"] += 1
    elif predlabel == "spam" and truelabel == "ham":  counts["fp"] += 1; counts["ham"]  += 1
    else:                                             counts["tn"] += 1; counts["ham"]  += 1


class Scores:
    def __init__(self, y_true: list, y_pred: list):
        self.tp = 0  # predicted ham,  actually ham  ✓
        self.tn = 0  # predicted spam, actually spam ✓
        self.fp = 0  # predicted ham,  actually spam ✗
        self.fn = 0  # predicted spam, actually ham  ✗

        self.spam = 0  # actual spam count
        self.ham  = 0  # actual ham count

        # for the whole list
        for true, pred in zip(y_true, y_pred):
            if   pred == "spam" and true == "spam":
                self.tp += 1;  self.spam += 1  # yay! spam
            elif pred == "ham"  and true == "spam":
                self.fn += 1;  self.spam += 1  # missed spam :(
            elif pred == "spam" and true == "ham":
                self.fp += 1;  self.ham  += 1  # wrong!
            elif pred == "ham"  and true == "ham":
                self.tn += 1;  self.ham  += 1  # yay! ham

        self.accuracy  = self._accuracy()
        self.precision = self._precision()
        self.recall    = self._recall()
        self.f1_score  = self._f1_score()

    def _accuracy(self) -> float:
        total = self.tp + self.tn + self.fp + self.fn
        return (self.tp + self.tn) / total if total > 0 else 0.0

    def _precision(self) -> float:
        denom = self.tp + self.fp  # this is what we want!
        return self.tp / denom if denom > 0 else 0.0

    def _recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    def _f1_score(self) -> float:
        denom = self.precision + self.recall
        return (2 * self.precision * self.recall) / denom if denom > 0 else 0.0

    def print_scores(self) -> None:
        print(f"\n True (Ham)  : {self.ham:,}")
        print(f" True (Spam) : {self.spam:,}")
        print(f"\n  TP={self.tp:,}  TN={self.tn:,}\n  FP={self.fp:,}  FN={self.fn:,}")
        print(f"  Accuracy  : {self.accuracy:.4f}")
        print(f"  Precision : {self.precision:.4f}")
        print(f"  Recall    : {self.recall:.4f}")
        print(f"  F1 Score  : {self.f1_score:.4f}")


def topwords(topn, vocabs, protectedwords=None):
    defaultprotected = {"url", "emailaddr", "multiexclaim", "price", "num", "attachment"}
    if protectedwords is None:
        protectedwords = defaultprotected

    filteredvocabs = copy.copy(vocabs)
    filteredvocabs.hamwords  = copy.copy(vocabs.hamwords)
    filteredvocabs.spamwords = copy.copy(vocabs.spamwords)
    filteredvocabs.Vham      = copy.copy(vocabs.Vham)
    filteredvocabs.Vspam     = copy.copy(vocabs.Vspam)

    combined = filteredvocabs.hamwords + filteredvocabs.spamwords
    candidates = [
        w for w, _ in combined.most_common()
        if w not in protectedwords
    ]
    removedwords = set(candidates[:topn])  # renamed — no longer shadows the function

    filteredvocabs.Vham  -= removedwords
    filteredvocabs.Vspam -= removedwords
    for word in removedwords:
        filteredvocabs.hamwords.pop(word, None)
        filteredvocabs.spamwords.pop(word, None)

    # print(f"[topwords] Removed {len(removedwords)} words from whole vocabulary.")
    # print(f"[topwords] Protected words: {protectedwords}")
    print(f"[topwords] Refined |Vham|={len(filteredvocabs.Vham):,}  |Vspam|={len(filteredvocabs.Vspam):,}")
    print(f"[topwords] Filtered words: {removedwords}\n")

    return filteredvocabs, removedwords
