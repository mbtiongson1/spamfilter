"""
main.py
Runs the full Naive Bayes spam classifier pipeline,
equivalent to executing the entire notebook top-to-bottom.

Usage:
    python main.py
"""

# ── 1. Config ────────────────────────────────────────────────────────────────
from config import source, show_source

show_source()

# ── 2. Dataset: load labels and build train/test split ───────────────────────
import dataset

dataset.load_labels()

tests = sorted(dataset.tests)
print(tests)

from dataset import TrainingSplit, load_or_create_split, print_split_summary

split = load_or_create_split(tests)
trec06 = split

print_split_summary(trec06)

# ── 3. Preprocessing: parse + clean (imported for use downstream) ────────────
from preprocessing import parse, clean  # noqa: F401 — used inside vocabulary/classifier

# ── 4. Vocabulary: build, inspect, save ─────────────────────────────────────
from vocabulary import VocabularyExtractor, save_vocabs, load_vocabs

training_vocabs = VocabularyExtractor(source, trec06)
training_vocabs.build_vocabs(trec06)
# will take a bit of time depending on runtime speed

# test print
print("Ham:", training_vocabs.Vham)
print("Ham Prior:", training_vocabs.hamdocprior)
print("\nSpam:", training_vocabs.Vspam)
print("Spam Prior:", training_vocabs.spamdocprior)
print(f'\nUnique Ham Words: {len(training_vocabs.Vham)}')
print(f'Unique Spam Words: {len(training_vocabs.Vspam)}')

# print email samples
training_vocabs.print_email("spam", 2)
training_vocabs.print_email("ham", 3)

# save vocabs
save_vocabs(training_vocabs)
# load_vocabs(training_vocabs)  # only run this if you already have vocabs.csv!

# ── 5. Classifier: verify priors and denominators ────────────────────────────
from classifier import priors, denom, classify, topwords

pham, pspam = priors(training_vocabs)
dham, dspam = denom(training_vocabs, lambda_val=0)

print(f"P(ham)={pham:.4f}  P(spam)={pspam:.4f}")
print(f"D(ham)={dham}  D(spam)={dspam}")

# ── 6. Single-email smoke test ───────────────────────────────────────────────
from evaluate import printresult

LAMBDA_VAL = 1
THRESHOLD  = 0.5  # just for testing...

dham, dspam = denom(training_vocabs, LAMBDA_VAL)

printresult(training_vocabs.hamtestemails[3400],  training_vocabs, LAMBDA_VAL, dham, dspam, THRESHOLD)
print()
printresult(training_vocabs.spamtestemails[6502], training_vocabs, LAMBDA_VAL, dham, dspam, THRESHOLD)

# ── 7. Lambda sweep evaluation ───────────────────────────────────────────────
from evaluate import run_lambda_sweep, run_topwords_sweep, tablegraph, plotgraph

LAMBDA_VALUES = [0, 2.0, 1.0, 0.5, 0.1, 0.005]  # put all lambdas
THRESHOLD = 0.5       # for testing...
PROGRESS_EVERY = 0    # for printing progress

allresults = run_lambda_sweep(training_vocabs, LAMBDA_VALUES, THRESHOLD, PROGRESS_EVERY)

preliminary = tablegraph(allresults)
plotgraph(preliminary)

# ── 8. Topwords sweep (Hovold improvement) ───────────────────────────────────
topn = [200, 100]
LAMBDAVAL = 0.005  # fixed
THRESHOLD = 0.5    # for testing
PROGRESS_EVERY = 0  # for testing

hovoldresults = run_topwords_sweep(training_vocabs, allresults, topn, LAMBDAVAL, THRESHOLD, PROGRESS_EVERY)

finalgraph = tablegraph(hovoldresults)
plotgraph(finalgraph)

hovold_vocabs100, removedwords100 = topwords(100, training_vocabs)
hovold_vocabs200, removedwords200 = topwords(200, training_vocabs)

# ── 9. Save model and results ────────────────────────────────────────────────
from storage import save_model, save_results

save_model(training_vocabs)
save_results(hovoldresults, path="results.csv")
