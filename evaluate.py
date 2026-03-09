import matplotlib.pyplot as plt
import pandas as pd

from classifier import priors, denom, classify, updatecounts, Scores, topwords


# ── Single-email debug helper ────────────────────────────────────────────────

def printresult(record, vocabs, lambdaval, dham, dspam, threshold=0.5):
    label, pspam, pham, _ = classify(
        record["raw"], vocabs, lambdaval, dham, dspam,
        threshold, alreadyclean=False,
    )
    print(f"  True Label : {record['label'].upper()}")
    print(f"  Predicted  : {label.upper()}")
    print(f"  P(spam)    : {pspam:.4f}")
    print(f"  P(ham)     : {pham:.4f}")


# ── Lambda sweep (Section 6) ─────────────────────────────────────────────────

def run_lambda_sweep(training_vocabs, LAMBDA_VALUES, THRESHOLD=0.5, PROGRESS_EVERY=0):
    testemails = training_vocabs.alltestemails
    ytrue = [r["label"] for r in testemails]
    cleantexts = [r["clean"] for r in testemails]
    total = len(cleantexts)

    V = len(training_vocabs.Vham | training_vocabs.Vspam)

    allresults = []

    for lambdaval in LAMBDA_VALUES:
        dham, dspam = denom(training_vocabs, lambdaval)

        ypred = []
        counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "spam": 0, "ham": 0}
        print(f"\n[lambda={lambdaval}] Classifying {total:,} emails...")

        for idx, (cleantext, truelabel) in enumerate(zip(cleantexts, ytrue), start=1):
            predlabel, _, _, _ = classify(
                cleantext, training_vocabs, lambdaval, dham, dspam,
                threshold=THRESHOLD, alreadyclean=True,
            )
            ypred.append(predlabel)
            updatecounts(predlabel, truelabel, counts)

            if PROGRESS_EVERY and idx % PROGRESS_EVERY == 0 or len(ypred) == total:
                p = counts["tp"] / (counts["tp"] + counts["fp"]) if (counts["tp"] + counts["fp"]) > 0 else 0.0
                r = counts["tp"] / (counts["tp"] + counts["fn"]) if (counts["tp"] + counts["fn"]) > 0 else 0.0
                print(f"  Processed {idx:,}/{total:,}...  P: {p:.4f}  R: {r:.4f}")

        result = Scores(ytrue, ypred)
        topn = "---"
        allresults.append({
            "lambda": lambdaval,
            "topn": topn,  # for results
            "tp": result.tp, "tn": result.tn, "fp": result.fp, "fn": result.fn,
            "accuracy": result.accuracy,
            "precision": result.precision,
            "recall": result.recall,
            "f1": result.f1_score,
        })
        result.print_scores()

    print("\nDone processing all lambda values.")
    return allresults


# ── Topwords sweep (Section 7 improvements) ──────────────────────────────────

def run_topwords_sweep(training_vocabs, allresults, topn, LAMBDAVAL=0.005, THRESHOLD=0.5, PROGRESS_EVERY=0):
    testemails = training_vocabs.alltestemails
    ytrue = [r["label"] for r in testemails]
    cleantexts = [r["clean"] for r in testemails]
    total = len(cleantexts)

    hovoldresults = allresults

    for x in topn:
        hovold_vocabs, removedwords = topwords(x, training_vocabs)  # x not topn, removedwords not topwords
        print(f"[topwords] Filtered words: {removedwords}")

        pham, pspam = priors(hovold_vocabs)
        dham, dspam = denom(hovold_vocabs, LAMBDAVAL)
        print(f"P(ham) new = {pham:.4f}  P(spam) new = {pspam:.4f}")
        print(f"D(ham) new = {dham}  D(spam) new = {dspam}")

        ypred = []
        counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "spam": 0, "ham": 0}
        print(f"\n[topn={x}] Classifying {total:,} emails...")  # x not topn

        for idx, (cleantext, truelabel) in enumerate(zip(cleantexts, ytrue), start=1):
            predlabel, _, _, _ = classify(
                cleantext, hovold_vocabs, LAMBDAVAL, dham, dspam,
                threshold=THRESHOLD, alreadyclean=True,
            )
            ypred.append(predlabel)
            updatecounts(predlabel, truelabel, counts)

            if PROGRESS_EVERY and idx % PROGRESS_EVERY == 0 or len(ypred) == total:
                p = counts["tp"] / (counts["tp"] + counts["fp"]) if (counts["tp"] + counts["fp"]) > 0 else 0.0
                r = counts["tp"] / (counts["tp"] + counts["fn"]) if (counts["tp"] + counts["fn"]) > 0 else 0.0
                print(f"  Processed {idx:,}/{total:,}...  P: {p:.4f}  R: {r:.4f}")

        result = Scores(ytrue, ypred)
        hovoldresults.append({
            "lambda": LAMBDAVAL,
            "topn": x,  # x not topn
            "tp": result.tp, "tn": result.tn, "fp": result.fp, "fn": result.fn,
            "accuracy": result.accuracy,
            "precision": result.precision,
            "recall": result.recall,
            "f1": result.f1_score,
        })
        result.print_scores()

    print(f"Done processing test emails with topn values of {topn}.\n")
    return hovoldresults


# ── Plotting helpers ─────────────────────────────────────────────────────────

def tablegraph(allresults):
    df = pd.DataFrame(allresults)
    df["lambda"] = df["lambda"].map(lambda x: format(x, "g"))

    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_columns", None)
    print(df[["lambda", "topn", "tp", "tn", "fp", "fn", "accuracy", "precision", "recall", "f1"]].to_markdown(index=False))
    bestidx = df["f1"].idxmax()
    best = df.loc[bestidx]
    print(
        f"\nBest by F1: lambda={best['lambda']} Top-n={best['topn']}  "
        f"P={best['precision']:.4f}  R={best['recall']:.4f}  F1={best['f1']:.4f}"
    )
    return df


def plotgraph(df):
    bestidx = df["f1"].idxmax()  # recompute from df — not in scope from tablegraph
    x = list(range(len(df)))

    plt.figure(figsize=(10, 6))
    plt.plot(x, df["accuracy"],  marker="o", linewidth=2, label="Accuracy")
    plt.plot(x, df["precision"], marker="o", linewidth=2, label="Precision")
    plt.plot(x, df["recall"],    marker="o", linewidth=2, label="Recall")
    plt.plot(x, df["f1"],        marker="o", linewidth=2, label="F1 Score")
    plt.scatter([bestidx], [df.loc[bestidx, "f1"]], color="red", zorder=5, label="Best F1")

    plt.xticks(x, df["lambda"])
    plt.xlabel("Lambda")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.title("Naive Bayes Metrics vs Lambda")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
