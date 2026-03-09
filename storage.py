import pandas as pd


def save_model(vocabs, path="model.csv"):
    V = len(vocabs.Vham | vocabs.Vspam)
    sumham  = sum(vocabs.hamwords.values())
    sumspam = sum(vocabs.spamwords.values())

    # Metadata rows — stored as sentinel words for easy reconstruction
    metarows = [
        {"word": "__hamdocprior__",  "hamcount": vocabs.hamdocprior,  "spamcount": "",      "invham": "", "invspam": ""},
        {"word": "__spamdocprior__", "hamcount": vocabs.spamdocprior, "spamcount": "",      "invham": "", "invspam": ""},
        {"word": "__sumham__",       "hamcount": sumham,              "spamcount": "",      "invham": "", "invspam": ""},
        {"word": "__sumspam__",      "hamcount": "",                  "spamcount": sumspam, "invham": "", "invspam": ""},
        {"word": "__V__",            "hamcount": V,                   "spamcount": "",      "invham": "", "invspam": ""},
    ]

    # One row per word in the combined vocabulary
    allwords = sorted(set(vocabs.hamwords.keys()) | set(vocabs.spamwords.keys()))
    wordrows = [
        {
            "word":      word,
            "hamcount":  vocabs.hamwords.get(word, 0),
            "spamcount": vocabs.spamwords.get(word, 0),
            "invham":    int(word in vocabs.Vham),
            "invspam":   int(word in vocabs.Vspam),
        }
        for word in allwords
    ]

    df = pd.DataFrame(metarows + wordrows)
    df.to_csv(path, index=False)
    print(f"[savemodel] Saved {len(allwords):,} words + 5 metadata rows → {path}")


def save_results(allresults, path="results.csv"):
    df = pd.DataFrame(allresults)
    df["lambda"] = df["lambda"].map(lambda x: format(x, "g"))
    df = df[["lambda", "topn", "tp", "tn", "fp", "fn", "accuracy", "precision", "recall", "f1"]]
    df.to_csv(path, index=False)
    print(f"[saveresults] Saved {len(df)} result rows → {path}")
