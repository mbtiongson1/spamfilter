import os
import csv

from config import source

# Module-level globals — populated by load_labels()
ham = set()
spam = set()
tests = set()  # iii only


def load_labels():
    global ham, spam, tests

    labelspath = os.path.join(source, 'labels')

    if os.path.exists(labelspath):
        with open(labelspath, 'r') as f:
            for line in f:
                # ['ham', '../data/000/000']
                parts = line.strip().split()
                if len(parts) == 2:
                    label = parts[0]
                    path_str = parts[1]

                    # '../data/000/000'
                    partsplit = path_str.split('/')
                    iii = partsplit[-2]  # folder index
                    jjj = partsplit[-1]  # file index

                    address = (iii, jjj)  # tuple
                    tests.add(iii)        # add folder
                    if label == 'ham':
                        ham.add(address)
                    elif label == 'spam':
                        spam.add(address)

        limit = 5
        hamfiles = list(ham)
        spamfiles = list(spam)
        tests_sorted = sorted(tests)
        tests.clear()
        tests.update(tests_sorted)

        print(f"Total Tests (folders): {len(tests)}")
        print(f"")
        print(f"Total Ham entries: {len(ham)}")
        print("Ex:", hamfiles[:limit], "..." if len(hamfiles) > limit else "")
        print(f"")
        print(f"Total Spam entries: {len(spam)}")
        print("Ex:", spamfiles[:limit], "..." if len(spamfiles) > limit else "")
    else:
        print(f"Labels file not found at {labelspath}")


class TrainingSplit:
    def __init__(self, fulltest):
        self.full = fulltest
        # 70/30 split
        first70 = int(len(fulltest) * 0.7)
        self.trainingset = fulltest[:first70]
        self.testingset = fulltest[first70:]
        # folder sets
        trainingfolders = set(self.trainingset)
        testingfolders = set(self.testingset)
        # spam ham splits
        self.hamtraining  = [(f, fi) for f, fi in ham  if f in trainingfolders]
        self.spamtraining = [(f, fi) for f, fi in spam if f in trainingfolders]
        self.hamtesting   = [(f, fi) for f, fi in ham  if f in testingfolders]
        self.spamtesting  = [(f, fi) for f, fi in spam if f in testingfolders]
        # full list with iii,jjj
        self.fulltraining = self.hamtraining + self.spamtraining
        self.fulltesting  = self.hamtesting  + self.spamtesting


def load_or_create_split(tests):
    filename = 'test.csv'

    if os.path.exists(filename):
        print(f"Loaded from {filename}")
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                fullset = row[1:]
                split = TrainingSplit(fullset)
    else:
        print(f"{filename} not found. Generating new test file...")
        split = TrainingSplit(tests)
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['0'] + split.full)
            writer.writerow(['1'] + split.trainingset)
            writer.writerow(['2'] + split.testingset)
        print("Split saved to test.csv.")

    print("Training set:", len(split.trainingset), split.trainingset[:5], "...")
    print("Testing set:", len(split.testingset), split.testingset[:5], "...")
    return split


def print_split_summary(trec06):
    spamperc = len(trec06.spamtraining) / len(trec06.fulltraining) * 100
    hamperc  = len(trec06.hamtraining)  / len(trec06.fulltraining) * 100

    print(f"Training spam % = {spamperc:.2f}%")
    print(f"Training ham %  = {hamperc:.2f}%")
    print(f"Training total: {len(trec06.fulltraining)} emails ({len(trec06.trainingset)} folders)")
    print(f"Testing total:  {len(trec06.fulltesting)} emails ({len(trec06.testingset)} folders)")
