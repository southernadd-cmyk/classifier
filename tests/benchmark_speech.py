"""Reproducible, validation-only speech model search on held-out MPs.
This is an experiment, not the classroom browser implementation.
Test accuracy is computed exactly once after selecting a validation winner.
"""
import json
import math
import re
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parent.parent
rows = json.loads((ROOT / "data/commons-speech-sample.json").read_text())["examples"]
labels = ["Labour", "Conservative"]
state = 20261119
def rand():
    global state
    state = (1664525 * state + 1013904223) & 0xffffffff
    return state / 4294967296
def shuffle(items):
    a = list(items)
    for i in range(len(a)-1, 0, -1):
        j = int(rand() * (i+1))
        a[i], a[j] = a[j], a[i]
    return a

speakers = {}
for row in rows:
    speakers.setdefault(str(row["mp"]), []).append(row)
groups = {label: [] for label in labels}
for speaker, records in speakers.items():
    labour = sum(r["party"] == "Labour" for r in records)
    primary = "Labour" if labour >= len(records) - labour else "Conservative"
    groups[primary].append((speaker, records))
parts = {"train": [], "validation": [], "test": []}
for label in labels:
    shuffled = shuffle(groups[label])
    n = max(1, round(len(shuffled) * .15))
    for index, (_, records) in enumerate(shuffled):
        group = "test" if index < n else "validation" if index < n + n else "train"
        parts[group].extend(records)
for group in parts:
    parts[group] = shuffle(parts[group])
assert [len(parts[x]) for x in ("train","validation","test")] == [10300,2029,2071]
sets = {k:set(str(r["mp"]) for r in v) for k,v in parts.items()}
assert not (sets["train"] & sets["validation"] or sets["train"] & sets["test"] or sets["validation"] & sets["test"])

# Mask explicit party labels before any word or character n-gram creation.
party_pattern = re.compile(r"(?i)\b(?:labour|conservatives?|tories|tory|lab|con)(?:['’]s)?\b")
def clean(t):
    return party_pattern.sub(" ", str(t)).lower()
docs = {k: [clean(r["text"]) for r in v] for k,v in parts.items()}
y = {k: np.array([r["party"] == "Labour" for r in v], dtype=np.int32) for k,v in parts.items()}
assert not any(party_pattern.search(t) for values in docs.values() for t in values)
print("SPLIT", {k:len(v) for k,v in docs.items()},flush=True)
print("CLASS_BALANCE", {k:int(sum(v)) for k,v in y.items()},flush=True)

representations = {}
for name,vectorizer in [
    ("word1-2",TfidfVectorizer(ngram_range=(1,2),min_df=3,max_features=75000,sublinear_tf=True,strip_accents="unicode")),
    ("word1-3",TfidfVectorizer(ngram_range=(1,3),min_df=3,max_features=100000,sublinear_tf=True,strip_accents="unicode")),
    ("char3-5",TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=4,max_features=85000,sublinear_tf=True,strip_accents="unicode")),
]:
    train = vectorizer.fit_transform(docs["train"])
    validation = vectorizer.transform(docs["validation"])
    representations[name]=(train,validation,vectorizer)
    print("FEATURES",name,train.shape,train.nnz,flush=True)

representations["word+char"] = (
    hstack([representations["word1-2"][0],representations["char3-5"][0]],format="csr"),
    hstack([representations["word1-2"][1],representations["char3-5"][1]],format="csr"),
    None
)
best={"acc":0}
trials=[]
def trial(name,algo,train,val,make):
    global best
    mdl = make()
    mdl.fit(train,y["train"])
    guessed = mdl.predict(val)
    acc = accuracy_score(y["validation"],guessed)
    trials.append((acc,name,algo))
    if acc>best["acc"]:
        best={"acc":acc,"name":name,"algo":algo}
    print(f"VALIDATION {name:10s} {algo:15s}: {acc:.4%}",flush=True)

for name,(tr,val,vec) in representations.items():
    for C in (0.03,0.1,0.3,1,3):
        trial(name,f"LinearSVC C={C}",tr,val,lambda C=C:LinearSVC(C=C,max_iter=2500,dual="auto"))
    if name in ("word1-2","char3-5"):
        for alpha in (.2,1.0):
            trial(name,f"CompNB a={alpha}",tr,val,lambda alpha=alpha:ComplementNB(alpha=alpha))

# NB-SVM: logistic-like linear margins on log-count-ratio-weighted binary ngrams.
cv=CountVectorizer(ngram_range=(1,2),binary=True,min_df=3,max_features=75000)
cx=cv.fit_transform(docs["train"]).tocsr()
vx=cv.transform(docs["validation"]).tocsr()
positive=np.asarray(cx[y["train"]==1].sum(axis=0)).ravel()+1
negative=np.asarray(cx[y["train"]==0].sum(axis=0)).ravel()+1
ratio=np.log(positive/positive.sum())-np.log(negative/negative.sum())
rx=cx.multiply(ratio).tocsr()
rv=vx.multiply(ratio).tocsr()
for C in (.03,.1,.3,1,3):
    trial("NB-SVM",f"LinearSVC C={C}",rx,rv,lambda C=C:LinearSVC(C=C,max_iter=2500,dual="auto"))

print("BEST VALIDATION",best,flush=True)
print("ALL_TRIALS",sorted(trials,reverse=True),flush=True)
# Single isolated test run with only the chosen hyperparameters.
name,algorithm=best["name"],best["algo"]
if name=="NB-SVM":
    selectedTrain=rx
    selectedTest=cv.transform(docs["test"]).multiply(ratio).tocsr()
else:
    if name=="word+char":
        selectedTrain=representations[name][0]
        selectedTest=hstack([
            representations["word1-2"][2].transform(docs["test"]),
            representations["char3-5"][2].transform(docs["test"])
        ],format="csr")
    else:
        selectedTrain=representations[name][0]
        selectedTest=representations[name][2].transform(docs["test"])
if algorithm.startswith("LinearSVC"):
    chosen=LinearSVC(C=float(algorithm.split("=")[-1]),max_iter=2500,dual="auto")
else:
    chosen=ComplementNB(alpha=float(algorithm.split("=")[-1]))
chosen.fit(selectedTrain,y["train"])
pred=chosen.predict(selectedTest)
print("HELD_OUT_TEST",name,algorithm,round(accuracy_score(y["test"],pred),5),
      int(sum(pred==y["test"])),len(pred),flush=True)
