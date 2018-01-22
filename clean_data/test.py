import pandas as pd

df = pd.read_csv("./cleaned/finestse/finestse.tsv", sep='\t',encoding = 'utf8', header=None, names=["docname", "docstr"])

for index, row in df.iterrows():
    print row

def findStyleMatches(doc, t):
    fashion_docs = []
    for index, row in style_df.iterrows():
        fashion_docs.append(nlp(row["word"]))
    tokens = allTokens(fashion_docs)
    #Remove duplicate tokens with equal text
    texts = set()
    tokens2 = []
    for token in tokens:
        if not token.text in texts:
            tokens2.append(token)
        texts.add(token.text)
    tokens = tokens2
    fashionLemmas = []
    for token in tokens:
        for lemmaToken in nlp(token.lemma_):
            fashionLemmas.append(lemmaToken)
    # Filter out equal lemmas
    lemmasText = set()
    temp = []
    for token in fashionLemmas:
        if token.text not in lemmasText:
            temp.append(token)
        lemmasText.add(token.text)
    fashionLemmas = temp
    match = []
    for token in doc:
        match = match + similar(token, fashionLemmas, t)
    return len(match)