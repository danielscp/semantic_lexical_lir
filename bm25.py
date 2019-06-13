import math

def compute_df(bows):
    df = {}
    for b in bows:
        for w in b:
            if not w[0] in df:
                df[w[0]] = 0
            df[w[0]] += 1
    return df

def compute_idf(df, corpus_size):
    idf = {}
    for w in df.keys():
        freq = df[w]
        idf[w] = math.log(corpus_size - freq + 0.5) - math.log(freq + 0.5)
    return idf

def doc_len(bow):
    return sum([float(c[1]) for c in bow])

def bow2dict(bow):
    f = {}
    for b in bow:
        f[b[0]]=b[1]
    return f


class BM25:

    def __init__(self, bows, K1 = 1.5, B = 0.75, EPSILON=0.25):
        self.K1 = K1
        self.B = B
        self.EPSILON = EPSILON
        self.fit(bows)

    def fit(self, bows):
        self.corpus_size = len(bows)
        self.docs_len = [doc_len(x) for x in bows]
        self.avgdl = sum(self.docs_len) / self.corpus_size
        self.f = [bow2dict(bow) for bow in bows]
        self.df = compute_df(bows)
        self.idf = compute_idf(self.df, self.corpus_size)
        self.average_idf = sum(float(val) for val in self.idf.values()) / len(self.idf)

    def score(self, bow, index):

        score = 0
        for word in bow:
            w = word[0]  # word index in dictionary
            if w not in self.f[index]:
                continue
            idf = self.idf[w] if self.idf[w] >= 0 else self.EPSILON * self.average_idf
            score += (idf * self.f[index][w] * (self.K1 + 1)
                      / (self.f[index][w] + self.K1 * (
                            1 - self.B + self.B * self.docs_len[index] / self.avgdl)))
        return score

    def similarities(self, bow):
        sims = [(i, self.score(bow, i)) for i in range(self.corpus_size)]
        return sims
