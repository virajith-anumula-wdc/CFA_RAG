from sentence_transformers import SentenceTransformer
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

sen = ["Word2Vec uses only complete words found in the training corpus to learn vectors. In contrast, FastText learns vectors for individual words and the n-grams found within them. The mean of the target word vector and its n-gram component vectors are used for training at each stage of the FastText process. Each combined vector creates the target, which is then uniformly updated using the adjustment calculated from the error. These calculations significantly increase the amount of computation in the training phase. A word must add up and average its n-gram parts at each point. Through various metrics, it has been demonstrated that these vectors are more accurate than Word2Vec vectors. The most notable enhancement to FastText is the N-gram feature, which addresses the OOV (out-of-vocabulary) problem. For instance, the word “aquarium” can be broken down into “aq/aqu/qua/uar/ari/riu/ium/um>,” where “<” and “>” denote the beginning and end of the word, respectively. Though the word embedder may not immediately recognise the word “Aquarius,” it can infer its meaning. This can be done because the words “aquarium” and “Aquarius” share a common root."]

em = model.encode(sen)
print(em)
