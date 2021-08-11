import itertools
import json
from urllib import parse, request

import neuralcoref
import nltk
import opennre
import pandas as pd
import spacy

# !pip install neuralcoref
# !pip install git+https://github.com/thunlp/OpenNRE

nltk.download("punkt")


class SpacyModelError(Exception):
    pass


class TextToKnowledge:
    """Extract knowledge from raw text using coreference resolution and relation extraction.

    This class is a copy-paste (with some esthetics updates) from Tomaz Bratanic blog post.

    Parameters
    ----------
        file: Knowledge base output file.
        key: Wikifer API access key.
        threshold: Threshold associated with Wikifer API.

    Example
    -------

    >>> from txtokb import pipeline

    >>> types = ["human", "person", "company", "enterprise", "business", "geographic region",
    ...    "human settlement", "geographic entity", "territorial entity type", "organization"]

    >>> pipeline = pipeline.TextToKnowledge(key="jueidnxsctiurpwykpumtsntlschpx", types=types)

    >>> text = "Elon Musk is a business magnate, industrial designer, and engineer. He is the founder, CEO, CTO, and chief designer of SpaceX. He is also early investor, CEO, and product architect of Tesla, Inc. He is also the founder of The Boring Company and the co-founder of Neuralink. A centibillionaire, Musk became the richest person in the world in January 2021, with an estimated net worth of $185 billion at the time, surpassing Jeff Bezos. Musk was born to a Canadian mother and South African father and raised in Pretoria, South Africa. He briefly attended the University of Pretoria before moving to Canada aged 17 to attend Queen's University. He transferred to the University of Pennsylvania two years later, where he received dual bachelor's degrees in economics and physics. He moved to California in 1995 to attend Stanford University, but decided instead to pursue a business career. He went on co-founding a web software company Zip2 with his brother Kimbal Musk."

    >>> pipeline.process_sentence(text = text):


    References
    ----------
        1. [Tomaz Bratanic. From Text to Knowledge: The Information Extraction Pipeline.](https://towardsdatascience.com/from-text-to-knowledge-the-information-extraction-pipeline-b65e7e30273e)
        2. [Wikifier API documentation.](http://wikifier.org/info.html)
        3. [Wikifier API registration.](http://wikifier.org/register.html)

    """

    def __init__(self, key: str, types=[]):
        self.key = key

        try:
            self.nlp = spacy.load("en")  # Only available in french
        except OSError:
            raise SpacyModelError()

        neuralcoref.add_to_pipe(self.nlp)

        self.model = opennre.get_model("wiki80_cnn_softmax")

        self.types = types

    def coreference(self, text: str):
        """Coreferences resolution using Neuralcoref and Spacy.

        Parameters
        ----------
            text: Text on which the coreference resolution is applied.

        Example
        -------

        >>> from txtokb import pipeline

        >>> pipeline = pipeline.TextToKnowledge(key="jueidnxsctiurpwykpumtsntlschpx")

        >>> pipeline.coreference(text = "I like Netflix, it contains great series."):
        "I like Netflix, Netflix contains great series.""

        References
        ----------
            1. [grovershreyf9t. Repetition of Named Entity in coref resolution](https://github.com/huggingface/neuralcoref/issues/288)
            2. [Tomaz Bratanic. From Text to Knowledge: The Information Extraction Pipeline.](https://towardsdatascience.com/from-text-to-knowledge-the-information-extraction-pipeline-b65e7e30273e)

        """
        doc = self.nlp(text)
        tokens = [token.text_with_ws for token in doc]
        for cluster in doc._.coref_clusters:
            cluster_main_words = set(cluster.main.text.split(" "))
            for coref in cluster:
                if coref != cluster.main:
                    if coref.text != cluster.main.text and not set(
                        coref.text.split(" ")
                    ).intersection(cluster_main_words):
                        tokens[coref.start] = cluster.main.text + doc[coref.end - 1].whitespace_
                        for i in range(coref.start + 1, coref.end):
                            tokens[i] = ""
        return "".join(tokens)

    def wikipedia_el(self, text: str):
        """Entity linking using wikifier.

        References
        ----------
          1. [Wikifier API documentation.](http://wikifier.org/info.html)
          2. [Tomaz Bratanic. From Text to Knowledge: The Information Extraction Pipeline.](https://towardsdatascience.com/from-text-to-knowledge-the-information-extraction-pipeline-b65e7e30273e)

        """
        data = parse.urlencode(
            [
                ("text", text),
                ("lang", "en"),
                ("userKey", self.key),
                ("pageRankSqThreshold", "%g" % 0.8),
                ("applyPageRankSqThreshold", "true"),
                ("wikiDataClasses", "true"),
                ("wikiDataClassIds", "false"),
                ("support", "true"),
                ("ranges", "false"),
                ("nTopDfValuesToIgnore", "100"),
                ("nWordsToIgnoreFromList", "100"),
                ("minLinkFrequency", "1"),
                ("includeCosines", "false"),
                ("maxMentionEntropy", "1"),
            ]
        )

        response = request.Request(
            "http://www.wikifier.org/annotate-article", data=data.encode("utf8"), method="POST"
        )

        with request.urlopen(response, timeout=60) as f:
            response = f.read()
            response = json.loads(response.decode("utf8"))

        entities = []

        if "annotations" in response:

            for entity in response["annotations"]:

                if self.types:

                    if not entity["wikiDataClasses"]:
                        continue

                    if not any([t["enLabel"] in self.types for t in entity["wikiDataClasses"]]):
                        continue

                entities.append(
                    {
                        "title": entity["secTitle"],
                        "url": entity["secUrl"],
                        "id": entity["wikiDataItemId"],
                        "types": entity["dbPediaTypes"],
                        "positions": [
                            (position["chFrom"], position["chTo"])
                            for position in entity["support"]
                        ],
                    }
                )

        return entities

    def relation_extraction(self, text: str, entities: list):
        """Knowledge extraction from text using OpenNRE.

        References
        ----------
            1. [Xu et, al. {O}pen{NRE}: An Open and Extensible Toolkit for Neural Relation Extraction](https://github.com/thunlp/OpenNRE)
            2. [Tomaz Bratanic. From Text to Knowledge: The Information Extraction Pipeline.](https://towardsdatascience.com/from-text-to-knowledge-the-information-extraction-pipeline-b65e7e30273e)

        """
        triples = []

        for permutation in itertools.permutations(entities, 2):
            for source in permutation[0]["positions"]:
                for target in permutation[1]["positions"]:
                    data = self.model.infer(
                        {
                            "text": text,
                            "h": {"pos": [source[0], source[1] + 1]},
                            "t": {"pos": [target[0], target[1] + 1]},
                        }
                    )

                    triples.append(
                        {
                            "head": permutation[0]["title"],
                            "relation": data[0],
                            "tail": permutation[1]["title"],
                            "score": data[1],
                        }
                    )
        return pd.DataFrame(triples).drop_duplicates(keep="first")

    def process_sentence(self, text: str):
        triples = []
        text = self.coreference(text=text)
        entities = self.wikipedia_el(text=text)
        if len(entities) > 1:
            triples.append(self.relation_extraction(text=text, entities=entities))
            return (
                pd.concat(triples)
                .sort_values(by="score", ascending=False)
                .drop_duplicates(keep="first", subset=["head", "tail"])
                .reset_index(drop=True)
            )
        return pd.DataFrame(columns=["head", "relation", "tail", "score"])
