import itertools
import json
from urllib import parse, request

import neuralcoref
import nltk
import opennre
import pandas as pd
import spacy
import torch
from transformers import LukeForEntityPairClassification, LukeTokenizer

# !pip install neuralcoref
# !pip install git+https://github.com/thunlp/OpenNRE

nltk.download("punkt")


__all__ = ["TextToKnowledge"]


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

    >>> from textokb import pipeline

    >>> types = ["human", "person", "company", "enterprise", "business", "geographic region",
    ...    "human settlement", "geographic entity", "territorial entity type", "organization"]

    >>> model = pipeline.TextToKnowledge(key="jueidnxsctiurpwykpumtsntlschpx", types=types)

    >>> text = "Elon Musk is a business magnate, industrial designer, and engineer. He is the founder, CEO, CTO, and chief designer of SpaceX. He is also early investor, CEO, and product architect of Tesla, Inc. He is also the founder of The Boring Company and the co-founder of Neuralink. A centibillionaire, Musk became the richest person in the world in January 2021, with an estimated net worth of $185 billion at the time, surpassing Jeff Bezos. Musk was born to a Canadian mother and South African father and raised in Pretoria, South Africa. He briefly attended the University of Pretoria before moving to Canada aged 17 to attend Queen's University. He transferred to the University of Pennsylvania two years later, where he received dual bachelor's degrees in economics and physics. He moved to California in 1995 to attend Stanford University, but decided instead to pursue a business career. He went on co-founding a web software company Zip2 with his brother Kimbal Musk."

    >>> model.process_sentence(text = text)
                              head       relation                        tail     score
    0                  Tesla, Inc.      architect                   Elon Musk  0.803398
    1                  Tesla, Inc.  field of work          The Boring Company  0.733903
    2                    Elon Musk      residence  University of Pennsylvania  0.648434
    3                    Elon Musk  field of work          The Boring Company  0.592007
    4                    Elon Musk   manufacturer                 Tesla, Inc.  0.553206
    5           The Boring Company   manufacturer                 Tesla, Inc.  0.515352
    6                    Elon Musk      developer                 Kimbal Musk  0.475639
    7   University of Pennsylvania     subsidiary                   Elon Musk  0.435384
    8           The Boring Company      developer                   Elon Musk  0.387753
    9                       SpaceX         winner                   Elon Musk  0.374090
    10                 Kimbal Musk        sibling                   Elon Musk  0.355944
    11                   Elon Musk   manufacturer                      SpaceX  0.221294


    >>> model = pipeline.TextToKnowledge(key="jueidnxsctiurpwykpumtsntlschpx", types=types, luke=True)

    >>> model.process_sentence(text = text)
                     head              relation                        tail      score
    0           Elon Musk          per:siblings                 Kimbal Musk  10.436224
    1         Kimbal Musk          per:siblings                   Elon Musk  10.040980
    2           Elon Musk  per:schools_attended  University of Pennsylvania   9.808870
    3  The Boring Company        org:founded_by                   Elon Musk   8.823962
    4           Elon Musk       per:employee_of                 Tesla, Inc.   8.245111
    5              SpaceX        org:founded_by                   Elon Musk   7.795369
    6           Elon Musk       per:employee_of                      SpaceX   7.765485
    7           Elon Musk       per:employee_of          The Boring Company   7.217330
    8         Tesla, Inc.        org:founded_by                   Elon Musk   7.002990


    References
    ----------
        1. [Tomaz Bratanic. From Text to Knowledge: The Information Extraction Pipeline.](https://towardsdatascience.com/from-text-to-knowledge-the-information-extraction-pipeline-b65e7e30273e)
        2. [Wikifier API documentation.](http://wikifier.org/info.html)
        3. [Wikifier API registration.](http://wikifier.org/register.html)

    """

    def __init__(self, key: str, types=[], device="cpu", luke=False):
        self.key = key

        try:
            self.nlp = spacy.load("en")
        except OSError:
            raise SpacyModelError()

        neuralcoref.add_to_pipe(self.nlp)

        if luke:
            self.model = LukeWrapper(device=device)
        else:
            self.model = opennre.get_model("wiki80_cnn_softmax")

        self.model = self.model.to(device)

        self.types = types

    def coreference(self, text: str):
        """Coreferences resolution using Neuralcoref and Spacy.

        Parameters
        ----------
            text: Text on which the coreference resolution is applied.

        Example
        -------

        >>> from textokb import pipeline

        >>> pipeline = pipeline.TextToKnowledge(key="jueidnxsctiurpwykpumtsntlschpx")

        >>> pipeline.coreference(text = "I like Netflix, it contains great series.")
        'I like Netflix, Netflix contains great series.'

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

                    if data[0] == "no_relation":
                        continue

                    triples.append(
                        {
                            "head": permutation[0]["title"],
                            "relation": data[0],
                            "tail": permutation[1]["title"],
                            "score": data[1],
                        }
                    )

        return pd.DataFrame(triples)

    def process_sentence(self, text: str):
        triples = []
        text = self.coreference(text=text)

        for sentence in nltk.sent_tokenize(text):
            entities = self.wikipedia_el(text=sentence)
            if len(entities) > 1:
                triples.append(self.relation_extraction(text=sentence, entities=entities))

        if triples:
            return (
                pd.concat(triples)
                .sort_values(by="score", ascending=False)
                .drop_duplicates(keep="first", subset=["head", "tail"])
                .reset_index(drop=True)
            )
        return pd.DataFrame(columns=["head", "relation", "tail", "score"])


class LukeWrapper(torch.nn.Module):
    """Wrapper for Opennre library API.

    Parameters
    ----------
        device:  cuda or cpu, device used to make the prediction.


    References
    ----------
        1. (LUKE (Language Understanding with Knowledge-based Embeddings) )[https://github.com/studio-ousia/luke]

    """

    def __init__(self, device: str):
        super(LukeWrapper, self).__init__()
        self.device = device
        self.model = LukeForEntityPairClassification.from_pretrained(
            "studio-ousia/luke-large-finetuned-tacred"
        ).eval()
        self.tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-large-finetuned-tacred")

    def infer(self, parameters: dict):
        entity_spans = [tuple(parameters["h"]["pos"]), tuple(parameters["t"]["pos"])]
        inputs = self.tokenizer(parameters["text"], entity_spans=entity_spans, return_tensors="pt")
        # Send inputs data to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # Ask for prediction
        with torch.no_grad():
            logits = self.model(**inputs).logits.flatten()
        y_pred = int(logits.argmax())
        return self.model.config.id2label[y_pred], logits[y_pred].item()
