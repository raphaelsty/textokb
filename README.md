# Extract knowledge from raw text

TThis repository is a copy and paste of ["From Text to Knowledge: The Information Extraction Pipeline"](https://towardsdatascience.com/from-text-to-knowledge-the-information-extraction-pipeline-b65e7e30273e) with some cosmetic updates. I made an installable version to evaluate it easily. The original code is available @ [trinity-ie](https://github.com/tomasonjo/trinity-ie).

In this blog post, Tomaz Bratanic presents a complete pipeline for extracting triples from raw text. The first step of the pipeline is to resolve the coreferences. The second step of the pipeline is to identify entities using the [Wikifier] API (http://wikifier.org/info.html). Finally, Tomaz Bratanic proposes to use the [Opennre](https://github.com/thunlp/OpenNRE) library to extract relations between entities within the text.

## 🔧 Installation

```sh
pip install git+https://username:token@github.com/raphaelsty/textokb --upgrade
```

```python
>>> from textokb import pipeline

>>> types = [
...     "human", 
...     "person", 
...     "company", 
...     "enterprise", 
...     "business", 
...     "geographic region", 
...     "human settlement", 
...     "geographic entity", 
...     "territorial entity type", 
...     "organization",
... ]

>>> pipeline = pipeline.TextToKnowledge(key="jueidnxsctiurpwykpumtsntlschpx", types=types)

>>> text = """ Elon Musk is a business magnate, industrial designer, and engineer. He is the founder, 
... CEO, CTO, and chief designer of SpaceX. He is also early investor, CEO, and product architect of 
... Tesla, Inc. He is also the founder of The Boring Company and the co-founder of Neuralink. A 
... centibillionaire, Musk became the richest person in the world in January 2021, with an estimated 
... net worth of $185 billion at the time, surpassing Jeff Bezos. Musk was born to a Canadian mother 
... and South African father and raised in Pretoria, South Africa. He briefly attended the University 
... of Pretoria before moving to Canada aged 17 to attend Queen's University. He transferred to the 
... University of Pennsylvania two years later, where he received dual bachelor's degrees in economics 
... and physics. He moved to California in 1995 to attend Stanford University, but decided instead to 
... pursue a business career. He went on co-founding a web software company Zip2 with his brother 
... Kimbal Musk."""

>>> pipeline.process_sentence(text = text)
                                head      relation         tail     score
0                            SpaceX      has part    Elon Musk  0.939970
1                          Pretoria      has part    Elon Musk  0.884259
2               Stanford University      has part    Elon Musk  0.884259
3        University of Pennsylvania      has part    Elon Musk  0.884259
4                The Boring Company      has part    Elon Musk  0.884259
..                              ...           ...          ...       ...
105                      Jeff Bezos    subsidiary       SpaceX  0.435877
106      University of Pennsylvania    subsidiary       SpaceX  0.435877
107  Queen's University at Kingston    subsidiary       SpaceX  0.435877
108                          SpaceX  manufacturer  Tesla, Inc.  0.345905
109                     Tesla, Inc.      owned by       SpaceX  0.310732
<BLANKLINE>
[110 rows x 4 columns]
```

#### Work in progress:

I failed to use the `wiki80_bert_softmax` model from [Opennre](https://github.com/thunlp/OpenNRE) due to a pre-trained model loading error (i.e. tensorflow error). I used the lighter model `wiki80_cnn_softmax` in the context of reproducing Tomaz Bratanic's blog post. It would be interesting to be able to easily add different models and especially transformers.