# Extract knowledge from raw text

This repository is a copy-paste of ["From Text to Knowledge: The Information Extraction Pipeline"](https://towardsdatascience.com/from-text-to-knowledge-the-information-extraction-pipeline-b65e7e30273e) with some cosmetic updates. I made an installable version to evaluate it easily. The original code is available @ [trinity-ie](https://github.com/tomasonjo/trinity-ie). To create some value, I added the [Luke](https://github.com/studio-ousia/luke) model to predict relations between entities.

In this blog post, Tomaz Bratanic presents a complete pipeline for extracting triples from raw text. The first step of the pipeline is to resolve the coreferences. The second step of the pipeline is to identify entities using the [Wikifier API](http://wikifier.org/info.html). Finally, Tomaz Bratanic proposes to use the [Opennre](https://github.com/thunlp/OpenNRE) library to extract relations between entities within the text.

## 🔧 Installation

```sh
pip install git+https://github.com/raphaelsty/textokb --upgrade
```

You will have to download spacy `en` model to do coreference resolution:

```sh
pip install spacy==2.1.0 && python -m spacy download en
```

## Quick start

```python
>>> from textokb import pipeline

# A list of types of entities that I search:
>>> types = [
...   "human", 
...   "person", 
...   "company", 
...   "enterprise", 
...   "business", 
...   "geographic region", 
...   "human settlement", 
...   "geographic entity", 
...   "territorial entity type", 
...   "organization",
... ]

>>> device = "cpu" # or device = "cuda" if you do own a gpu.

>>> pipeline = pipeline.TextToKnowledge(key="jueidnxsctiurpwykpumtsntlschpx", types=types, device=device)

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
```

By default the model used is `wiki80_cnn_softmax`. I also added the model [Luke (Language Understanding with Knowledge-based Embeddings)](https://github.com/studio-ousia/luke) which provide a pre-trained models to do relation extraction. The results of the Luke model seem to be of better quality but the number of predicted relationships is smaller.

## Here is how to use LUKE

```python
>>> from textokb import pipeline

# A list of types of entities that I search:
>>> types = [
...   "human", 
...   "person", 
...   "company", 
...   "enterprise", 
...   "business", 
...   "geographic region", 
...   "human settlement", 
...   "geographic entity", 
...   "territorial entity type", 
...   "organization",
... ]

>>> device = "cpu" # or device = "cuda" if you do own a gpu.

>>> pipeline = pipeline.TextToKnowledge(key="jueidnxsctiurpwykpumtsntlschpx", types=types, device=device, luke=True)

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
```

Here is the list of available relations using Luke `studio-ousia/luke-large-finetuned-tacred`:

```python
[
    'no_relation',
    'org:alternate_names',
    'org:city_of_headquarters',
    'org:country_of_headquarters',
    'org:dissolved',
    'org:founded',
    'org:founded_by',
    'org:member_of',
    'org:members',
    'org:number_of_employees/members',
    'org:parents',
    'org:political/religious_affiliation',
    'org:shareholders',
    'org:stateorprovince_of_headquarters',
    'org:subsidiaries',
    'org:top_members/employees',
    'org:website',
    'per:age',
    'per:alternate_names',
    'per:cause_of_death',
    'per:charges',
    'per:children',
    'per:cities_of_residence',
    'per:city_of_birth',
    'per:city_of_death',
    'per:countries_of_residence',
    'per:country_of_birth',
    'per:country_of_death',
    'per:date_of_birth',
    'per:date_of_death',
    'per:employee_of',
    'per:origin',
    'per:other_family',
    'per:parents',
    'per:religion',
    'per:schools_attended',
    'per:siblings',
    'per:spouse',
    'per:stateorprovince_of_birth',
    'per:stateorprovince_of_death',
    'per:stateorprovinces_of_residence',
    'per:title'
]
```

## ♻️ Work in progress

I failed to use the `wiki80_bert_softmax` model from [Opennre](https://github.com/thunlp/OpenNRE) due to a pre-trained model loading error (i.e. Tensorflow errors on Mac M1). I used the lighter model `wiki80_cnn_softmax` when reproducing Tomaz Bratanic's blog post. It would be interesting to be able to easily add different models and especially transformers. The API I used are not optimized for batch predictions. There are a lot of room for improvement by simply updating Opennre and Luke APIs.
