"""Database-specific search strategies for the systematic-review update.

PREREGISTERED holds the three concept blocks exactly as registered
(OSF e5w47, §10; osf_registration_draft.md). ADDED holds the terms the
amendment adds (protocol_amendment.md, A2); they widen the search and
never remove a registered term. Any later change is a protocol deviation:
record it in DEVIATIONS with the date and the reason.
"""

PREREGISTERED = {
    "personality": [
        '"Big Five"', '"Five-Factor Model"', '"FFM"', '"HEXACO"', '"BFI"',
        '"NEO-PI-R"', '"NEO-FFI"', '"IPIP"', '"conscientiousness"',
        '"openness"', '"extraversion"', '"agreeableness"', '"neuroticism"',
        '"emotional stability"', '"personality traits"',
    ],
    "online": [
        '"online learning"', '"e-learning"', '"distance learning"',
        '"remote learning"', '"virtual learning"', '"blended learning"',
        '"hybrid learning"', '"MOOC"', '"massive open online course"',
        '"web-based learning"', '"computer-mediated learning"',
        '"learning management system"', '"LMS"', '"online course"',
        '"synchronous online"', '"asynchronous online"',
    ],
    "outcome": [
        '"academic performance"', '"academic achievement"', '"GPA"',
        '"grade point average"', '"test score"', '"course grade"',
        '"learning outcome"', '"learning performance"', '"academic success"',
    ],
}

# Spelling variants, plurals and synonyms the registered blocks miss.
ADDED = {
    "personality": ['"five factor"', '"extroversion"', '"personality"'],
    "online": [
        '"elearning"', '"online education"', '"distance education"',
        '"online instruction"', '"MOOCs"', '"massive open online courses"',
        '"online courses"', '"emergency remote teaching"',
        '"emergency remote learning"', '"online class"', '"online classes"',
    ],
    "outcome": [
        '"test scores"', '"course grades"', '"learning outcomes"', '"grades"',
        '"exam performance"', '"examination performance"', '"final grade"',
        '"course performance"', '"course completion"', '"academic outcomes"',
    ],
}

BLOCKS = ("personality", "online", "outcome")


def terms(block, amended=True):
    return PREREGISTERED[block] + (ADDED[block] if amended else [])


def _block(ts, field=""):
    return "(" + " OR ".join(f"{t}{field}" for t in ts) + ")"


def pubmed_query(amended=True):
    """PubMed: title/abstract field tag on every term; English only."""
    q = " AND ".join(_block(terms(b, amended), "[tiab]") for b in BLOCKS)
    return f"{q} AND english[la]"


def eric_query(amended=True):
    """ERIC API (Solr/Lucene syntax over the default search fields)."""
    return " AND ".join(_block(terms(b, amended)) for b in BLOCKS)


def openalex_query(amended=True):
    """OpenAlex title_and_abstract.search (Boolean AND/OR, quoted phrases).
    Language is filtered separately (language:en) in run_search.py."""
    return " AND ".join(_block(terms(b, amended)) for b in BLOCKS)


def online_block():
    """The online-learning block alone, used to filter forward citations of
    the benchmark meta-analyses (run_search.search_citations)."""
    return _block(terms("online"))


DATABASES = {
    "pubmed": {"name": "PubMed/MEDLINE", "query": pubmed_query},
    "eric": {"name": "ERIC", "query": eric_query},
    "openalex": {"name": "OpenAlex", "query": openalex_query},
}

# (date, change, reason) — empty until the amendment is registered
DEVIATIONS = []
