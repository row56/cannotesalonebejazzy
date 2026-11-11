import re
import unicodedata
import numpy as np

UNKNOWN = "unknown"
COMPOSER_ERA = {
    "bach": "baroque",
    "handel": "baroque",
    "scarlatti": "baroque",
    "rameau": "baroque",
    "couperin": "baroque",
    "vivaldi": "baroque",
    "purcell": "baroque",
    "telemann": "baroque",
    "buxtehude": "baroque",
    "frescobaldi": "baroque",
    "sweelinck": "baroque",
    "byrd": "baroque",
    "mozart": "classical",
    "haydn": "classical",
    "clementi": "classical",
    "salieri": "classical",
    "cimarosa": "classical",
    "dussek": "classical",
    "hummel": "classical",
    "cherubini": "classical",
    "field": "classical",
    "beethoven": "romantic",
    "schubert": "romantic",
    "chopin": "romantic",
    "schumann": "romantic",
    "liszt": "romantic",
    "brahms": "romantic",
    "mendelssohn": "romantic",
    "tchaikovsky": "romantic",
    "grieg": "romantic",
    "dvorak": "romantic",
    "franck": "romantic",
    "faure": "romantic",
    "saint-saens": "romantic",
    "bruckner": "romantic",
    "wagner": "romantic",
    "alkan": "romantic",
    "moszkowski": "romantic",
    "godowsky": "romantic",
    "balakirev": "romantic",
    "borodin": "romantic",
    "mussorgsky": "romantic",
    "rimsky-korsakov": "romantic",
    "smetana": "romantic",
    "albeniz": "romantic",
    "granados": "romantic",
    "sinding": "romantic",
    "gottschalk": "romantic",
    "debussy": "impressionist",
    "ravel": "impressionist",
    "satie": "impressionist",
    "scriabin": "impressionist",
    "rachmaninoff": "impressionist",
    "rachmaninov": "impressionist",
    "medtner": "impressionist",
    "szymanowski": "impressionist",
    "delius": "impressionist",
    "de falla": "impressionist",
    "falla": "impressionist",
    "respighi": "impressionist",
    "prokofiev": "modern",
    "shostakovich": "modern",
    "stravinsky": "modern",
    "bartok": "modern",
    "hindemith": "modern",
    "schoenberg": "modern",
    "berg": "modern",
    "webern": "modern",
    "messiaen": "modern",
    "poulenc": "modern",
    "milhaud": "modern",
    "ives": "modern",
    "copland": "modern",
    "cage": "modern",
    "ligeti": "modern",
    "kabalevsky": "modern",
    "khachaturian": "modern",
    "villa-lobos": "modern",
    "gershwin": "modern",
    "barber": "modern",
    "britten": "modern",
    "janacek": "modern",
    "kapustin": "modern",
    "glass": "modern",
}
ERA_ORDER = ["baroque", "classical", "romantic", "impressionist", "modern"]


def normalise(name: str):
    name = unicodedata.normalize("NFKD", name)
    name = "".join((c for c in name if not unicodedata.combining(c)))
    name = re.sub("[^\\w\\s-]", " ", name.lower())
    return re.sub("\\s+", " ", name).strip()


def era_of(artist: str):
    key = normalise(artist)
    if key in COMPOSER_ERA:
        return COMPOSER_ERA[key]
    for token in key.replace("-", " ").split():
        if token in COMPOSER_ERA:
            return COMPOSER_ERA[token]
    for composer in COMPOSER_ERA:
        if " " in composer or "-" in composer:
            if composer in key:
                return COMPOSER_ERA[composer]
    return UNKNOWN


class LabelSet:

    def __init__(self, fields: dict[str, list[str]]):
        self.codes: dict[str, np.ndarray] = {}
        self.names: dict[str, list[str]] = {}
        for field, values in fields.items():
            names = sorted(set(values))
            index = {n: i for i, n in enumerate(names)}
            self.names[field] = names
            self.codes[field] = np.array([index[v] for v in values], dtype=np.int32)

    @property
    def fields(self):
        return list(self.codes)

    def __len__(self):
        return len(next(iter(self.codes.values()))) if self.codes else 0

    def counts(self, field: str):
        c = np.bincount(self.codes[field], minlength=len(self.names[field]))
        return dict(zip(self.names[field], c.tolist()))

    def viable(self, field: str, minimum: int):
        return [n for n, c in self.counts(field).items() if c >= minimum and n != UNKNOWN]

    def to_npz_dict(self):
        out = {}
        for field in self.codes:
            out[f"label_{field}"] = self.codes[field]
            out[f"label_{field}_names"] = np.array(self.names[field], dtype=object)
        return out

    @classmethod
    def from_npz(cls, blob):
        obj = cls.__new__(cls)
        obj.codes, obj.names = ({}, {})
        for key in blob.files:
            if key.startswith("label_") and (not key.endswith("_names")):
                field = key[len("label_") :]
                obj.codes[field] = blob[key]
                obj.names[field] = list(blob[f"label_{field}_names"])
        return obj

    def values(self, field: str):
        return self.codes[field]
