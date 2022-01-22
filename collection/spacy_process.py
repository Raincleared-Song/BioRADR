import torch
import spacy
from .ncbi_api import get_pmids
from scispacy.linking import EntityLinker
from scispacy.abbreviation import AbbreviationDetector


def perform_ner(sentence: str):
    assert torch.cuda.is_available()
    spacy.prefer_gpu(0)
    nlp = spacy.load('en_core_sci_lg')
    paper = get_pmids(['28483577'])
    paper = ' '.join(paper['28483577']['texts'])
    nlp.add_pipe('abbreviation_detector')
    nlp.add_pipe('scispacy_linker', config={'resolve_abbreviations': True, 'linker_name': 'mesh'})
    paper = nlp(paper)
    for abrv in paper._.abbreviations:
        print(f"{abrv} \t ({abrv.start}, {abrv.end}) {abrv._.long_form}")
    for entity in paper.ents:
        print(entity, entity._.kb_ents)


if __name__ == '__main__':
    perform_ner('')
