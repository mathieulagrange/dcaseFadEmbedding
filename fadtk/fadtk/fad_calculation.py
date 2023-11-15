from argparse import ArgumentParser

from .fad import FrechetAudioDistance, log
from .model_loader import *
from .fad_batch import cache_embedding_files

def calculate_fad(model_type, baseline, eval, workers=8, s='/usr/bin/sox', inf=True, indiv='/path/to/indiv'):

    models = {m.name: m for m in get_all_models()}
    print(models.keys())
    model = models[model_type]
    for d in [baseline, eval]:
        if Path(d).is_dir():
            cache_embedding_files(d, model, workers=workers)

    fad = FrechetAudioDistance(model, audio_load_worker=workers, load_model=False)

    if inf:
        assert Path(eval).is_dir(), "FAD-inf requires a directory as the evaluation dataset"
        score = fad.score_inf(baseline, list(Path(eval).glob('*.*')))
    elif indiv:
        assert Path(eval).is_dir(), "Individual FAD requires a directory as the evaluation dataset"
        fad.score_individual(baseline, eval, Path(indiv))
        log.info(f"Individual FAD scores saved to {indiv}")
        exit(0)
    else:
        score = fad.score(baseline, eval)

    return(score)