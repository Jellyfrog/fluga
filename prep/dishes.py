"""The dish vocabulary: prep/dishes/*.json, each dish a list of USDA FNDDS foods with gram weights.

    python prep/dishes.py check [prep/dishes/x.json ...]   # validate, print each dish's taste
    python prep/dishes.py tsv > data/usda/dishes.tsv         # every dish with nutrients and taste

Dish entry:
  {"id": "takoyaki", "name": "Takoyaki", "desc": "battered octopus balls with sauce and bonito",
   "aliases": ["octopus balls"], "food101": null,
   "fndds": [[<fdc_id>, <grams>], ...]}

One FNDDS food when USDA has the dish itself; a short recipe of FNDDS foods when it doesn't. A dish's
nutrients are the gram-weighted mean of its components', and its taste follows from those with the
curves in usda_tastes.py (not from averaging the components' tastes; the curves are not linear).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from usda_tastes import load as load_fndds, taste  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DISH_DIR = ROOT / 'prep/dishes'
FOOD101 = json.loads((ROOT / 'prep/classifier/food101_labels.json').read_text())
KEYS = ('sugar', 'sodium', 'water', 'caffeine', 'theobromine', 'bitter_share')
SLUG = re.compile(r'^[a-z0-9]+(_[a-z0-9]+)*$')


def dish_nutrients(dish, foods):
    total = sum(g for _, g in dish['fndds'])
    return {k: sum(foods[f][k] * g for f, g in dish['fndds']) / total for k in KEYS}


def validate(dishes, foods, where=''):
    errs, seen = [], {}
    for i, d in enumerate(dishes):
        at = f"{where}[{i}] {d.get('id', '?')}"
        if not isinstance(d.get('id'), str) or not SLUG.match(d['id']):
            errs.append(f'{at}: id must be snake_case')
        if not isinstance(d.get('name'), str) or not d['name'].strip():
            errs.append(f'{at}: name missing')
        if not isinstance(d.get('desc', ''), str) or len(d.get('desc', '').split()) > 14:
            errs.append(f'{at}: desc must be a short string (<= 14 words)')
        if not isinstance(d.get('aliases', []), list) or not all(isinstance(a, str) for a in d.get('aliases', [])):
            errs.append(f'{at}: aliases must be a list of strings')
        if d.get('food101') is not None and d['food101'] not in FOOD101:
            errs.append(f"{at}: food101 label {d['food101']!r} is not a Food-101 class")
        comps = d.get('fndds')
        if not isinstance(comps, list) or not comps:
            errs.append(f'{at}: fndds must be a non-empty list of [fdc_id, grams]')
            continue
        for c in comps:
            if not (isinstance(c, list) and len(c) == 2 and isinstance(c[0], int) and isinstance(c[1], (int, float)) and c[1] > 0):
                errs.append(f'{at}: bad component {c!r}')
            elif c[0] not in foods:
                errs.append(f'{at}: fdc_id {c[0]} is not an FNDDS food')
        key = d.get('id')
        if key in seen:
            errs.append(f'{at}: duplicate id (also {seen[key]})')
        seen[key] = at
    return errs


def load_all(paths=None):
    paths = paths or sorted(DISH_DIR.glob('*.json'))
    out = []
    for p in paths:
        for d in json.loads(Path(p).read_text()):
            d['_file'] = Path(p).name
            out.append(d)
    return out


def with_taste(dishes, foods):
    for d in dishes:
        d['nutrients'] = dish_nutrients(d, foods)
        d['taste'] = taste(d['nutrients'])
    return dishes


def main():
    cmd, paths = (sys.argv[1] if len(sys.argv) > 1 else 'check'), sys.argv[2:]
    foods = load_fndds()
    dishes = load_all(paths)
    errs = validate(dishes, foods, where=','.join(Path(p).name for p in paths) or 'all')
    if errs:
        print('\n'.join(errs))
        print(f'{len(errs)} problems in {len(dishes)} dishes')
        sys.exit(1)
    with_taste(dishes, foods)
    if cmd == 'check':
        for d in dishes:
            t = d['taste']
            parts = ' + '.join(f"{g:g} g {foods[f]['description'][:40]}" for f, g in d['fndds'])
            print(f"{d['name'][:30]:30s} sweet {t['sweet']:.2f} bitter {t['bitter']:.2f} salty {t['salty']:.2f} "
                  f"watery {t['watery']:.2f}  <- {parts}")
        print(f'OK: {len(dishes)} dishes')
    elif cmd == 'tsv':
        print('id\tname\tfile\t' + '\t'.join(KEYS) + '\tsweet\tbitter\tsalty\twatery')
        for d in dishes:
            print('\t'.join([d['id'], d['name'], d['_file'], *(f"{d['nutrients'][k]:.3g}" for k in KEYS),
                             *(f"{d['taste'][k]:.2f}" for k in ('sweet', 'bitter', 'salty', 'watery'))]))


if __name__ == '__main__':
    main()
