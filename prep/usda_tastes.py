"""Taste profiles from USDA FNDDS (FoodData Central survey foods, public domain).

    python prep/usda_tastes.py            # writes data/usda/fndds_index.tsv (one row per FNDDS food)

Each FNDDS food lists nutrients per 100 g and its ingredients (SR Legacy foods, with gram weights).
The four fly channels come from them with saturating curves, 1 - exp(-x/k):

  sweet   total sugars, g/100 g                         (k = 15: fruit ~0.5, cake ~0.9)
  salty   sodium, mg/100 g                              (k = 500: soup ~0.45, pizza ~0.7)
  bitter  caffeine and theobromine, mg/100 g, plus the gram share of bitter ingredients
          (coffee, tea and cocoa are already counted by caffeine/theobromine)
  watery  water from 55 to 90 g/100 g, linear: soups and drinks count, pasta a little, bread not at all

Nutrient ids are looked up by name in nutrient.csv, not hard-coded.
"""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FNDDS = next(p for p in (ROOT / 'data/usda').glob('FoodData_Central_survey_food_csv_*') if p.is_dir())
OUT = ROOT / 'data/usda/fndds_index.tsv'

NUTRIENTS = {'sugar': 'Total Sugars', 'sodium': 'Sodium, Na', 'water': 'Water',
             'caffeine': 'Caffeine', 'theobromine': 'Theobromine'}

# Bitterness of ingredients that caffeine/theobromine miss, 0..1, matched on SR descriptions.
BITTER_INGREDIENTS = [
    (r'arugula|radicchio|endive|escarole|chicory|dandelion|broccoli raab|rapini|balsam-pear|bitter gourd|bitter melon', 1.0),
    # Hopped beer only: not root beer or ginger ale (soft drinks), which matched a plain \bbeer\b|\bale\b.
    (r'tonic water|^alcoholic bever\w*, (beer|ale)|non-alcoholic beer', 0.8),
    (r'mustard greens|grapefruit|watercress', 0.7),
    (r'\bkale\b|collards|turnip greens|brussels sprouts', 0.6),
    # Olives themselves, not olive oil ("Oil, olive, ..."), which tastes close to neutral.
    (r'^olives\b|cranberries, raw|wine, (table|rose)|vermouth|campari|liqueur, coffee', 0.4),
    (r'\bbroccoli\b|cabbage, red|asparagus', 0.25),
]
BITTER_RE = [(re.compile(p, re.I), b) for p, b in BITTER_INGREDIENTS]

K_SUGAR, K_SODIUM, K_CAFFEINE, K_THEOBROMINE, K_BITTER_SHARE = 15.0, 500.0, 60.0, 600.0, 0.6


def sat(x: float, k: float) -> float:
    return 1 - math.exp(-max(0.0, x) / k)


def taste(n: dict) -> dict:
    bitter_x = n['caffeine'] / K_CAFFEINE + n['theobromine'] / K_THEOBROMINE + n['bitter_share'] / K_BITTER_SHARE
    return {
        'sweet': sat(n['sugar'], K_SUGAR),
        'bitter': sat(bitter_x, 1.0),
        'salty': sat(n['sodium'], K_SODIUM),
        'watery': max(0.0, min(1.0, (n['water'] - 55) / 35)),
    }


def ingredient_bitterness(desc: str) -> float:
    return max((b for r, b in BITTER_RE if r.search(desc)), default=0.0)


def load() -> dict[int, dict]:
    ids = {}
    for r in csv.DictReader(open(FNDDS / 'nutrient.csv')):
        for key, name in NUTRIENTS.items():
            if r['name'] == name:
                ids[r['nutrient_nbr']] = key  # survey foods' food_nutrient.csv uses the legacy number
    assert len(ids) == len(NUTRIENTS), ids
    foods = {int(r['fdc_id']): {'description': r['description'], **{k: 0.0 for k in NUTRIENTS},
                                 'bitter_share': 0.0, 'bitter_from': []}
             for r in csv.DictReader(open(FNDDS / 'food.csv'))}
    for r in csv.DictReader(open(FNDDS / 'food_nutrient.csv')):
        key = ids.get(r['nutrient_id'])
        if key:
            foods[int(r['fdc_id'])][key] = float(r['amount'] or 0)
    grams: dict[int, float] = {}
    for r in csv.DictReader(open(FNDDS / 'input_food.csv')):
        f, g = int(r['fdc_id']), float(r['gram_weight'] or 0)
        grams[f] = grams.get(f, 0) + g
        b = ingredient_bitterness(r['sr_description'])
        if b:
            foods[f]['bitter_share'] += g * b
            foods[f]['bitter_from'].append(r['sr_description'].split(',')[0])
    for f, food in foods.items():
        if grams.get(f):
            food['bitter_share'] /= grams[f]
        food['taste'] = taste(food)
    return foods


def main():
    foods = load()
    cols = ['sugar', 'sodium', 'water', 'caffeine', 'theobromine', 'bitter_share']
    with open(OUT, 'w') as fh:
        fh.write('\t'.join(['fdc_id', 'description', *cols, 'sweet', 'bitter', 'salty', 'watery', 'bitter_from']) + '\n')
        for f, d in sorted(foods.items(), key=lambda kv: kv[1]['description']):
            t = d['taste']
            fh.write('\t'.join([str(f), d['description'], *(f'{d[c]:.3g}' for c in cols),
                                *(f'{t[c]:.2f}' for c in ('sweet', 'bitter', 'salty', 'watery')),
                                ';'.join(sorted(set(d['bitter_from'])))]) + '\n')
    print(f'{len(foods)} foods -> {OUT.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
