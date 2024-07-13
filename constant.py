STATUS_LEN: int = 15
DIGIT2LABEL_NAME: dict[int, str] = {
    1: 'RightAtrium',
    2: 'RightVentricle',
    3: 'LeftAtrium',
    4: 'LeftVentricle',
    5: 'MyocardiumLV',
    6: 'Aorta',
    7: 'Coronaries8',
    8: 'Fat',
    9: 'Bypass',
    10: 'Plaque'
}
LABEL_NAME2DIGIT: dict[str, int] = {value: key for key, value in DIGIT2LABEL_NAME.items()}
EXPAND_SHORT_NAME: dict[str, str] = {
    'LM': "Left main coronary artery",
    "LAD": "Left anterior descending artery",
    "LCX": "Left circumflex artery",
    "RCA": "Right coronary artery"
}

