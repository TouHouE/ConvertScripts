STATUS_LEN: int = 15
DIGIT2LABEL_NAME = {
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
LABEL_NAME2DIGIT = {value: key for key, value in DIGIT2LABEL_NAME.items()}