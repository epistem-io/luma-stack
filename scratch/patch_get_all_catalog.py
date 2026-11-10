with open('src/luma_ge/predictor.py', 'r', encoding='utf-8') as f:
    text = f.read()

target = '    @classmethod\n    def check_availability('
addition = '''    @classmethod
    def get_all_catalog(cls) -> dict:
        """Return a copy of the entire predictor catalog dictionary."""
        return cls.PREDICTORS.copy()

'''

if 'def get_all_catalog' not in text:
    new_text = text.replace(target, addition + target, 1)
    with open('src/luma_ge/predictor.py', 'w', encoding='utf-8') as f:
        f.write(new_text)
    print('Added get_all_catalog to PredictorCatalog')
else:
    print('Already present')
