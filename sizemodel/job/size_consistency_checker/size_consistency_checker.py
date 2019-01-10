import json

import pandas as pd
from pkg_resources import resource_filename


class SizeChecker():
    def __init__(self):
        model_weight_file = resource_filename(__name__, 'size_checker_weights.json')
        with open(model_weight_file, 'r') as f:
            data = f.read()
        model_weights = json.loads(data)
        model_dict = {d['modeltargetvar']: {'model_input_vars': d['modelinputvars'],
                                            'coefficients': d['coefficients'],
                                            'threshold': d['threshold']}
                      for d in model_weights}
        self.model_dict = model_dict

    def find_wrong_input(self, df):
        """
        Assuming the needed variables for prediction are contained in the given
        database, return a dataframe that indicates plausible sizes by True and others False.
        """
        consistency_df = pd.DataFrame(index=df.index)
        for target_var, model in self.model_dict.items():
            if target_var not in df.columns:
                continue
            target = df[target_var]
            df[model['model_input_vars']] = df[model['model_input_vars']].astype('float')
            try:
                n = len(model['model_input_vars'])
                pred = pd.Series(index=df.index, data=model['coefficients'][0])
                for i in range(n):
                    pred += df[model['model_input_vars'][i]] * model['coefficients'][i+1]
                consistency_df[target_var] = (pred - target).abs() <= model['threshold']
            except KeyError as e:
                print('caught exception because a column was missing', e)
                continue
        return consistency_df
