from copy import deepcopy
from typing import List

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from Definitions import *
from BayesRegression import BernoulliRegression as BReg



class ModelSpec:
    spec = {}
    
    def __init__(self): 
        pass
        
    def __str__(self): 
        return str(self.ToDict())

    def __repr__(self): 
        return self.__str__()
    
    def ToDict(self) -> dict: 
        return self.spec
    
    @staticmethod
    def __ReadInput(inp):
        if (inp is None) or (inp == []): 
            return None
        elif isinstance(inp, str):
            return [inp]
        elif isinstance(inp, list):
            return list(inp)
        else: 
            raise ValueError("Unknown input {}".format(inp))
    
    def FromLists(self, targets, media, non_media, relevance_groups, report_splits):
        self.spec[TARGETS] = ModelSpec.__ReadInput(targets)
        self.spec[MEDIA] = ModelSpec.__ReadInput(media)
        self.spec[NON_MEDIA] = ModelSpec.__ReadInput(non_media)
        self.spec[RELEVANCE_GROUPS] = ModelSpec.__ReadInput(relevance_groups)
        self.spec[REPORT_SPLITS] = ModelSpec.__ReadInput(report_splits)
        return self
    
    def FromDict(self, spec: dict) -> None:
        for section in SPEC_SECTIONS:
            if section in spec.keys():
                self.spec[section] = ModelSpec.__ReadInput(spec[section])
            else:
                print("Warning! No {} section in spec".format(section)) 

    def ValidateSpecVsData(self, data: pd.DataFrame) -> bool:
        for _, section in self.spec.items():
            if section is not None:
                for v in section:
                    assert v in data, "Variable {} not found in data".format(v)
        return True
    
    def Targets(self) -> list[str]:
        # список переменных, даже если одна
        return self.spec[TARGETS]
    
    def Media(self) -> list[str]:
        # список переменных, даже если одна
        return self.spec[MEDIA]
    
    def NonMedia(self) -> list[str]:
        # список переменных, даже если одна
        return self.spec[NON_MEDIA]
    
    def RelevanceGroup(self) -> str | None:
        # одна переменная, строка
        if self.spec[RELEVANCE_GROUPS] is None:
            return None
        else:
            return self.spec[RELEVANCE_GROUPS][0]
        
    def ReportSplits(self) -> list[str]:
        # список переменных, даже если одна
        return self.spec[REPORT_SPLITS]
        

def PrepareInput(data: pd.DataFrame, inp):
    if (inp is None) or (inp == []):
        return None
    elif isinstance(inp, str) or isinstance(inp, list):
        return data[inp].values
    else:
        raise ValueError("Some shit in input {}".format(inp))
    

class MediaDecomposition:
    spec: ModelSpec = None
    data: pd.DataFrame = None
    models: list
    X_media: np.array = None
    X_non_media: np.array = None
    X_split: np.array = None

    report_splits: dict = None

    def __init__(self) -> None:
        self.models = list()
        self.report_splits = dict()

    def PrepareModelInputs(self, spec: ModelSpec, data: pd.DataFrame):
        self.X_media = PrepareInput(data, spec.Media())
        self.X_non_media = PrepareInput(data, spec.NonMedia())
        
        # LabelEncoder нужен потому что данные могут быть 1,2,... а нужно 0,1,...
        self.X_split = PrepareInput(data, spec.RelevanceGroup())
        if self.X_split is not None: 
            self.X_split = LabelEncoder().fit_transform(self.X_split)
        
        if spec.ReportSplits() is not None:
            ohe = OneHotEncoder(sparse_output=False).fit(data[spec.ReportSplits()])
            self.report_splits['Total'] = pd.Series([True] * len(data))
            self.report_splits.update(
                dict(pd.DataFrame(
                        ohe.transform(data[spec.ReportSplits()]).astype(bool), 
                        columns=ohe.get_feature_names_out()
                    ).items()
                )
            )


    def Fit(self, spec: ModelSpec, data: pd.DataFrame, show_traces=False):
        spec.ValidateSpecVsData(data)

        self.models.clear()
        self.report_splits.clear()
        self.spec = deepcopy(spec)
        self.data = data.copy()

        self.PrepareModelInputs(spec, data)
    
        for t in spec.Targets():
            y = PrepareInput(data, t)
            self.models.append(
                BReg(t).Fit(self.X_media, self.X_non_media, self.X_split, y=y, show_trace=show_traces)
            )
    
    def Contributions(self):
        #

        # dims: (resps, model_elements, targets)
        contribs_all_targets = np.stack(
            [m.Contributions(self.X_media, self.X_non_media, self.X_split) for m in self.models], 
            axis=-1
        )
        data_all_targets = self.data[self.spec.Targets()]

        result = {}
        report_index = ['Base', 'Non-media'] + self.spec.Media()
        report_columns = self.spec.Targets()
        
        for name, filt in self.report_splits.items():
            rep = pd.DataFrame(contribs_all_targets[filt].mean(axis=0), index=report_index, columns=report_columns)
            rep[rep < 0] = 0 
            obs = data_all_targets[filt].mean()
            rep = rep / (rep.sum() / obs)
            rep.loc["Observed", :] = obs
            result[name] = rep
        
        return pd.concat(result)
         