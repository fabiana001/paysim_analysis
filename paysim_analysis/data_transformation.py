
from typing import List
from time import time
import pandas as pd
from sklearn.preprocessing import StandardScaler

class DatasetTransformer:
    def __init__(self):
        self.target = 'isFraud'
        self.cols_to_normalize = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest',
                             'avgAmtOrigStep', 'countOrigStep', 'avgAmtOrig',
                             'countOrig', 'avgAmtDestStep', 'countDestStep', 'avgAmtDest', 'countDest']
        self.all_cols = self.cols_to_normalize + ['is_CASH_OUT', self.target]

    def _data_enrichment(self, df_to_enrich: pd.DataFrame, node_name: str, cols_name: List[str],
                         cols_name_step: List[str]):
        """
        The function calculates the number and avg amount of transactions of each node_name and (node_name, step)
        It returns a new dataframe having as new columns cols_name + cols_name_step
        :param df_to_enrich: input dataframe
        :param node_name: key feature for the aggregation
        :param cols_name: list of two column name, first element represents count, the second represents the avg amount
        :param cols_name_step: list of two column name, first element represents count, the second represents the avg amount
        :return:
        """
        df = df_to_enrich.groupby(node_name).agg({"amount": "mean", "step": "count"})
        df_step = df_to_enrich.groupby([node_name, "step"]).agg({"amount": "mean", "step": "count"})
        df.columns = cols_name
        df_step.columns = cols_name_step

        df_1 = df_to_enrich.merge(df, on=node_name, how="inner").merge(df_step, on=[node_name, "step"], how="inner")
        assert len(df_1) == len(df_to_enrich)
        return df_1

    def _data_normalization(self, df:pd.DataFrame) -> pd.DataFrame:
        """
        Standard Scaling
        :param df:
        :return:
        """
        scaler = StandardScaler()
        scaler.fit(df[self.cols_to_normalize])
        df[self.cols_to_normalize] = scaler.transform(df[self.cols_to_normalize])
        return df

    def _filtering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Data record removing
        :param df:
        :return:
        """
        df = df[df["type"].isin(["TRANSFER", "CASH_OUT"])]
        df = df[~df["nameDest"].str.startswith("M")]
        return df

    def run(self, paysim_df:pd.DataFrame) -> pd.DataFrame:
        """
        The function executes the following steps:
          1. Data records filtering
          1. Data Enrichment
          3. Data transformation using z-score
        """
        initial_df = len(paysim_df)
        df = paysim_df.copy()
        t0 = time()
        #remove useless data records
        df = self._filtering(df)

        #add src and dst info
        df_1 = self._data_enrichment(df, "nameOrig", ["avgAmtOrig", "countOrig"], ["avgAmtOrigStep", "countOrigStep"])
        df_enriched = self._data_enrichment(df_1, "nameDest", ["avgAmtDest", "countDest"], ["avgAmtDestStep", "countDestStep"])
        #t1 = time() - t0
        #print(f'DATA AGGREGATION done in {int(t1)} sec')

        df_enriched["is_CASH_OUT"] = df_enriched["type"] == "CASH_OUT"
        df_enriched[self.target] = df_enriched[self.target].astype('bool')

        #z-score
        df_enriched = self._data_normalization(df_enriched)

        #remove useless features
        df_enriched = df_enriched[self.all_cols]

        final_df = len(df_enriched)
        t1 = time() - t0
        print(f"FILTERING STEP, removed {initial_df - final_df} data records")
        print(f'DATA ENRICHMENT done in {str(int(t1))} sec')
        return df_enriched

