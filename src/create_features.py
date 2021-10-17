import pandas as pd
from utils import decorate, Timer

from sklearn.preprocessing import StandardScaler, RobustScaler
from tqdm import tqdm

class AbstractBaseBlock:
    def fit(self, input_df: pd.DataFrame, y=None):
        return self.transform(input_df)

    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()


class AddMultiplyingDividing(AbstractBaseBlock):
    def transform(self, input_df):
        input_df['area'] = input_df['time_step'] * input_df['u_in']
        input_df['area'] = input_df.groupby('breath_id')['area'].cumsum()
        input_df['cross'] = input_df['u_in']*input_df['u_out']
        input_df['cross2'] = input_df['time_step']*input_df['u_out']
        input_df['u_in_cumsum'] = (input_df['u_in']).groupby(input_df['breath_id']).cumsum()
        input_df['one'] = 1
        input_df['count'] = (input_df['one']).groupby(input_df['breath_id']).cumsum()
        input_df['u_in_cummean'] = input_df['u_in_cumsum'] / input_df['count']
        input_df = input_df.merge(
            input_df[input_df["u_out"]==0].groupby('breath_id')['u_in'].agg(["mean", "std", "max"]).add_prefix("u_out0_").reset_index(),
            on="breath_id"
        )
        input_df = input_df.merge(
            input_df[input_df["u_out"]==1].groupby('breath_id')['u_in'].agg(["mean", "std", "max"]).add_prefix("u_out1_").reset_index(),
            on="breath_id"
        )

        output_df = pd.DataFrame(
            {
                "area": input_df['area'],
                #"cross": input_df['cross'],
                #"cross2": input_df['cross2'],
                "u_in_cumsum": input_df['u_in_cumsum'],
                "u_in_cummean": input_df['u_in_cummean'],
                "u_out0_mean": input_df['u_out0_mean'],
                "u_out0_max": input_df['u_out0_max'],
                "u_out0_max": input_df['u_out0_std'],
                "u_out1_mean": input_df['u_out1_mean'],
                "u_out1_max": input_df['u_out1_max'],
                "u_out1_max": input_df['u_out1_std'],
            }
        )
        cfg.cont_seq_cols += output_df.add_suffix(f'@{self.__class__.__name__}').columns.tolist()
        return output_df


class RCDummry(AbstractBaseBlock):
    def transform(self, input_df):
        input_df['R_dummy'] = input_df['R'].astype(str)
        input_df['C_dummy'] = input_df['C'].astype(str)
        #input_df['RC_dummy'] = input_df['R_dummy'] + input_df['C_dummy']
        output_df = pd.get_dummies(input_df[["R_dummy", "C_dummy"]])
        cfg.cont_seq_cols += output_df.add_suffix(f'@{self.__class__.__name__}').columns.tolist()
        return output_df


class AddBreathTimeAndUInTime(AbstractBaseBlock):
    def transform(self, input_df):
        output_df = pd.DataFrame(
            {
                "breath_time": input_df['time_step'] - input_df['time_step'].shift(1),
                "u_in_time": input_df['u_in'] - input_df['u_in'].shift(1)
            }
        )
        output_df.loc[input_df['time_step'] == 0, 'breath_time'] = output_df['breath_time'].mean()
        output_df.loc[input_df['time_step'] == 0, 'u_in_time'] = output_df['u_in_time'].mean()
        cfg.cont_seq_cols += output_df.add_suffix(f'@{self.__class__.__name__}').columns.tolist()
        return output_df

class LagFeatures(AbstractBaseBlock):
    def transform(self, input_df):
        output_df = pd.DataFrame(
            {
                "u_in_lag1": input_df.groupby("breath_id")["u_in"].shift(1).fillna(0),
                "u_in_lag2": input_df.groupby("breath_id")["u_in"].shift(2).fillna(0),
                "u_in_lag3": input_df.groupby("breath_id")["u_in"].shift(3).fillna(0),
                "u_in_lag4": input_df.groupby("breath_id")["u_in"].shift(4).fillna(0),
                "u_in_lag-1": input_df.groupby("breath_id")["u_in"].shift(-1).fillna(0),
                "u_in_lag-2": input_df.groupby("breath_id")["u_in"].shift(-2).fillna(0),
                "u_in_lag-3": input_df.groupby("breath_id")["u_in"].shift(-3).fillna(0),
                "u_in_lag-4": input_df.groupby("breath_id")["u_in"].shift(-4).fillna(0),
                "u_out_lag1": input_df.groupby("breath_id")["u_out"].shift(1).fillna(0),
                "u_out_lag2": input_df.groupby("breath_id")["u_out"].shift(2).fillna(0),
                "u_out_lag3": input_df.groupby("breath_id")["u_out"].shift(3).fillna(0),
                "u_out_lag4": input_df.groupby("breath_id")["u_out"].shift(4).fillna(0),
                #"u_out_lag-1": input_df.groupby("breath_id")["u_out"].shift(-1).fillna(0),
                #"u_out_lag-2": input_df.groupby("breath_id")["u_out"].shift(-2).fillna(0),
                #"u_out_lag-3": input_df.groupby("breath_id")["u_out"].shift(-3).fillna(0),
                #"u_out_lag-4": input_df.groupby("breath_id")["u_out"].shift(-4).fillna(0),
            }
        )
        output_df["u_in_lag1_diff"] = output_df["u_in_lag1"] - input_df["u_in"]
        output_df["u_in_lag2_diff"] = output_df["u_in_lag2"] - input_df["u_in"]
        output_df["u_in_lag3_diff"] = output_df["u_in_lag3"] - input_df["u_in"]
        output_df["u_in_lag4_diff"] = output_df["u_in_lag4"] - input_df["u_in"]

        output_df["u_in_rolling_mean2"] = input_df[["breath_id", "u_in"]].groupby("breath_id").rolling(2).mean()["u_in"].reset_index(drop=True)
        output_df["u_in_rolling_mean4"] = input_df[["breath_id", "u_in"]].groupby("breath_id").rolling(4).mean()["u_in"].reset_index(drop=True)
        output_df["u_in_rolling_mean10"] = input_df[["breath_id", "u_in"]].groupby("breath_id").rolling(10).mean()["u_in"].reset_index(drop=True)
        if not cfg.debug:
            output_df["u_in_rolling_max2"] = input_df[["breath_id", "u_in"]].groupby("breath_id").rolling(2).max()["u_in"].reset_index(drop=True)
            output_df["u_in_rolling_max4"] = input_df[["breath_id", "u_in"]].groupby("breath_id").rolling(4).max()["u_in"].reset_index(drop=True)
            output_df["u_in_rolling_max10"] = input_df[["breath_id", "u_in"]].groupby("breath_id").rolling(10).max()["u_in"].reset_index(drop=True)
            output_df["u_in_rolling_min2"] = input_df[["breath_id", "u_in"]].groupby("breath_id").rolling(2).min()["u_in"].reset_index(drop=True)
            output_df["u_in_rolling_min4"] = input_df[["breath_id", "u_in"]].groupby("breath_id").rolling(4).min()["u_in"].reset_index(drop=True)
            output_df["u_in_rolling_min10"] = input_df[["breath_id", "u_in"]].groupby("breath_id").rolling(10).min()["u_in"].reset_index(drop=True)
            output_df["u_in_rolling_std2"] = input_df[["breath_id", "u_in"]].groupby("breath_id").rolling(2).std()["u_in"].reset_index(drop=True)
            output_df["u_in_rolling_std4"] = input_df[["breath_id", "u_in"]].groupby("breath_id").rolling(4).std()["u_in"].reset_index(drop=True)
            output_df["u_in_rolling_std10"] = input_df[["breath_id", "u_in"]].groupby("breath_id").rolling(10).std()["u_in"].reset_index(drop=True)
        for col in output_df.columns:
            output_df[col] = output_df[col].fillna(output_df[col].mean())
        cfg.cont_seq_cols += output_df.add_suffix(f'@{self.__class__.__name__}').columns.tolist()
        return output_df


def run_blocks(input_df, blocks, y=None, test=False):
    out_df = pd.DataFrame()

    print(decorate('start run blocks...'))

    with Timer(prefix='run test={}'.format(test)):
        for block in blocks:
            with Timer(prefix='out_df shape: {} \t- {}'.format(out_df.shape, str(block))):
                if not test:
                    out_i = block.fit(input_df.copy(), y=y)
                else:
                    out_i = block.transform(input_df.copy())

            assert len(input_df) == len(out_i), block
            name = block.__class__.__name__
            out_df = pd.concat([out_df, out_i.add_suffix(f'@{name}')], axis=1)
    print(f"out_df shape: {out_df.shape}")

    return pd.concat([input_df, out_df], axis=1)


def apply_normalization(cfg, train, test):
    train_col_order = ["u_out"] + train.columns.drop("u_out").tolist()
    test_col_order = ["u_out"] + test.columns.drop("u_out").tolist()
    train = train[train_col_order]
    test = test[test_col_order]
    scaler = RobustScaler()
    scaler_targets = [col for col in cfg.cont_seq_cols if col != "u_out"]
    print(f"Apply Standerd Scaler these columns: {scaler_targets}")
    for scaler_target in tqdm(scaler_targets):
        scaler.fit(train.loc[:,[scaler_target]])
        train.loc[:,[scaler_target]] = scaler.transform(train.loc[:,[scaler_target]])
        test.loc[:,[scaler_target]] = scaler.transform(test.loc[:,[scaler_target]])
    # display(train.head())
    # display(test.head())
    return train, test