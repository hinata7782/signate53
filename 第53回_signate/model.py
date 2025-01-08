import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import xgboost as xgb
from catboost import CatBoostRegressor
from catboost import Pool
from sklearn.linear_model import LinearRegression


df_train = pd.read_csv('data\\train.csv')
df_test = pd.read_csv('data\\test.csv')

# データの情報を表示
df_train.info()


#前処理
object = df_train.select_dtypes(include = 'object')

#ワンホットエンコーディング
object_one = ['MS Zoning', 'Land Contour', 'Lot Config', 'Bldg Type', 'House Style',
                'Roof Style', 'Foundation', 'Electrical', 'Paved Drive', 'Sale Type', 'Sale Condition',
                'Neighborhood', 'Exterior 1st', 'Exterior 2nd']
object_one_selected = object[object_one]

one_hot_encoder = OneHotEncoder()
object_one_hot = one_hot_encoder.fit_transform(object_one_selected).toarray()
object_one_hot_df = pd.DataFrame(object_one_hot, columns=one_hot_encoder.get_feature_names_out(object_one_selected.columns))
object_one_hot_df = object_one_hot_df.drop('Exterior 2nd_AsbShng', axis=1)
object_one_hot_df = object_one_hot_df.drop('Neighborhood_BrDale', axis=1)


#ラベルエンコーディング
object_label_columns = ['Lot Shape', 'Exter Qual', 'Heating QC', 'Central Air', 'Kitchen Qual']

object_label_encoder = LabelEncoder()

object_label_selected = {}

for i in object_label_columns:
    object_label = object_label_encoder.fit_transform(object[i])
    object_label_selected[i] = object_label

object_label_df = pd.DataFrame(object_label_selected)


object_df = pd.concat([object_one_hot_df, object_label_df], axis=1)


#数値データ
other_data = df_train.drop(object.columns, axis=1)
other_data = other_data.drop('SalePrice', axis=1)

other_data_df = pd.DataFrame()

#四分位範囲（IQR）によるクリッピング
for column in other_data.columns:
    if other_data[column].dtype in ['int64', 'float64']:
        q1 = other_data[column].quantile(0.25)
        q3 = other_data[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        other_data_df[column] = other_data[column].clip(lower=lower_bound, upper=upper_bound)


#データの結合
df_train_concat_xgb = pd.concat([object_df, other_data_df], axis=1)
df_train_concat_xgb = df_train_concat_xgb.drop('Electrical_FuseF', axis=1)
df_train_concat_cat = pd.concat([object, other_data_df], axis=1)
response_variable = df_train['SalePrice']

#相関係数の計算
high_count = 0
low_count = 0

for target in df_train_concat.columns:
    r = np.corrcoef(df_train_concat_xgb[target], response_variable)[0][1]
    if r > 0.5 or r < -0.5:
        print(f"強い＿相関関係 = {r:.4f}")
        high_count += 1
    else:
        print(f"弱い＿相関関係 = {r:.4f}")
        low_count += 1

print(f"強い相関関係の数 = {high_count}")
print(f"弱い相関関係の数 = {low_count}")


#クロスバリデーション
#非線形回帰モデル(CatBoost)を使用
x_xgb = df_train_concat_xgb
x_cat = df_train_concat_cat
x_cat.columns = [f"x_cat_{col}" for col in x_cat.columns]
x_xgb.columns = [f"x_xgb_{col}" for col in x_xgb.columns]
x_combined = pd.concat([x_cat, x_xgb], axis=1)
y = response_variable

kf = KFold(n_splits=5, shuffle=True, random_state=0)
rmse_list = []  # RMSEを保存するリスト
mae_list = []   # MAEを保存するリスト


# モデルの定義
categorical_columns = x_cat.select_dtypes(include=['object']).columns
cat_features = [x_cat.columns.get_loc(col) for col in categorical_columns]

cat_model  = CatBoostRegressor(
        iterations=60,
        learning_rate=0.1,
        depth=10,
        loss_function='RMSE',
        custom_metric=['MAE'],
        logging_level='Silent'
    )

params = {
        'objective': 'reg:squarederror',         # 回帰タスク
        'eval_metric': ['rmse', 'mae'],          #モデル評価
        'max_depth': 4,                          # ツリーの深さ 
        'learning_rate': 0.03,                    # 学習率
        'subsample': 0.7                         # サブサンプル比率
    }


# スタッキング用の特徴量とターゲットラベルの格納用リスト
stacked_features = []
y_val_all = []  # すべてのターゲットラベルを保存

# クロスバリデーションでの処理
for train_index, val_index in kf.split(x_combined):
    x_train_fold = x_combined.iloc[train_index]
    x_val_fold = x_combined.iloc[val_index]
    
    # x_cat と x_xgb を正しく分割
    x_train_fold_cat = x_train_fold.filter(regex='^x_cat_')  # x_cat のみ
    x_val_fold_cat = x_val_fold.filter(regex='^x_cat_')      # x_cat のみ

    x_train_fold_xgb = x_train_fold.filter(regex='^x_xgb_')  # x_xgb のみ
    x_val_fold_xgb = x_val_fold.filter(regex='^x_xgb_')    # x_xgb のみ
    
    # y_train_fold と y_val_fold
    y_train_fold = y.iloc[train_index] 
    y_val_fold = y.iloc[val_index]
    
    #catboost用
    cat_train_data = Pool(data=x_train_fold_cat, label=y_train_fold, cat_features=cat_features)
    cat_eval_data = Pool(data=x_val_fold_cat, label=y_val_fold, cat_features=cat_features)

    #xgboost用
    xgb_train_data = xgb.DMatrix(x_train_fold_xgb, label=y_train_fold)
    xgb_eval_data = xgb.DMatrix(x_val_fold_xgb, label=y_val_fold)

    cat_model.fit(cat_train_data)

    evals = [(xgb_train_data, 'train'), (xgb_eval_data, 'eval')]
    xgb_model = xgb.train(
        params,
        xgb_train_data,
        num_boost_round=100,
        early_stopping_rounds=10,
        evals=evals,
    )

    # 予測
    cat_preds = cat_model.predict(cat_eval_data)
    xgb_preds = xgb_model.predict(xgb_eval_data)
    
    # スタッキング用の特徴量を作成
    stacked_features.append(np.column_stack((cat_preds, xgb_preds)))
    y_val_all.append(y_val_fold)  # 各foldのy_val_foldを保存

    # RMSEとMAEの計算（CatBoostとXGBoostの両方）
    rmse_cat = np.sqrt(mean_squared_error(y_val_fold, cat_preds))
    mae_cat = np.mean(np.abs(y_val_fold - cat_preds))
    rmse_xgb = np.sqrt(mean_squared_error(y_val_fold, xgb_preds))
    mae_xgb = np.mean(np.abs(y_val_fold - xgb_preds))

    # リストに追加
    rmse_list.append((rmse_cat, rmse_xgb))
    mae_list.append((mae_cat, mae_xgb))

# スタッキング特徴量を一つの配列にまとめる
stacked_features = np.vstack(stacked_features)  # 2次元の配列に変換
y_val_all = np.hstack(y_val_all)  # y_val_allも一つにまとめる

# メタモデル（線形回帰など）の学習
meta_model = LinearRegression()
meta_model.fit(stacked_features, y_val_all)

# 最終予測
final_preds = meta_model.predict(stacked_features)

# 最終評価
final_rmse = np.sqrt(mean_squared_error(y_val_all, final_preds))
final_mae = np.mean(np.abs(y_val_all - final_preds))

# 最終評価を出力
print(f"Final RMSE: {final_rmse}")
print(f"Final MAE: {final_mae}")

# 各foldのRMSEとMAEを平均値として計算
mean_rmse_cat = np.mean([x[0] for x in rmse_list])
mean_rmse_xgb = np.mean([x[1] for x in rmse_list])
mean_mae_cat = np.mean([x[0] for x in mae_list])
mean_mae_xgb = np.mean([x[1] for x in mae_list])

# 平均値を表示
print(f"Average RMSE (CatBoost): {mean_rmse_cat}")
print(f"Average RMSE (XGBoost): {mean_rmse_xgb}")
print(f"Average MAE (CatBoost): {mean_mae_cat}")
print(f"Average MAE (XGBoost): {mean_mae_xgb}")
