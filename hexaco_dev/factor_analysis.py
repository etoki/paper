import pandas as pd
from factor_analyzer import FactorAnalyzer
import numpy as np
from sklearn.preprocessing import StandardScaler

# データの読み込み
# file_path_in = 'csv/fa_240_ch.csv'
file_path_in = 'csv/hexaco-jp_facet_for_fa_v7.csv'
# file_path_in = 'csv/hexaco-jp24_facet_for_fa_v2.csv'
data = pd.read_csv(file_path_in)

# ID列を削除
data_clean = data.drop(columns=["userid"])
# data_clean = data.drop(columns=["メールアドレス"])

# 欠損値の補完（今回は平均値で補完）
data_clean = data_clean.fillna(data_clean.mean())

# データの標準化
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_clean)

# 因子分析を実施、主成分分析とスクリープロットから因子数決定
fa = FactorAnalyzer(n_factors=6, rotation="varimax") 
fa.fit(data_scaled)

# 因子負荷量の取得
factor_loadings = fa.loadings_

# 共通性の取得
communalities = fa.get_communalities()

# 固有値の取得
eigenvalues, _ = fa.get_eigenvalues()

# ω係数の計算 (信頼性分析)
def calculate_omega(loadings, communalities):
    # ω係数は主に因子負荷量と共通性から計算されます
    factor_variances = np.sum(np.square(loadings), axis=0)
    total_variance = np.sum(communalities)
    omega = factor_variances / total_variance
    return omega

# omega = calculate_omega(factor_loadings, communalities)

# DataFrameへの変換
factor_loadings_df = pd.DataFrame(factor_loadings, index=data_clean.columns, columns=[f"Factor{i+1}" for i in range(factor_loadings.shape[1])])
# communalities_df = pd.DataFrame(communalities, index=data_clean.columns, columns=["Communalities"])
# eigenvalues_df = pd.DataFrame(eigenvalues, columns=["Eigenvalues"])
# omega_df = pd.DataFrame(omega, index=[f"Factor{i+1}" for i in range(len(omega))], columns=["Omega"])

# CSVファイルとして出力
factor_loadings_df.to_csv('csv/hexaco-jp_fa_v7.csv', index=True)
# factor_loadings_df.to_csv('csv/hexaco-jp24_fa_v2.csv', index=True)

# factor_loadings_df.to_csv('csv/factor_loadings_240_ch.csv', index=True)

# communalities_df.to_csv('csv/communalities.csv', index=True)
# eigenvalues_df.to_csv('csv/eigenvalues.csv', index=False)
# omega_df.to_csv('csv/omega.csv', index=True)

print("ファイルはCSVとして保存されました。")
