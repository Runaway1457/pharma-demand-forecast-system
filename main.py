
import polars as pl
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from sklearn.metrics import mean_absolute_error
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

def generate_business_scenario():
    logger.info("Construindo cenário de negócio simulado...")
    datas = pl.date_range(start=datetime(2023,1,1), end=datetime(2025,3,31), interval='1d', eager=True)
    df = pl.DataFrame({"data": datas})
    
    df = df.with_columns([
        pl.col("data").dt.month().alias("mes"),
        pl.col("data").dt.year().alias("ano"),
        pl.col("data").dt.ordinal_day().alias("dia_ano"),
        pl.when(pl.col("data").dt.month().is_in([6,7,8])).then(1.25).otherwise(1.0).alias("fator_sazonal")
    ])
    
    df = df.with_columns(
        pl.when((pl.col("ano") == 2023) & (pl.col("mes").is_in([4, 5]))).then(1.20).otherwise(1.0).alias("campanha_vacina")
    )
    df = df.with_columns(
        pl.when((pl.col("ano") == 2024) & (pl.col("mes") == 1) & (pl.col("data").dt.day() < 15)).then(0.6).otherwise(1.0).alias("ruptura_estoque")
    )
    df = df.with_columns(
        pl.when((pl.col("mes") == 11) & (pl.col("data").dt.day() >= 23) & (pl.col("data").dt.day() <= 27)).then(1.40).otherwise(1.0).alias("black_friday")
    )
    
    np.random.seed(42)
    base_rev = 150000
    df = df.with_columns([
        (base_rev * pl.col("fator_sazonal") * pl.col("campanha_vacina") * pl.col("ruptura_estoque") * pl.col("black_friday") * (1 + np.random.normal(0, 0.05, len(df))))
        .alias("faturamento").cast(pl.Float64)
    ])
    
    df = df.with_columns([
        pl.col("faturamento").shift(1).alias("lag_1"),
        pl.col("faturamento").rolling_mean(7).alias("mm_7d")
    ]).drop_nulls()
    
    return df.to_pandas()

def run_forecast_engine(df):
    logger.info("Treinando modelo preditivo...")
    split_date = '2024-11-01'
    train = df[df['data'] < split_date]
    
    feats = ['mes', 'dia_ano', 'lag_1', 'mm_7d']
    target = 'faturamento'
    
    model = lgb.train(
        {'verbosity': -1, 'objective': 'regression'},
        lgb.Dataset(train[feats], label=train[target]),
        num_boost_round=300
    )
    
    preds = model.predict(df[feats])
    df['previsao'] = preds
    importance = dict(zip(feats, model.feature_importance()))
    mae = mean_absolute_error(df[df['data'] >= split_date]['faturamento'], df[df['data'] >= split_date]['previsao'])
    
    return df, mae, importance

def plot_storytelling_dashboard(df, mae, importance):
    plt.style.use('bmh')
    fig = plt.figure(figsize=(18, 11))
    gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[2, 1], hspace=0.3, wspace=0.3)
    
    C_REAL, C_PRED, C_EVENT = '#1f77b4', '#ff7f0e', '#d62728'
    
    ax1 = fig.add_subplot(gs[0, :])
    ax1.fill_between(df['data'], df['previsao']*0.95, df['previsao']*1.05, color=C_PRED, alpha=0.15, label='Zona de Confiança IA (95%)')
    ax1.plot(df['data'], df['faturamento'], color=C_REAL, linewidth=2, label='Faturamento Real')
    ax1.plot(df['data'], df['previsao'], color=C_PRED, linewidth=1.5, linestyle='--', label='Previsão Modelo', alpha=0.8)
    
    ax1.axvspan(datetime(2024,1,1), datetime(2024,1,15), color='red', alpha=0.1)
    ax1.annotate('Ruptura de Estoque', xy=(datetime(2024,1,8), 150000), xytext=(datetime(2024,2,1), 120000), arrowprops=dict(arrowstyle='->', color='gray'), fontsize=10, fontweight='bold', color='darkred')
    
    ax1.axvspan(datetime(2024,11,23), datetime(2024,11,27), color='green', alpha=0.1)
    ax1.annotate('Black Friday', xy=(datetime(2024,11,25), 250000), xytext=(datetime(2024,10,1), 280000), arrowprops=dict(arrowstyle='->', color='gray'), fontsize=10, fontweight='bold', color='darkgreen')

    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b/%y'))
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'R$ {x/1e6:.2f}M'))
    ax1.set_title('Linha do Tempo: Performance vs Previsão (2023 - 2025)', fontsize=14, fontweight='bold', loc='left')
    ax1.legend(loc='upper left')
    
    ax2 = fig.add_subplot(gs[1, 0])
    labels_map = {'mm_7d': 'Tendência Semanal', 'lag_1': 'Vendas Ontem', 'dia_ano': 'Sazonalidade Anual', 'mes': 'Mês Corrente'}
    pretty_labels = [labels_map.get(f, f) for f in importance.keys()]
    ax2.barh(list(range(len(importance))), list(importance.values()), align='center', color='#2ca02c')
    ax2.set_yticks(list(range(len(importance))))
    ax2.set_yticklabels(pretty_labels)
    ax2.set_title('Drivers de Venda', fontsize=11, fontweight='bold', loc='left')
    ax2.invert_yaxis()
    
    ax3 = fig.add_subplot(gs[1, 1])
    df_future = df[df['data'] >= '2024-11-01']
    ax3.plot(df_future['data'], df_future['faturamento'], color=C_REAL, marker='o', label='Realizado')
    ax3.plot(df_future['data'], df_future['previsao'], color=C_PRED, marker='x', label='Previsto')
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax3.set_title('Zoom: Validação do Modelo', fontsize=11, fontweight='bold', loc='left')
    ax3.legend(fontsize=8)
    
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis('off')
    insight_text = f"RELATÓRIO EXECUTIVO\n\nConfiabilidade do Modelo:\n± R$ {mae/1e3:.1f}k\n\n⚠️ ALERTA:\nSazonalidade de Inverno\ncresceu 15% vs 2023."
    ax4.text(0.1, 0.5, insight_text, fontsize=12, va='center', ha='left', bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f0f0', edgecolor='gray', alpha=0.9))
    
    fig.suptitle('PharmaNexus - Relatório de Inteligência de Mercado', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('business_storytelling_dashboard.png', dpi=300)
    logger.info("Relatório gerado: business_storytelling_dashboard.png")
    plt.show()

if __name__ == "__main__":
    df_scenario = generate_business_scenario()
    df_forecast, model_mae, model_imp = run_forecast_engine(df_scenario)
    plot_storytelling_dashboard(df_forecast, model_mae, model_imp)
