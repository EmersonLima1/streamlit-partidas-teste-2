# Importação das bibliotecas necessárias

# Importando a classe LabelEncoder do módulo sklearn.preprocessing para transformar variáveis categóricas em numéricas
from sklearn.preprocessing import LabelEncoder

# Importando a função train_test_split do módulo sklearn.model_selection para dividir os dados em conjuntos de treinamento e teste
from sklearn.model_selection import train_test_split

# Importando a classe MultiOutputClassifier do módulo sklearn.multioutput para lidar com problemas de múltiplas saídas em classificação
from sklearn.multioutput import MultiOutputClassifier

# Importando a classe RandomForestClassifier do módulo sklearn.ensemble para usar um algoritmo de aprendizado de máquina baseado em floresta aleatória
from sklearn.ensemble import RandomForestClassifier

# Importando as funções accuracy_score, precision_score, recall_score e f1_score do módulo sklearn.metrics para avaliar o desempenho do modelo de classificação
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Importando a função tabulate do módulo tabulate para formatar a saída dos resultados em uma tabela
from tabulate import tabulate

# Importando o módulo pandas e dando o alias pd para trabalhar com dados em formato de dataframe
import pandas as pd

# Importando o módulo numpy e dando o alias np para trabalhar com cálculos matemáticos em arrays
import numpy as np

# Importando a classe datetime do módulo datetime
from datetime import datetime

# Importando o módulo re para usar expressões regulares
import re

# Configurando a exibição do pandas para mostrar todas as colunas do dataframe
pd.set_option("display.max_columns", None)

# Importando o módulo warnings para evitar a exibição de mensagens de aviso
import warnings

# Ignorando todos os tipos de mensagens de aviso
warnings.filterwarnings("ignore")

# Método que gera o dataframe completo com as médias e os resultados das partidas. Este método é usado no modelo de machine learning e para filtrar as últimas n partidas em casa e fora de casa dos times

def df_completo_partidas_casa_fora(partidas_df):

  partidas_df['date_GMT'] = pd.to_datetime(partidas_df['date_GMT'])
  partidas_df['date_GMT'] = partidas_df['date_GMT'].dt.date

  # Agrupa os dados por time da casa e calcula as médias acumuladas
  agrupado_casa = partidas_df.groupby(['home_team_name'])
  # média de gols marcados pelo time da casa
  media_gols_casa = agrupado_casa['home_team_goal_count'].apply(lambda x: x.expanding().mean())
  # média de gols marcados pelo time da casa no primeiro tempo
  media_gols_primeiro_casa = agrupado_casa['home_team_goal_count_half_time'].apply(lambda x: x.expanding().mean())
  # média de escanteios pelo time da casa
  media_escanteios_casa = agrupado_casa['home_team_corner_count'].apply(lambda x: x.expanding().mean())
  # média de cartões amarelos pelo time da casa
  media_cartoes_amarelos_casa = agrupado_casa['home_team_yellow_cards'].apply(lambda x: x.expanding().mean())
  # média de cartões vermelhos pelo time da casa
  media_cartoes_vermelhos_casa = agrupado_casa['home_team_red_cards'].apply(lambda x: x.expanding().mean())
  # média de cartões pelo time da casa no primeiro tempo
  media_cartoes_primeiro_casa = agrupado_casa['home_team_first_half_cards'].apply(lambda x: x.expanding().mean())
  # média de cartões pelo time da casa no segundo tempo
  media_cartoes_segundo_casa = agrupado_casa['home_team_second_half_cards'].apply(lambda x: x.expanding().mean())
  # média de chutes no gol pelo time da casa
  media_chutes_no_gol_casa = agrupado_casa['home_team_shots_on_target'].apply(lambda x: x.expanding().mean())
  # média de chutes fora do gol pelo time da casa
  media_chutes_fora_do_gol_casa = agrupado_casa['home_team_shots_off_target'].apply(lambda x: x.expanding().mean())
  # média de faltas do time da casa
  media_faltas_casa = agrupado_casa['home_team_fouls'].apply(lambda x: x.expanding().mean())
  # média de posse de bola do time da casa
  media_posse_bola_casa = agrupado_casa['home_team_possession'].apply(lambda x: x.expanding().mean())

  # Agrupa os dados por time de fora e calcula as médias acumuladas
  agrupado_fora = partidas_df.groupby(['away_team_name'])
  # média de gols marcados pelo time de fora
  media_gols_fora = agrupado_fora['away_team_goal_count'].apply(lambda x: x.expanding().mean())
  # média de gols marcados pelo time de fora no primeiro tempo
  media_gols_primeiro_fora = agrupado_fora['away_team_goal_count_half_time'].apply(lambda x: x.expanding().mean())
  # média de escanteios pelo time de fora
  media_escanteios_fora = agrupado_fora['away_team_corner_count'].apply(lambda x: x.expanding().mean())
  # média de cartões amarelos pelo time de fora
  media_cartoes_amarelos_fora = agrupado_fora['away_team_yellow_cards'].apply(lambda x: x.expanding().mean())
  # média de cartões vermelhos pelo time de fora
  media_cartoes_vermelhos_fora = agrupado_fora['away_team_red_cards'].apply(lambda x: x.expanding().mean())
  # média de cartões pelo time de fora no primeiro tempo
  media_cartoes_primeiro_fora = agrupado_fora['away_team_first_half_cards'].apply(lambda x: x.expanding().mean())
  # média de cartões pelo time de fora no segundo tempo
  media_cartoes_segundo_fora = agrupado_fora['away_team_second_half_cards'].apply(lambda x: x.expanding().mean())
  # média de chutes no gol pelo time de fora
  media_chutes_no_gol_fora = agrupado_fora['away_team_shots_on_target'].apply(lambda x: x.expanding().mean())
  # média de chutes fora do gol pelo time de fora
  media_chutes_fora_do_gol_fora = agrupado_fora['away_team_shots_off_target'].apply(lambda x: x.expanding().mean())
  # média de faltas do time de fora
  media_faltas_fora = agrupado_fora['away_team_fouls'].apply(lambda x: x.expanding().mean())
  # média de posse de bola do time de fora
  media_posse_bola_fora = agrupado_fora['away_team_possession'].apply(lambda x: x.expanding().mean())

  # adicionando as médias ao DataFrame original para cada time

  # média de gols marcados
  partidas_df['media_gols_casa'] = media_gols_casa.values
  partidas_df['media_gols_fora'] = media_gols_fora.values

  # média de gols marcados no primeiro tempo
  partidas_df['media_gols_primeiro_casa'] = media_gols_primeiro_casa.values
  partidas_df['media_gols_primeiro_fora'] = media_gols_primeiro_fora.values

  # média de escanteios
  partidas_df['media_escanteios_casa'] = media_escanteios_casa.values
  partidas_df['media_escanteios_fora'] = media_escanteios_fora.values

  # média de cartões amarelos
  partidas_df['media_cartoes_amarelos_casa'] = media_cartoes_amarelos_casa.values
  partidas_df['media_cartoes_amarelos_fora'] = media_cartoes_amarelos_fora.values

  # média de cartões vermelhos
  partidas_df['media_cartoes_vermelhos_casa'] = media_cartoes_vermelhos_casa.values
  partidas_df['media_cartoes_vermelhos_fora'] = media_cartoes_vermelhos_fora.values

  # média de cartões no primeiro tempo
  partidas_df['media_cartoes_primeiro_casa'] = media_cartoes_primeiro_casa.values
  partidas_df['media_cartoes_primeiro_fora'] = media_cartoes_primeiro_fora.values

  # média de cartões no segundo tempo
  partidas_df['media_cartoes_segundo_casa'] = media_cartoes_segundo_casa.values
  partidas_df['media_cartoes_segundo_fora'] = media_cartoes_segundo_fora.values

  # média de chutes no gol
  partidas_df['media_chutes_no_gol_casa'] = media_chutes_no_gol_casa.values
  partidas_df['media_chutes_no_gol_fora'] = media_chutes_no_gol_fora.values

  # média de chutes para fora
  partidas_df['media_chutes_fora_do_gol_casa'] = media_chutes_fora_do_gol_casa.values
  partidas_df['media_chutes_fora_do_gol_fora'] = media_chutes_fora_do_gol_fora.values

  # média de faltas
  partidas_df['media_faltas_casa'] = media_faltas_casa.values
  partidas_df['media_faltas_fora'] = media_faltas_fora.values

  # média de posse de bola
  partidas_df['media_posse_bola_casa'] = media_posse_bola_casa.values
  partidas_df['media_posse_bola_fora'] = media_posse_bola_fora.values
      
  # criação do dataframe com as médias de gols
  df_partidas_casa_fora_1 = pd.DataFrame(columns=['time_casa', 'time_fora','arbitro', 'data_partida', 'rodada','gols_time_casa', 'gols_time_fora','gols_time_casa_primeiro',
                               'gols_time_fora_primeiro','escanteios_casa','escanteios_fora','cartoes_amarelos_casa','cartoes_amarelos_fora',
                               'cartoes_vermelhos_casa','cartoes_vermelhos_fora','cartoes_primeiro_casa','cartoes_primeiro_fora', 'cartoes_segundo_casa','cartoes_segundo_fora',
                               'media_gols_casa','media_gols_fora','media_gols_primeiro_casa','media_gols_primeiro_fora','media_escanteios_casa',
                               'media_escanteios_fora','media_cartoes_amarelos_casa','media_cartoes_amarelos_fora','media_cartoes_vermelhos_casa',
                               'media_cartoes_vermelhos_fora','media_cartoes_primeiro_casa','media_cartoes_primeiro_fora','media_cartoes_segundo_casa',
                               'media_cartoes_segundo_fora','media_chutes_no_gol_casa','media_chutes_no_gol_fora','media_chutes_fora_do_gol_casa','media_chutes_fora_do_gol_fora',
                               'media_faltas_casa','media_faltas_fora','media_posse_bola_casa','media_posse_bola_fora'])

  # loop pelo dataframe original
  for index, row in partidas_df.iterrows():

      # atribuindo os valores encontrados anteriormente em um dicionário
      novas_colunas_1 = {
        'time_casa': row['home_team_name'],
        'time_fora': row['away_team_name'],
        'arbitro': row['referee'],
        'data_partida': row['date_GMT'],
        'rodada': row['Game Week'],
        'gols_time_casa': row['home_team_goal_count'],
        'gols_time_fora': row['away_team_goal_count'],
        'gols_time_casa_primeiro': row['home_team_goal_count_half_time'],
        'gols_time_fora_primeiro': row['away_team_goal_count_half_time'],
        'escanteios_casa': row['home_team_corner_count'],
        'escanteios_fora': row['away_team_corner_count'],
        'cartoes_amarelos_casa': row['home_team_yellow_cards'],
        'cartoes_amarelos_fora': row['away_team_yellow_cards'],
        'cartoes_vermelhos_casa': row['home_team_red_cards'],
        'cartoes_vermelhos_fora': row['away_team_red_cards'],
        'cartoes_primeiro_casa': row['home_team_first_half_cards'],
        'cartoes_primeiro_fora': row['away_team_first_half_cards'],
        'cartoes_segundo_casa': row['home_team_second_half_cards'],
        'cartoes_segundo_fora': row['away_team_second_half_cards'],
        'media_gols_casa': row['media_gols_casa'],
        'media_gols_fora': row['media_gols_fora'],
        'media_gols_primeiro_casa': row['media_gols_primeiro_casa'],
        'media_gols_primeiro_fora': row['media_gols_primeiro_fora'],
        'media_escanteios_casa': row['media_escanteios_casa'],
        'media_escanteios_fora': row['media_escanteios_fora'],
        'media_cartoes_amarelos_casa': row['media_cartoes_amarelos_casa'],
        'media_cartoes_amarelos_fora': row['media_cartoes_amarelos_fora'],
        'media_cartoes_vermelhos_casa': row['media_cartoes_vermelhos_casa'],
        'media_cartoes_vermelhos_fora': row['media_cartoes_vermelhos_fora'],
        'media_cartoes_primeiro_casa': row['media_cartoes_primeiro_casa'],
        'media_cartoes_primeiro_fora': row['media_cartoes_primeiro_fora'],
        'media_cartoes_segundo_casa': row['media_cartoes_segundo_casa'],
        'media_cartoes_segundo_fora': row['media_cartoes_segundo_fora'],
        'media_chutes_no_gol_casa': row['media_chutes_no_gol_casa'],
        'media_chutes_no_gol_fora': row['media_chutes_no_gol_fora'],
        'media_chutes_fora_do_gol_casa': row['media_chutes_fora_do_gol_casa'],
        'media_chutes_fora_do_gol_fora': row['media_chutes_fora_do_gol_fora'],
        'media_faltas_casa': row['media_faltas_casa'],
        'media_faltas_fora': row['media_faltas_fora'],
        'media_posse_de_bola_casa': row['media_posse_bola_casa'],
        'media_posse_de_bola_fora': row['media_posse_bola_fora']
      }

      # adicionando o dicionário ao dataframe de médias
      df_partidas_casa_fora_1 = df_partidas_casa_fora_1.append(novas_colunas_1, ignore_index=True)

  # Criando colunas de resultados que serão as variáveis target do algoritmo de Machine Learning
  df_partidas_casa_fora_1['resultado_partida'] = df_partidas_casa_fora_1.apply(lambda row: 'casa' if row['gols_time_casa'] > row['gols_time_fora'] else ('fora' if row['gols_time_casa'] < row['gols_time_fora'] else 'empate'), axis=1)
  df_partidas_casa_fora_1['resultado_intervalo'] = df_partidas_casa_fora_1.apply(lambda row: 'casa' if row['gols_time_casa_primeiro'] > row['gols_time_fora_primeiro'] else ('fora' if row['gols_time_casa_primeiro'] < row['gols_time_fora_primeiro'] else 'empate'), axis=1)
  df_partidas_casa_fora_1['resultado_num_gols_over_under'] = df_partidas_casa_fora_1.apply(lambda row: 
    'menos de 1.5 gols' if (row['gols_time_casa'] + row['gols_time_fora']) <= 1.5 else
    'entre 1.5 e 2.5 gols' if (row['gols_time_casa'] + row['gols_time_fora']) > 1.5 and (row['gols_time_casa'] + row['gols_time_fora']) <= 2.5 else
    'mais de 2.5 gols', axis=1)
  df_partidas_casa_fora_1['resultado_ambas_equipes_marcaram'] = df_partidas_casa_fora_1.apply(lambda row: 'ambas marcaram' if row['gols_time_casa'] >= 1 and row['gols_time_fora'] >= 1 else 'ambas não marcaram', axis=1)
  df_partidas_casa_fora_1['resultado_num_cartoes_amarelos'] = df_partidas_casa_fora_1.apply(lambda row: 'mais de 3.5 cartões amarelos' if row['cartoes_amarelos_casa'] + row['cartoes_amarelos_fora'] > 3.5 else 'menos de 3.5 cartões amarelos', axis = 1)
  df_partidas_casa_fora_1['resultado_num_cartoes_vermelhos'] = df_partidas_casa_fora_1.apply(lambda row: 'aconteceu cartões vermelhos' if row['cartoes_vermelhos_casa'] > 0 or row['cartoes_vermelhos_fora'] > 0 else 'não aconteceu cartões vermelhos', axis = 1)
  df_partidas_casa_fora_1['resultado_num_cartoes_totais'] = df_partidas_casa_fora_1.apply(lambda row: 'mais de 4.5 cartões totais' if row['cartoes_amarelos_casa'] + row['cartoes_amarelos_fora'] + row['cartoes_vermelhos_casa'] + row['cartoes_vermelhos_casa'] > 4.5 else 'menos de 4.5 cartões totais', axis = 1)
  df_partidas_casa_fora_1['resultado_ambas_equipes_receberam_cartoes'] = df_partidas_casa_fora_1.apply(lambda row: 'ambas receberam' if (row['cartoes_amarelos_casa'] > 0 and row['cartoes_amarelos_fora'] > 0) or (row['cartoes_vermelhos_casa'] > 0 and row['cartoes_vermelhos_fora'] > 0)  or (row['cartoes_amarelos_casa'] > 0 and row['cartoes_vermelhos_fora'] > 0) or (row['cartoes_vermelhos_casa'] > 0 and row['cartoes_amarelos_fora'] > 0) else 'ambas não receberam', axis=1)
  df_partidas_casa_fora_1['resultado_cartoes_ambos_tempos'] = df_partidas_casa_fora_1.apply(lambda row: 'aconteceu ambos tempos' if (row['cartoes_primeiro_casa'] > 0 or row['cartoes_primeiro_fora'] > 0) and (row['cartoes_segundo_casa'] > 0 or row['cartoes_segundo_fora'] > 0) else 'nao aconteceu ambos tempos', axis=1)
  df_partidas_casa_fora_1['resultado_num_escanteios'] = df_partidas_casa_fora_1.apply(lambda row: 
    'menos de 7.5 escanteios' if (row['escanteios_casa'] + row['escanteios_fora']) <= 7.5 else
    'entre 7.5 e 8.5 escanteios' if (row['escanteios_casa'] + row['escanteios_fora']) > 7.5 and (row['escanteios_casa'] + row['escanteios_casa']) <= 8.5 else
    'entre 8.5 e 9.5 escanteios' if (row['escanteios_casa'] + row['escanteios_fora']) > 8.5 and (row['escanteios_casa'] + row['escanteios_casa']) <= 9.5 else
    'mais de 9.5 escanteios', axis=1)
  df_partidas_casa_fora_1['resultado_num_cartoes_primeiro'] = df_partidas_casa_fora_1.apply(lambda row: 'mais de 1.5 cartões no primeiro tempo' if (row['cartoes_primeiro_casa'] + row['cartoes_primeiro_fora']) > 1.5 else 'menos de 1.5 cartões no primeiro tempo', axis=1) 
  df_partidas_casa_fora_1['resultado_num_cartoes_segundo'] = df_partidas_casa_fora_1.apply(lambda row: 'mais de 1.5 cartões no segundo tempo' if (row['cartoes_segundo_casa'] + row['cartoes_segundo_fora']) > 1.5 else 'menos de 1.5 cartões no segundo tempo',axis=1) 

  df_partidas_casa_fora_2 = df_partidas_casa_fora_1
  df_partidas_casa_fora_3 = df_partidas_casa_fora_1
  df_partidas_casa_fora_4 = df_partidas_casa_fora_1

  # selecionando apenas as colunas que serão usadas para treinar o modelo
  df_partidas_casa_fora_2 = df_partidas_casa_fora_2.reindex(columns=['time_casa', 'time_fora','arbitro', 'data_partida', 'rodada', 'media_gols_casa','media_gols_fora','media_gols_primeiro_casa',
       'media_gols_primeiro_fora','media_escanteios_casa', 'media_escanteios_fora','media_cartoes_amarelos_casa', 'media_cartoes_amarelos_fora',
       'media_cartoes_vermelhos_casa', 'media_cartoes_vermelhos_fora','media_cartoes_primeiro_casa', 'media_cartoes_primeiro_fora',
       'media_cartoes_segundo_casa', 'media_cartoes_segundo_fora','media_chutes_no_gol_casa', 'media_chutes_no_gol_fora','media_chutes_fora_do_gol_casa',
       'media_chutes_fora_do_gol_fora','media_faltas_casa', 'media_faltas_fora','media_posse_de_bola_casa','media_posse_de_bola_fora','resultado_partida',
       'resultado_intervalo', 'resultado_num_gols_over_under', 'resultado_ambas_equipes_marcaram', 'resultado_num_cartoes_amarelos', 'resultado_num_cartoes_vermelhos', 
       'resultado_num_cartoes_totais', 'resultado_ambas_equipes_receberam_cartoes', 'resultado_cartoes_ambos_tempos', 'resultado_num_escanteios',
       'resultado_num_cartoes_primeiro','resultado_num_cartoes_segundo'])
  
  df_partidas_casa_fora_3 = df_partidas_casa_fora_3.reindex(columns=['time_casa', 'time_fora', 'arbitro', 'data_partida','rodada', 'gols_time_casa', 'gols_time_fora',
                                                                     'gols_time_casa_primeiro','gols_time_fora_primeiro','escanteios_casa','escanteios_fora',
                                                                     'cartoes_amarelos_casa','cartoes_amarelos_fora','cartoes_vermelhos_casa','cartoes_vermelhos_fora',
                                                                     'cartoes_primeiro_casa','cartoes_primeiro_fora','cartoes_segundo_casa','cartoes_segundo_fora'])
  
  # Criar coluna 'resultado' com o placar no formato desejado
  df_partidas_casa_fora_4['Geral'] = df_partidas_casa_fora_4['time_casa'] + ' ' + df_partidas_casa_fora_4['gols_time_casa'].astype(str) + ' x ' + df_partidas_casa_fora_4['gols_time_fora'].astype(str) + ' ' + df_partidas_casa_fora_4['time_fora']

  # Reordenar as colunas do DataFrame
  df_partidas_casa_fora_4 = df_partidas_casa_fora_4.reindex(columns=['time_casa','time_fora','Geral'])

  return (df_partidas_casa_fora_2, df_partidas_casa_fora_3, df_partidas_casa_fora_4)

# Método que gera o dataframe com as médias e os resultados das partidas, porém esse método é relacionado a partidas em geral dos times

def df_completo_partidas_gerais(partidas_df):

  partidas_df['date_GMT'] = pd.to_datetime(partidas_df['date_GMT'])
  partidas_df['date_GMT'] = partidas_df['date_GMT'].dt.date

  # criar uma lista com todos os times presentes no dataset
  teams = list(set(partidas_df['home_team_name']) | set(partidas_df['away_team_name']))

  # criar um dicionário para armazenar as médias de cada time
  medias_partidas = {team: {'total_gols': 0, 'total_gols_primeiro': 0, 'total_escanteios': 0, 'total_cartoes_amarelos': 0, 'total_cartoes_vermelhos': 0,
                            'total_cartoes_primeiro': 0, 'total_cartoes_segundo': 0, 'total_chutes_no_gol': 0, 'total_fora_do_gol': 0, 'total_faltas': 0,
                            'total_posse_de_bola': 0,'partidas_jogadas': 0, 'media_gols': 0, 'media_gols_primeiro': 0,'media_escanteios': 0, 'media_cartoes_amarelos': 0, 'media_cartoes_vermelhos' : 0, 'media_cartoes_primeiro': 0, 'media_cartoes_segundo': 0, 'media_chutes_no_gol': 0, 'media_chute_fora_do_gol': 0, 'media_faltas': 0, 'media_posse_bola': 0} for team in teams}

  # calcular as médias acumulativas de gols para cada time
  for index, row in partidas_df.iterrows():

      home_team = row['home_team_name']
      away_team = row['away_team_name']
      home_gols = row['home_team_goal_count']
      away_gols = row['away_team_goal_count']
      home_gols_primeiro = row['home_team_goal_count_half_time']
      away_gols_primeiro = row['away_team_goal_count_half_time']
      home_corner = row['home_team_corner_count']
      away_corner = row['away_team_corner_count']
      home_yellow_cards = row['home_team_yellow_cards']
      away_yellow_cards = row['away_team_yellow_cards']
      home_red_cards = row['home_team_red_cards']
      away_red_cards = row['away_team_red_cards']
      home_cartoes_primeiro = row['home_team_first_half_cards']
      away_cartoes_primeiro = row['away_team_first_half_cards']
      home_cartoes_segundo = row['home_team_second_half_cards']
      away_cartoes_segundo = row['away_team_second_half_cards']
      home_chutes_no_gol = row['home_team_shots_on_target']
      away_chutes_no_gol = row['away_team_shots_on_target']
      home_chutes_fora_do_gol = row['home_team_shots_off_target']
      away_chutes_fora_do_gol = row['away_team_shots_off_target']
      home_fouls = row['home_team_fouls']
      away_fouls = row['away_team_fouls']
      home_possession = row['home_team_possession']
      away_possession = row['away_team_possession']

      # atualizar as médias do time da casa
      medias_partidas[home_team]['total_gols'] += home_gols
      medias_partidas[home_team]['total_gols_primeiro'] += home_gols_primeiro
      medias_partidas[home_team]['total_escanteios'] += home_corner
      medias_partidas[home_team]['total_cartoes_amarelos'] += home_yellow_cards
      medias_partidas[home_team]['total_cartoes_vermelhos'] += home_red_cards
      medias_partidas[home_team]['total_cartoes_primeiro'] += home_cartoes_primeiro
      medias_partidas[home_team]['total_cartoes_segundo'] += home_cartoes_segundo
      medias_partidas[home_team]['total_chutes_no_gol'] += home_chutes_no_gol
      medias_partidas[home_team]['total_fora_do_gol'] += home_chutes_fora_do_gol
      medias_partidas[home_team]['total_faltas'] += home_fouls
      medias_partidas[home_team]['total_posse_de_bola'] += home_possession

      medias_partidas[home_team]['partidas_jogadas'] += 1

      medias_partidas[home_team]['media_gols'] = medias_partidas[home_team]['total_gols'] / medias_partidas[home_team]['partidas_jogadas']
      medias_partidas[home_team]['media_gols_primeiro'] = medias_partidas[home_team]['total_gols_primeiro'] / medias_partidas[home_team]['partidas_jogadas']
      medias_partidas[home_team]['media_escanteios'] = medias_partidas[home_team]['total_escanteios'] / medias_partidas[home_team]['partidas_jogadas']
      medias_partidas[home_team]['media_cartoes_amarelos'] = medias_partidas[home_team]['total_cartoes_amarelos'] / medias_partidas[home_team]['partidas_jogadas']
      medias_partidas[home_team]['media_cartoes_vermelhos'] = medias_partidas[home_team]['total_cartoes_vermelhos'] / medias_partidas[home_team]['partidas_jogadas']
      medias_partidas[home_team]['media_cartoes_primeiro'] = medias_partidas[home_team]['total_cartoes_primeiro'] / medias_partidas[home_team]['partidas_jogadas']
      medias_partidas[home_team]['media_cartoes_segundo'] = medias_partidas[home_team]['total_cartoes_segundo'] / medias_partidas[home_team]['partidas_jogadas']
      medias_partidas[home_team]['media_chutes_no_gol'] = medias_partidas[home_team]['total_chutes_no_gol'] / medias_partidas[home_team]['partidas_jogadas']
      medias_partidas[home_team]['media_chutes_fora_do_gol'] = medias_partidas[home_team]['total_fora_do_gol'] / medias_partidas[home_team]['partidas_jogadas']
      medias_partidas[home_team]['media_faltas'] = medias_partidas[home_team]['total_faltas'] / medias_partidas[home_team]['partidas_jogadas']
      medias_partidas[home_team]['media_posse_de_bola'] = medias_partidas[home_team]['total_posse_de_bola'] / medias_partidas[home_team]['partidas_jogadas']

      # atualizar as médias do time visitante
      medias_partidas[away_team]['total_gols'] += away_gols
      medias_partidas[away_team]['total_gols'] += away_gols
      medias_partidas[away_team]['total_gols_primeiro'] += away_gols_primeiro
      medias_partidas[away_team]['total_escanteios'] += away_corner
      medias_partidas[away_team]['total_cartoes_amarelos'] += away_yellow_cards
      medias_partidas[away_team]['total_cartoes_vermelhos'] += away_red_cards
      medias_partidas[away_team]['total_cartoes_primeiro'] += away_cartoes_primeiro
      medias_partidas[away_team]['total_cartoes_segundo'] += away_cartoes_segundo
      medias_partidas[away_team]['total_chutes_no_gol'] += away_chutes_no_gol
      medias_partidas[away_team]['total_fora_do_gol'] += away_chutes_fora_do_gol
      medias_partidas[away_team]['total_faltas'] += away_fouls
      medias_partidas[away_team]['total_posse_de_bola'] += away_possession

      medias_partidas[away_team]['partidas_jogadas'] += 1

      medias_partidas[away_team]['media_gols'] = medias_partidas[away_team]['total_gols'] / medias_partidas[away_team]['partidas_jogadas']
      medias_partidas[away_team]['media_gols_primeiro'] = medias_partidas[away_team]['total_gols_primeiro'] / medias_partidas[away_team]['partidas_jogadas']
      medias_partidas[away_team]['media_escanteios'] = medias_partidas[away_team]['total_escanteios'] / medias_partidas[away_team]['partidas_jogadas']
      medias_partidas[away_team]['media_cartoes_amarelos'] = medias_partidas[away_team]['total_cartoes_amarelos'] / medias_partidas[away_team]['partidas_jogadas']
      medias_partidas[away_team]['media_cartoes_vermelhos'] = medias_partidas[away_team]['total_cartoes_vermelhos'] / medias_partidas[away_team]['partidas_jogadas']
      medias_partidas[away_team]['media_cartoes_primeiro'] = medias_partidas[away_team]['total_cartoes_primeiro'] / medias_partidas[away_team]['partidas_jogadas']
      medias_partidas[away_team]['media_cartoes_segundo'] = medias_partidas[away_team]['total_cartoes_segundo'] / medias_partidas[away_team]['partidas_jogadas']
      medias_partidas[away_team]['media_chutes_no_gol'] = medias_partidas[away_team]['total_chutes_no_gol'] / medias_partidas[away_team]['partidas_jogadas']
      medias_partidas[away_team]['media_chutes_fora_do_gol'] = medias_partidas[away_team]['total_fora_do_gol'] / medias_partidas[away_team]['partidas_jogadas']
      medias_partidas[away_team]['media_faltas'] = medias_partidas[away_team]['total_faltas'] / medias_partidas[away_team]['partidas_jogadas']
      medias_partidas[away_team]['media_posse_de_bola'] = medias_partidas[away_team]['total_posse_de_bola'] / medias_partidas[away_team]['partidas_jogadas']

      # adicionar as médias ao dataframe original
      partidas_df.at[index, 'media_gols_casa'] = round(medias_partidas[home_team]['media_gols'], 2)
      partidas_df.at[index, 'media_gols_fora'] = round(medias_partidas[away_team]['media_gols'], 2)
      partidas_df.at[index, 'media_gols_primeiro_casa'] = round(medias_partidas[home_team]['media_gols_primeiro'], 2)
      partidas_df.at[index, 'media_gols_primeiro_fora'] = round(medias_partidas[away_team]['media_gols_primeiro'], 2)
      partidas_df.at[index, 'media_escanteios_casa'] = round(medias_partidas[home_team]['media_escanteios'], 2)
      partidas_df.at[index, 'media_escanteios_fora'] = round(medias_partidas[away_team]['media_escanteios'], 2)
      partidas_df.at[index, 'media_cartoes_amarelos_casa'] = round(medias_partidas[home_team]['media_cartoes_amarelos'], 2)
      partidas_df.at[index, 'media_cartoes_amarelos_fora'] = round(medias_partidas[away_team]['media_cartoes_amarelos'], 2)
      partidas_df.at[index, 'media_cartoes_vermelhos_casa'] = round(medias_partidas[home_team]['media_cartoes_vermelhos'], 2)
      partidas_df.at[index, 'media_cartoes_vermelhos_fora'] = round(medias_partidas[away_team]['media_cartoes_vermelhos'], 2)
      partidas_df.at[index, 'media_cartoes_primeiro_casa'] = round(medias_partidas[home_team]['media_cartoes_primeiro'], 2)
      partidas_df.at[index, 'media_cartoes_primeiro_fora'] = round(medias_partidas[away_team]['media_cartoes_primeiro'], 2)
      partidas_df.at[index, 'media_cartoes_segundo_casa'] = round(medias_partidas[home_team]['media_cartoes_segundo'], 2)
      partidas_df.at[index, 'media_cartoes_segundo_fora'] = round(medias_partidas[away_team]['media_cartoes_segundo'], 2)
      partidas_df.at[index, 'media_chutes_no_gol_casa'] = round(medias_partidas[home_team]['media_chutes_no_gol'], 2)
      partidas_df.at[index, 'media_chutes_no_gol_fora'] = round(medias_partidas[away_team]['media_chutes_no_gol'], 2)
      partidas_df.at[index, 'media_chutes_fora_do_gol_casa'] = round(medias_partidas[home_team]['media_chutes_fora_do_gol'], 2)
      partidas_df.at[index, 'media_chutes_fora_do_gol_fora'] = round(medias_partidas[away_team]['media_chutes_fora_do_gol'], 2)
      partidas_df.at[index, 'media_faltas_casa'] = round(medias_partidas[home_team]['media_faltas'], 2)
      partidas_df.at[index, 'media_faltas_fora'] = round(medias_partidas[away_team]['media_faltas'], 2)
      partidas_df.at[index, 'media_posse_de_bola_casa'] = round(medias_partidas[home_team]['media_posse_de_bola'], 2)
      partidas_df.at[index, 'media_posse_de_bola_fora'] = round(medias_partidas[away_team]['media_posse_de_bola'], 2)

  # criação do dataframe com as médias de gols
  df_partidas_gerais_1 = pd.DataFrame(columns=['time_casa', 'time_fora','arbitro', 'data_partida','rodada','gols_time_casa', 'gols_time_fora','gols_time_casa_primeiro',
                               'gols_time_fora_primeiro','escanteios_casa','escanteios_fora','cartoes_amarelos_casa','cartoes_amarelos_fora',
                               'cartoes_vermelhos_casa','cartoes_vermelhos_fora','cartoes_primeiro_casa','cartoes_primeiro_fora', 'cartoes_segundo_casa','cartoes_segundo_fora',
                               'media_gols_casa','media_gols_fora','media_gols_primeiro_casa','media_gols_primeiro_fora','media_escanteios_casa',
                               'media_escanteios_fora','media_cartoes_amarelos_casa','media_cartoes_amarelos_fora','media_cartoes_vermelhos_casa',
                               'media_cartoes_vermelhos_fora','media_cartoes_primeiro_casa','media_cartoes_primeiro_fora','media_cartoes_segundo_casa',
                               'media_cartoes_segundo_fora','media_chutes_no_gol_casa','media_chutes_no_gol_fora','media_chutes_fora_do_gol_casa','media_chutes_fora_do_gol_fora',
                               'media_faltas_casa','media_faltas_fora','media_posse_de_bola_casa','media_posse_de_bola_fora'])

  # loop pelo dataframe original
  for index, row in partidas_df.iterrows():

      # atribuindo os valores encontrados anteriormente em um dicionário
      novas_colunas_1 = {
        'time_casa': row['home_team_name'],
        'time_fora': row['away_team_name'],
        'arbitro': row['referee'],
        'data_partida': row['date_GMT'],
        'rodada': row['Game Week'],
        'gols_time_casa': row['home_team_goal_count'],
        'gols_time_fora': row['away_team_goal_count'],
        'gols_time_casa_primeiro': row['home_team_goal_count_half_time'],
        'gols_time_fora_primeiro': row['away_team_goal_count_half_time'],
        'escanteios_casa': row['home_team_corner_count'],
        'escanteios_fora': row['away_team_corner_count'],
        'cartoes_amarelos_casa': row['home_team_yellow_cards'],
        'cartoes_amarelos_fora': row['away_team_yellow_cards'],
        'cartoes_vermelhos_casa': row['home_team_red_cards'],
        'cartoes_vermelhos_fora': row['away_team_red_cards'],
        'cartoes_primeiro_casa': row['home_team_first_half_cards'],
        'cartoes_primeiro_fora': row['away_team_first_half_cards'],
        'cartoes_segundo_casa': row['home_team_second_half_cards'],
        'cartoes_segundo_fora': row['away_team_second_half_cards'],
        'media_gols_casa': row['media_gols_casa'],
        'media_gols_fora': row['media_gols_fora'],
        'media_gols_primeiro_casa': row['media_gols_primeiro_casa'],
        'media_gols_primeiro_fora': row['media_gols_primeiro_fora'],
        'media_escanteios_casa': row['media_escanteios_casa'],
        'media_escanteios_fora': row['media_escanteios_fora'],
        'media_cartoes_amarelos_casa': row['media_cartoes_amarelos_casa'],
        'media_cartoes_amarelos_fora': row['media_cartoes_amarelos_fora'],
        'media_cartoes_vermelhos_casa': row['media_cartoes_vermelhos_casa'],
        'media_cartoes_vermelhos_fora': row['media_cartoes_vermelhos_fora'],
        'media_cartoes_primeiro_casa': row['media_cartoes_primeiro_casa'],
        'media_cartoes_primeiro_fora': row['media_cartoes_primeiro_fora'],
        'media_cartoes_segundo_casa': row['media_cartoes_segundo_casa'],
        'media_cartoes_segundo_fora': row['media_cartoes_segundo_fora'],
        'media_chutes_no_gol_casa': row['media_chutes_no_gol_casa'],
        'media_chutes_no_gol_fora': row['media_chutes_no_gol_fora'],
        'media_chutes_fora_do_gol_casa': row['media_chutes_fora_do_gol_casa'],
        'media_chutes_fora_do_gol_fora': row['media_chutes_fora_do_gol_fora'],
        'media_faltas_casa': row['media_faltas_casa'],
        'media_faltas_fora': row['media_faltas_fora'],
        'media_posse_de_bola_casa': row['media_posse_de_bola_casa'],
        'media_posse_de_bola_fora': row['media_posse_de_bola_fora']
      }

      # adicionando o dicionário ao dataframe de médias
      df_partidas_gerais_1 = df_partidas_gerais_1.append(novas_colunas_1, ignore_index=True)

  # Criando colunas de resultados que serão as variáveis target do algoritmo de Machine Learning
  df_partidas_gerais_1['resultado_partida'] = df_partidas_gerais_1.apply(lambda row: 'casa' if row['gols_time_casa'] > row['gols_time_fora'] else ('fora' if row['gols_time_casa'] < row['gols_time_fora'] else 'empate'), axis=1)
  df_partidas_gerais_1['resultado_intervalo'] = df_partidas_gerais_1.apply(lambda row: 'casa' if row['gols_time_casa_primeiro'] > row['gols_time_fora_primeiro'] else ('fora' if row['gols_time_casa_primeiro'] < row['gols_time_fora_primeiro'] else 'empate'), axis=1)
  df_partidas_gerais_1['resultado_num_gols_over_under'] = df_partidas_gerais_1.apply(lambda row: 
    'menos de 1.5 gols' if (row['gols_time_casa'] + row['gols_time_fora']) <= 1.5 else
    'entre 1.5 e 2.5 gols' if (row['gols_time_casa'] + row['gols_time_fora']) > 1.5 and (row['gols_time_casa'] + row['gols_time_fora']) <= 2.5 else
    'mais de 2.5 gols', axis=1)
  df_partidas_gerais_1['resultado_ambas_equipes_marcaram'] = df_partidas_gerais_1.apply(lambda row: 'ambas marcaram' if row['gols_time_casa'] >= 1 and row['gols_time_fora'] >= 1 else 'ambas não marcaram', axis=1)
  df_partidas_gerais_1['resultado_num_cartoes_amarelos'] = df_partidas_gerais_1.apply(lambda row: 'mais de 3.5 cartões amarelos' if row['cartoes_amarelos_casa'] + row['cartoes_amarelos_fora'] > 3.5 else 'menos de 3.5 cartões amarelos', axis = 1)
  df_partidas_gerais_1['resultado_num_cartoes_vermelhos'] = df_partidas_gerais_1.apply(lambda row: 'aconteceu cartões vermelhos' if row['cartoes_vermelhos_casa'] > 0 or row['cartoes_vermelhos_fora'] > 0 else 'não aconteceu cartões vermelhos', axis = 1)
  df_partidas_gerais_1['resultado_num_cartoes_totais'] = df_partidas_gerais_1.apply(lambda row: 'mais de 4.5 cartões totais' if row['cartoes_amarelos_casa'] + row['cartoes_amarelos_fora'] + row['cartoes_vermelhos_casa'] + row['cartoes_vermelhos_casa'] > 4.5 else 'menos de 4.5 cartões totais', axis = 1)
  df_partidas_gerais_1['resultado_ambas_equipes_receberam_cartoes'] = df_partidas_gerais_1.apply(lambda row: 'ambas receberam' if (row['cartoes_amarelos_casa'] > 0 and row['cartoes_amarelos_fora'] > 0) or (row['cartoes_vermelhos_casa'] > 0 and row['cartoes_vermelhos_fora'] > 0)  or (row['cartoes_amarelos_casa'] > 0 and row['cartoes_vermelhos_fora'] > 0) or (row['cartoes_vermelhos_casa'] > 0 and row['cartoes_amarelos_fora'] > 0) else 'ambas não receberam', axis=1)
  df_partidas_gerais_1['resultado_cartoes_ambos_tempos'] = df_partidas_gerais_1.apply(lambda row: 'aconteceu ambos tempos' if (row['cartoes_primeiro_casa'] > 0 or row['cartoes_primeiro_fora'] > 0) and (row['cartoes_segundo_casa'] > 0 or row['cartoes_segundo_fora'] > 0) else 'nao aconteceu ambos tempos', axis=1)
  df_partidas_gerais_1['resultado_num_escanteios'] = df_partidas_gerais_1.apply(lambda row: 
    'menos de 7.5 escanteios' if (row['escanteios_casa'] + row['escanteios_fora']) <= 7.5 else
    'entre 7.5 e 8.5 escanteios' if (row['escanteios_casa'] + row['escanteios_fora']) > 7.5 and (row['escanteios_casa'] + row['escanteios_casa']) <= 8.5 else
    'entre 8.5 e 9.5 escanteios' if (row['escanteios_casa'] + row['escanteios_fora']) > 8.5 and (row['escanteios_casa'] + row['escanteios_casa']) <= 9.5 else
    'mais de 9.5 escanteios', axis=1)
  df_partidas_gerais_1['resultado_num_cartoes_primeiro'] = df_partidas_gerais_1.apply(lambda row: 'mais de 1.5 cartões no primeiro tempo' if (row['cartoes_primeiro_casa'] + row['cartoes_primeiro_fora']) > 1.5 else 'menos de 1.5 cartões no primeiro tempo', axis=1) 
  df_partidas_gerais_1['resultado_num_cartoes_segundo'] = df_partidas_gerais_1.apply(lambda row: 'mais de 1.5 cartões no segundo tempo' if (row['cartoes_segundo_casa'] + row['cartoes_segundo_fora']) > 1.5 else 'menos de 1.5 cartões no segundo tempo',axis=1) 

  df_partidas_gerais_2 = df_partidas_gerais_1
  df_partidas_gerais_3 = df_partidas_gerais_1
  df_partidas_gerais_4 = df_partidas_gerais_1

  # selecionando apenas as colunas que serão usadas para treinar o modelo
  df_partidas_gerais_2 = df_partidas_gerais_2.reindex(columns=['time_casa', 'time_fora','arbitro','data_partida','rodada','media_gols_casa','media_gols_fora','media_gols_primeiro_casa',
       'media_gols_primeiro_fora','media_escanteios_casa', 'media_escanteios_fora','media_cartoes_amarelos_casa', 'media_cartoes_amarelos_fora',
       'media_cartoes_vermelhos_casa', 'media_cartoes_vermelhos_fora','media_cartoes_primeiro_casa', 'media_cartoes_primeiro_fora',
       'media_cartoes_segundo_casa', 'media_cartoes_segundo_fora','media_chutes_no_gol_casa', 'media_chutes_no_gol_fora','media_chutes_fora_do_gol_casa',
       'media_chutes_fora_do_gol_fora','media_faltas_casa', 'media_faltas_fora','media_posse_de_bola_casa','media_posse_de_bola_fora','resultado_partida',
       'resultado_intervalo', 'resultado_num_gols_over_under','resultado_ambas_equipes_marcaram', 'resultado_num_cartoes_amarelos', 'resultado_num_cartoes_vermelhos', 
       'resultado_num_cartoes_totais', 'resultado_ambas_equipes_receberam_cartoes', 'resultado_cartoes_ambos_tempos', 'resultado_num_escanteios',
       'resultado_num_cartoes_primeiro','resultado_num_cartoes_segundo'])
  
  df_partidas_gerais_3 = df_partidas_gerais_3.reindex(columns=['time_casa', 'time_fora', 'arbitro', 'data_partida','rodada','gols_time_casa', 'gols_time_fora',
                                                                     'gols_time_casa_primeiro','gols_time_fora_primeiro','escanteios_casa','escanteios_fora',
                                                                     'cartoes_amarelos_casa','cartoes_amarelos_fora','cartoes_vermelhos_casa','cartoes_vermelhos_fora',
                                                                     'cartoes_primeiro_casa','cartoes_primeiro_fora','cartoes_segundo_casa','cartoes_segundo_fora'])
  
  # Criar coluna 'resultado' com o placar no formato desejado
  df_partidas_gerais_4['Geral'] = df_partidas_gerais_4['time_casa'] + ' ' + df_partidas_gerais_4['gols_time_casa'].astype(str) + ' x ' + df_partidas_gerais_4['gols_time_fora'].astype(str) + ' ' + df_partidas_gerais_4['time_fora']

  # Reordenar as colunas do DataFrame
  df_partidas_gerais_4 = df_partidas_gerais_4.reindex(columns=['time_casa','time_fora','Geral'])

  return (df_partidas_gerais_2, df_partidas_gerais_3, df_partidas_gerais_4)

# Método que gera o dataframe que vai ser utilizado como nova previsão (partidas em casa e fora)

def nova_previsao_partidas_casa_fora(ultimas_partidas_casa_fora, arbitro, time_casa, time_fora):

  # criar uma lista com todos os times presentes no dataset
  teams = list(set(ultimas_partidas_casa_fora['home_team_name']) | set(ultimas_partidas_casa_fora['away_team_name']))

  # criar um dicionário para armazenar as médias de cada time
  medias_partidas = {team: {'total_gols': 0, 'total_gols_primeiro': 0, 'total_escanteios': 0, 'total_cartoes_amarelos': 0, 'total_cartoes_vermelhos': 0,
                            'total_cartoes_primeiro': 0, 'total_cartoes_segundo': 0, 'total_chutes_no_gol': 0, 'total_fora_do_gol': 0, 'total_faltas': 0,
                            'total_posse_de_bola': 0,'partidas_jogadas': 0, 'media_gols': 0, 'media_gols_primeiro': 0,'media_escanteios': 0, 'media_cartoes_amarelos': 0, 'media_cartoes_vermelhos' : 0, 'media_cartoes_primeiro': 0, 'media_cartoes_segundo': 0, 'media_chutes_no_gol': 0, 'media_chute_fora_do_gol': 0, 'media_faltas': 0, 'media_posse_bola': 0} for team in teams}

  # calcular as médias acumulativas de gols para cada time
  for index, row in ultimas_partidas_casa_fora.iterrows():

      home_team = row['home_team_name']
      away_team = row['away_team_name']
      home_gols = row['home_team_goal_count']
      away_gols = row['away_team_goal_count']
      home_gols_primeiro = row['home_team_goal_count_half_time']
      away_gols_primeiro = row['away_team_goal_count_half_time']
      home_corner = row['home_team_corner_count']
      away_corner = row['away_team_corner_count']
      home_yellow_cards = row['home_team_yellow_cards']
      away_yellow_cards = row['away_team_yellow_cards']
      home_red_cards = row['home_team_red_cards']
      away_red_cards = row['away_team_red_cards']
      home_cartoes_primeiro = row['home_team_first_half_cards']
      away_cartoes_primeiro = row['away_team_first_half_cards']
      home_cartoes_segundo = row['home_team_second_half_cards']
      away_cartoes_segundo = row['away_team_second_half_cards']
      home_chutes_no_gol = row['home_team_shots_on_target']
      away_chutes_no_gol = row['away_team_shots_on_target']
      home_chutes_fora_do_gol = row['home_team_shots_off_target']
      away_chutes_fora_do_gol = row['away_team_shots_off_target']
      home_fouls = row['home_team_fouls']
      away_fouls = row['away_team_fouls']
      home_possession = row['home_team_possession']
      away_possession = row['away_team_possession']

      # atualizar as médias do time da casa
      medias_partidas[home_team]['total_gols'] += home_gols
      medias_partidas[home_team]['total_gols_primeiro'] += home_gols_primeiro
      medias_partidas[home_team]['total_escanteios'] += home_corner
      medias_partidas[home_team]['total_cartoes_amarelos'] += home_yellow_cards
      medias_partidas[home_team]['total_cartoes_vermelhos'] += home_red_cards
      medias_partidas[home_team]['total_cartoes_primeiro'] += home_cartoes_primeiro
      medias_partidas[home_team]['total_cartoes_segundo'] += home_cartoes_segundo
      medias_partidas[home_team]['total_chutes_no_gol'] += home_chutes_no_gol
      medias_partidas[home_team]['total_fora_do_gol'] += home_chutes_fora_do_gol
      medias_partidas[home_team]['total_faltas'] += home_fouls
      medias_partidas[home_team]['total_posse_de_bola'] += home_possession

      medias_partidas[home_team]['partidas_jogadas'] += 1

      medias_partidas[home_team]['media_gols'] = medias_partidas[home_team]['total_gols'] / medias_partidas[home_team]['partidas_jogadas']
      medias_partidas[home_team]['media_gols_primeiro'] = medias_partidas[home_team]['total_gols_primeiro'] / medias_partidas[home_team]['partidas_jogadas']
      medias_partidas[home_team]['media_escanteios'] = medias_partidas[home_team]['total_escanteios'] / medias_partidas[home_team]['partidas_jogadas']
      medias_partidas[home_team]['media_cartoes_amarelos'] = medias_partidas[home_team]['total_cartoes_amarelos'] / medias_partidas[home_team]['partidas_jogadas']
      medias_partidas[home_team]['media_cartoes_vermelhos'] = medias_partidas[home_team]['total_cartoes_vermelhos'] / medias_partidas[home_team]['partidas_jogadas']
      medias_partidas[home_team]['media_cartoes_primeiro'] = medias_partidas[home_team]['total_cartoes_primeiro'] / medias_partidas[home_team]['partidas_jogadas']
      medias_partidas[home_team]['media_cartoes_segundo'] = medias_partidas[home_team]['total_cartoes_segundo'] / medias_partidas[home_team]['partidas_jogadas']
      medias_partidas[home_team]['media_chutes_no_gol'] = medias_partidas[home_team]['total_chutes_no_gol'] / medias_partidas[home_team]['partidas_jogadas']
      medias_partidas[home_team]['media_chutes_fora_do_gol'] = medias_partidas[home_team]['total_fora_do_gol'] / medias_partidas[home_team]['partidas_jogadas']
      medias_partidas[home_team]['media_faltas'] = medias_partidas[home_team]['total_faltas'] / medias_partidas[home_team]['partidas_jogadas']
      medias_partidas[home_team]['media_posse_de_bola'] = medias_partidas[home_team]['total_posse_de_bola'] / medias_partidas[home_team]['partidas_jogadas']

      # atualizar as médias do time visitante
      medias_partidas[away_team]['total_gols'] += away_gols
      medias_partidas[away_team]['total_gols_primeiro'] += away_gols_primeiro
      medias_partidas[away_team]['total_escanteios'] += away_corner
      medias_partidas[away_team]['total_cartoes_amarelos'] += away_yellow_cards
      medias_partidas[away_team]['total_cartoes_vermelhos'] += away_red_cards
      medias_partidas[away_team]['total_cartoes_primeiro'] += away_cartoes_primeiro
      medias_partidas[away_team]['total_cartoes_segundo'] += away_cartoes_segundo
      medias_partidas[away_team]['total_chutes_no_gol'] += away_chutes_no_gol
      medias_partidas[away_team]['total_fora_do_gol'] += away_chutes_fora_do_gol
      medias_partidas[away_team]['total_faltas'] += away_fouls
      medias_partidas[away_team]['total_posse_de_bola'] += away_possession

      medias_partidas[away_team]['partidas_jogadas'] += 1

      medias_partidas[away_team]['media_gols'] = medias_partidas[away_team]['total_gols'] / medias_partidas[away_team]['partidas_jogadas']
      medias_partidas[away_team]['media_gols_primeiro'] = medias_partidas[away_team]['total_gols_primeiro'] / medias_partidas[away_team]['partidas_jogadas']
      medias_partidas[away_team]['media_escanteios'] = medias_partidas[away_team]['total_escanteios'] / medias_partidas[away_team]['partidas_jogadas']
      medias_partidas[away_team]['media_cartoes_amarelos'] = medias_partidas[away_team]['total_cartoes_amarelos'] / medias_partidas[away_team]['partidas_jogadas']
      medias_partidas[away_team]['media_cartoes_vermelhos'] = medias_partidas[away_team]['total_cartoes_vermelhos'] / medias_partidas[away_team]['partidas_jogadas']
      medias_partidas[away_team]['media_cartoes_primeiro'] = medias_partidas[away_team]['total_cartoes_primeiro'] / medias_partidas[away_team]['partidas_jogadas']
      medias_partidas[away_team]['media_cartoes_segundo'] = medias_partidas[away_team]['total_cartoes_segundo'] / medias_partidas[away_team]['partidas_jogadas']
      medias_partidas[away_team]['media_chutes_no_gol'] = medias_partidas[away_team]['total_chutes_no_gol'] / medias_partidas[away_team]['partidas_jogadas']
      medias_partidas[away_team]['media_chutes_fora_do_gol'] = medias_partidas[away_team]['total_fora_do_gol'] / medias_partidas[away_team]['partidas_jogadas']
      medias_partidas[away_team]['media_faltas'] = medias_partidas[away_team]['total_faltas'] / medias_partidas[away_team]['partidas_jogadas']
      medias_partidas[away_team]['media_posse_de_bola'] = medias_partidas[away_team]['total_posse_de_bola'] / medias_partidas[away_team]['partidas_jogadas']

      # adicionar as médias ao dataframe original
      ultimas_partidas_casa_fora.at[index, 'media_gols_casa'] = round(medias_partidas[home_team]['media_gols'], 2)
      ultimas_partidas_casa_fora.at[index, 'media_gols_fora'] = round(medias_partidas[away_team]['media_gols'], 2)
      ultimas_partidas_casa_fora.at[index, 'media_gols_primeiro_casa'] = round(medias_partidas[home_team]['media_gols_primeiro'], 2)
      ultimas_partidas_casa_fora.at[index, 'media_gols_primeiro_fora'] = round(medias_partidas[away_team]['media_gols_primeiro'], 2)
      ultimas_partidas_casa_fora.at[index, 'media_escanteios_casa'] = round(medias_partidas[home_team]['media_escanteios'], 2)
      ultimas_partidas_casa_fora.at[index, 'media_escanteios_fora'] = round(medias_partidas[away_team]['media_escanteios'], 2)
      ultimas_partidas_casa_fora.at[index, 'media_cartoes_amarelos_casa'] = round(medias_partidas[home_team]['media_cartoes_amarelos'], 2)
      ultimas_partidas_casa_fora.at[index, 'media_cartoes_amarelos_fora'] = round(medias_partidas[away_team]['media_cartoes_amarelos'], 2)
      ultimas_partidas_casa_fora.at[index, 'media_cartoes_vermelhos_casa'] = round(medias_partidas[home_team]['media_cartoes_vermelhos'], 2)
      ultimas_partidas_casa_fora.at[index, 'media_cartoes_vermelhos_fora'] = round(medias_partidas[away_team]['media_cartoes_vermelhos'], 2)
      ultimas_partidas_casa_fora.at[index, 'media_cartoes_primeiro_casa'] = round(medias_partidas[home_team]['media_cartoes_primeiro'], 2)
      ultimas_partidas_casa_fora.at[index, 'media_cartoes_primeiro_fora'] = round(medias_partidas[away_team]['media_cartoes_primeiro'], 2)
      ultimas_partidas_casa_fora.at[index, 'media_cartoes_segundo_casa'] = round(medias_partidas[home_team]['media_cartoes_segundo'], 2)
      ultimas_partidas_casa_fora.at[index, 'media_cartoes_segundo_fora'] = round(medias_partidas[away_team]['media_cartoes_segundo'], 2)
      ultimas_partidas_casa_fora.at[index, 'media_chutes_no_gol_casa'] = round(medias_partidas[home_team]['media_chutes_no_gol'], 2)
      ultimas_partidas_casa_fora.at[index, 'media_chutes_no_gol_fora'] = round(medias_partidas[away_team]['media_chutes_no_gol'], 2)
      ultimas_partidas_casa_fora.at[index, 'media_chutes_fora_do_gol_casa'] = round(medias_partidas[home_team]['media_chutes_fora_do_gol'], 2)
      ultimas_partidas_casa_fora.at[index, 'media_chutes_fora_do_gol_fora'] = round(medias_partidas[away_team]['media_chutes_fora_do_gol'], 2)
      ultimas_partidas_casa_fora.at[index, 'media_faltas_casa'] = round(medias_partidas[home_team]['media_faltas'], 2)
      ultimas_partidas_casa_fora.at[index, 'media_faltas_fora'] = round(medias_partidas[away_team]['media_faltas'], 2)
      ultimas_partidas_casa_fora.at[index, 'media_posse_de_bola_casa'] = round(medias_partidas[home_team]['media_posse_de_bola'], 2)
      ultimas_partidas_casa_fora.at[index, 'media_posse_de_bola_fora'] = round(medias_partidas[away_team]['media_posse_de_bola'], 2)

  # atribuindo os valores encontrados anteriormente em um dicionário
  dados = {
    'time_casa': time_casa,
    'time_fora': time_fora,
    'arbitro': arbitro,
    'media_gols_casa': round(medias_partidas[home_team]['media_gols'], 2),
    'media_gols_fora': round(medias_partidas[away_team]['media_gols'], 2),
    'media_gols_primeiro_casa': round(medias_partidas[home_team]['media_gols_primeiro'], 2),
    'media_gols_primeiro_fora': round(medias_partidas[away_team]['media_gols_primeiro'], 2),
    'media_escanteios_casa': round(medias_partidas[home_team]['media_escanteios'], 2),
    'media_escanteios_fora': round(medias_partidas[away_team]['media_escanteios'], 2),
    'media_cartoes_amarelos_casa': round(medias_partidas[home_team]['media_cartoes_amarelos'], 2),
    'media_cartoes_amarelos_fora': round(medias_partidas[away_team]['media_cartoes_amarelos'], 2),
    'media_cartoes_vermelhos_casa': round(medias_partidas[home_team]['media_cartoes_vermelhos'], 2),
    'media_cartoes_vermelhos_fora': round(medias_partidas[away_team]['media_cartoes_vermelhos'], 2),
    'media_cartoes_primeiro_casa': round(medias_partidas[home_team]['media_cartoes_primeiro'], 2),
    'media_cartoes_primeiro_fora': round(medias_partidas[away_team]['media_cartoes_primeiro'], 2),
    'media_cartoes_segundo_casa': round(medias_partidas[home_team]['media_cartoes_segundo'], 2),
    'media_cartoes_segundo_fora': round(medias_partidas[away_team]['media_cartoes_segundo'], 2),
    'media_chutes_no_gol_casa': round(medias_partidas[home_team]['media_chutes_no_gol'], 2),
    'media_chutes_no_gol_fora': round(medias_partidas[away_team]['media_chutes_no_gol'], 2),
    'media_chutes_fora_do_gol_casa': round(medias_partidas[home_team]['media_chutes_fora_do_gol'], 2),
    'media_chutes_fora_do_gol_fora': round(medias_partidas[away_team]['media_chutes_fora_do_gol'], 2),
    'media_faltas_casa': round(medias_partidas[home_team]['media_faltas'], 2),
    'media_faltas_fora': round(medias_partidas[away_team]['media_faltas'], 2),
    'media_posse_de_bola_casa': round(medias_partidas[home_team]['media_posse_de_bola'], 2),
    'media_posse_de_bola_fora': round(medias_partidas[away_team]['media_posse_de_bola'], 2)
}

  # adicionando o dicionário ao dataframe de médias
  df_nova_previsao_casa_fora = pd.DataFrame(dados, index=[0])
  
  return (df_nova_previsao_casa_fora)

# Método que gera o dataframe que vai ser utilizado como nova previsão (confrontos diretos)

def nova_previsao_direto(confrontos_diretos, arbitro, time_casa, time_fora):

  gols_do_time_da_casa_em_casa = 0
  gols_do_time_de_fora_fora_de_casa = 0
  gols_primeiro_do_time_da_casa_em_casa = 0
  gols_primeiro_do_time_de_fora_fora_de_casa = 0
  escanteios_do_time_da_casa_em_casa = 0
  escanteios_do_time_de_fora_fora_de_casa = 0
  cartoes_amarelos_time_da_casa_em_casa = 0
  cartoes_amarelos_time_de_fora_fora_de_casa = 0
  cartoes_vermelhos_time_da_casa_em_casa = 0
  cartoes_vermelhos_time_de_fora_fora_de_casa = 0
  cartoes_primeiro_tempo_time_da_casa_em_casa = 0
  cartoes_primeiro_tempo_time_de_fora_fora_de_casa = 0
  cartoes_segundo_tempo_time_da_casa_em_casa = 0
  cartoes_segundo_tempo_time_de_fora_fora_de_casa = 0
  chutes_no_gol_time_da_casa_em_casa = 0
  chutes_no_gol_time_de_fora_fora_de_casa = 0
  chutes_fora_do_gol_time_da_casa_em_casa = 0
  chutes_fora_do_gol_time_de_fora_fora_de_casa = 0
  faltas_do_time_da_casa_em_casa = 0
  faltas_do_time_de_fora_fora_de_casa = 0
  posse_de_bola_do_time_da_casa_em_casa = 0
  posse_de_bola_do_time_de_fora_fora_de_casa = 0

  gols_do_time_de_fora_em_casa = 0
  gols_do_time_da_casa_fora_de_casa = 0
  gols_primeiro_do_time_de_fora_em_casa = 0
  gols_primeiro_do_time_da_casa_fora_de_casa = 0
  escanteios_do_time_de_fora_em_casa = 0
  escanteios_do_time_da_casa_fora_de_casa = 0
  cartoes_amarelos_time_de_fora_em_casa = 0
  cartoes_amarelos_time_da_casa_fora_de_casa = 0
  cartoes_vermelhos_time_de_fora_em_casa = 0
  cartoes_vermelhos_time_da_casa_fora_de_casa = 0
  cartoes_primeiro_tempo_time_de_fora_em_casa = 0
  cartoes_primeiro_tempo_time_da_casa_fora_de_casa = 0
  cartoes_segundo_tempo_time_de_fora_em_casa = 0
  cartoes_segundo_tempo_time_da_casa_fora_de_casa = 0
  chutes_no_gol_time_de_fora_em_casa = 0
  chutes_no_gol_time_da_casa_fora_de_casa = 0
  chutes_fora_do_gol_time_de_fora_em_casa = 0
  chutes_fora_do_gol_time_da_casa_fora_de_casa = 0
  faltas_do_time_de_fora_em_casa = 0
  faltas_do_time_da_casa_fora_de_casa = 0
  posse_de_bola_do_time_de_fora_em_casa = 0
  posse_de_bola_do_time_da_casa_fora_de_casa = 0

  confrontos_diretos['date_GMT'] = pd.to_datetime(confrontos_diretos['date_GMT'])
  confrontos_diretos['date_GMT'] = confrontos_diretos['date_GMT'].dt.date

  # filtro para selecionar apenas as partidas do time mandante em casa
  partidas_casa_time = confrontos_diretos[(confrontos_diretos['home_team_name'] == time_casa)]

  # filtro para selecionar apenas as partidas do time visitante em casa
  partidas_fora_time = confrontos_diretos[(confrontos_diretos['home_team_name'] == time_fora)]

  if not partidas_casa_time.empty:

    # gols do time mandante jogando em casa
    gols_do_time_da_casa_em_casa = partidas_casa_time['home_team_goal_count'].sum()
    # gols do time visitante jogando fora de casa
    gols_do_time_de_fora_fora_de_casa = partidas_casa_time['away_team_goal_count'].sum()

    # gols no primeiro tempo do time mandante jogando em casa 
    gols_primeiro_do_time_da_casa_em_casa = partidas_casa_time['home_team_goal_count_half_time'].sum()
    # gols no primeiro tempo do time visitante jogando fora de casa
    gols_primeiro_do_time_de_fora_fora_de_casa = partidas_casa_time['away_team_goal_count_half_time'].sum()

    # escanteios do time mandante jogando em casa
    escanteios_do_time_da_casa_em_casa = partidas_casa_time['home_team_corner_count'].sum()
    # escanteios do time visitante jogando fora de casa
    escanteios_do_time_de_fora_fora_de_casa = partidas_casa_time['away_team_corner_count'].sum()

    # cartões amarelos do time mandante jogando em casa
    cartoes_amarelos_time_da_casa_em_casa = partidas_casa_time['home_team_yellow_cards'].sum()
    # cartões amarelos do time visitante jogando fora de casa
    cartoes_amarelos_time_de_fora_fora_de_casa = partidas_casa_time['away_team_yellow_cards'].sum()

    # cartões vermelhos do time mandante jogando em casa
    cartoes_vermelhos_time_da_casa_em_casa = partidas_casa_time['home_team_red_cards'].sum()
    # cartões vermelhos do time visitante jogando fora de casa
    cartoes_vermelhos_time_de_fora_fora_de_casa = partidas_casa_time['away_team_red_cards'].sum()
    
    # cartões no primeiro tempo do time mandante jogando em casa
    cartoes_primeiro_tempo_time_da_casa_em_casa = partidas_casa_time['home_team_first_half_cards'].sum()
    # cartões no primeiro tempo do time visitante jogando fora de casa
    cartoes_primeiro_tempo_time_de_fora_fora_de_casa = partidas_casa_time['away_team_first_half_cards'].sum()

    # cartões no segundo tempo do time mandante jogando em casa
    cartoes_segundo_tempo_time_da_casa_em_casa = partidas_casa_time['home_team_second_half_cards'].sum()
    # cartões no segundo tempo do time visitante jogando fora de casa
    cartoes_segundo_tempo_time_de_fora_fora_de_casa = partidas_casa_time['away_team_second_half_cards'].sum()

    # chutes no gol pelo time mandante jogando em casa
    chutes_no_gol_time_da_casa_em_casa = partidas_casa_time['home_team_shots_on_target'].sum()
    # chutes no gol pelo time visitante jogando fora de casa
    chutes_no_gol_time_de_fora_fora_de_casa = partidas_casa_time['away_team_shots_on_target'].sum()

    # chutes fora do gol pelo time mandante jogando em casa
    chutes_fora_do_gol_time_da_casa_em_casa = partidas_casa_time['home_team_shots_off_target'].sum()
    # chutes fora do gol pelo time visitante jogando fora de casa
    chutes_fora_do_gol_time_de_fora_fora_de_casa = partidas_casa_time['away_team_shots_off_target'].sum()

    # faltas cometidas pelo time mandante jogando em casa
    faltas_do_time_da_casa_em_casa = partidas_casa_time['home_team_fouls'].sum()
    # faltas cometidas pelo time visitante jogando fora de casa
    faltas_do_time_de_fora_fora_de_casa = partidas_casa_time['away_team_fouls'].sum()

    # posse de bola do time mandante jogando em casa
    posse_de_bola_do_time_da_casa_em_casa = partidas_casa_time['home_team_possession'].sum()
    # posse de bola do time visitante jogando fora de casa
    posse_de_bola_do_time_de_fora_fora_de_casa = partidas_casa_time['away_team_possession'].sum()


  elif not partidas_fora_time.empty:

    # gols do time visitante jogando em casa
    gols_do_time_de_fora_em_casa = partidas_fora_time['home_team_goal_count'].sum()
    # gols do time mandante jogando fora de casa
    gols_do_time_da_casa_fora_de_casa = partidas_fora_time['away_team_goal_count'].sum()
  
    # gols no primeiro tempo do time visitante jogando em casa
    gols_primeiro_do_time_de_fora_em_casa = partidas_fora_time['home_team_goal_count_half_time'].sum()
    # gols no primeiro tempo do time mandante jogando fora de casa
    gols_primeiro_do_time_da_casa_fora_de_casa = partidas_fora_time['away_team_goal_count_half_time'].sum()

    # escanteios do time visitante jogando em casa
    escanteios_do_time_de_fora_em_casa = partidas_fora_time['home_team_corner_count'].sum()
    # escanteios do time mandante jogando fora de casa
    escanteios_do_time_da_casa_fora_de_casa = partidas_fora_time['away_team_corner_count'].sum()

    # cartões amarelos do time visitante jogando em casa
    cartoes_amarelos_time_de_fora_em_casa = partidas_fora_time['home_team_yellow_cards'].sum()
    # cartões amarelos do time mandante jogando fora de casa
    cartoes_amarelos_time_da_casa_fora_de_casa = partidas_fora_time['away_team_yellow_cards'].sum()

    # cartões vermelhos do time visitante jogando em casa
    cartoes_vermelhos_time_de_fora_em_casa = partidas_fora_time['home_team_red_cards'].sum()
    # cartões vermelhos do time mandante jogando fora de casa
    cartoes_vermelhos_time_da_casa_fora_de_casa = partidas_fora_time['away_team_red_cards'].sum()

    # cartões no primeiro tempo do time visitante jogando em casa
    cartoes_primeiro_tempo_time_de_fora_em_casa = partidas_fora_time['home_team_first_half_cards'].sum()
    # cartões no primeiro tempo do time mandante jogando fora de casa
    cartoes_primeiro_tempo_time_da_casa_fora_de_casa = partidas_fora_time['away_team_first_half_cards'].sum()
    
    # cartões no segundo tempo do time visitante jogando em casa
    cartoes_segundo_tempo_time_de_fora_em_casa = partidas_fora_time['home_team_second_half_cards'].sum()
    # cartões no segundo tempo do time mandante jogando fora de casa
    cartoes_segundo_tempo_time_da_casa_fora_de_casa = partidas_fora_time['away_team_second_half_cards'].sum() 

    # chutes no gol pelo time visitante jogando em casa
    chutes_no_gol_time_de_fora_em_casa = partidas_fora_time['away_team_shots_on_target'].sum()
    # chutes no gol pelo time mandante jogando fora de casa
    chutes_no_gol_time_da_casa_fora_de_casa = partidas_fora_time['home_team_shots_on_target'].sum()

    # chutes fora do gol pelo time visitante jogando em casa
    chutes_fora_do_gol_time_de_fora_em_casa = partidas_fora_time['away_team_shots_off_target'].sum()
    # chutes fora do gol pelo time mandante jogando fora de casa
    chutes_fora_do_gol_time_da_casa_fora_de_casa = partidas_fora_time['home_team_shots_off_target'].sum()

    # faltas cometidas pelo time visitante jogando em casa
    faltas_do_time_de_fora_em_casa = partidas_fora_time['away_team_fouls'].sum()
    # faltas cometidas pelo time mandante jogando fora de casa
    faltas_do_time_da_casa_fora_de_casa = partidas_fora_time['home_team_fouls'].sum()

    # posse de bola do time visitante jogando em casa
    posse_de_bola_do_time_de_fora_em_casa = partidas_fora_time['away_team_possession'].sum()
    # posse de bola do time mandante jogando fora de casa
    posse_de_bola_do_time_da_casa_fora_de_casa = partidas_fora_time['home_team_possession'].sum()


  # Médias

  # média de gols do time da casa
  media_gols_casa = (gols_do_time_da_casa_em_casa + gols_do_time_da_casa_fora_de_casa) / (partidas_casa_time.shape[0] + partidas_fora_time.shape[0])
  # média de gols do time visitante
  media_gols_fora = (gols_do_time_de_fora_fora_de_casa + gols_do_time_de_fora_em_casa) / (partidas_casa_time.shape[0] + partidas_fora_time.shape[0])

  # média de gols no primeiro tempo do time da casa
  media_gols_primeiro_casa = (gols_primeiro_do_time_da_casa_em_casa + gols_primeiro_do_time_da_casa_fora_de_casa) / (partidas_casa_time.shape[0] + partidas_fora_time.shape[0])
  # média de gols no primeiro tempo do time visitante
  media_gols_primeiro_fora = (gols_primeiro_do_time_de_fora_fora_de_casa + gols_primeiro_do_time_de_fora_em_casa) / (partidas_casa_time.shape[0] + partidas_fora_time.shape[0])

  # média de escanteios do time da casa
  media_escanteios_casa = (escanteios_do_time_da_casa_em_casa + escanteios_do_time_da_casa_fora_de_casa) / (partidas_casa_time.shape[0] + partidas_fora_time.shape[0])
  # média de escanteios do time visitante
  media_escanteios_fora = (escanteios_do_time_de_fora_fora_de_casa + escanteios_do_time_de_fora_em_casa) / (partidas_casa_time.shape[0] + partidas_fora_time.shape[0])

  # média de cartões amarelos do time da casa
  media_cartoes_amarelos_casa = (cartoes_amarelos_time_da_casa_em_casa + cartoes_amarelos_time_da_casa_fora_de_casa) / (partidas_casa_time.shape[0] + partidas_fora_time.shape[0])
  # média de cartões amarelos do time visitante
  media_cartoes_amarelos_fora = (cartoes_amarelos_time_de_fora_fora_de_casa + cartoes_amarelos_time_de_fora_em_casa) / (partidas_casa_time.shape[0] + partidas_fora_time.shape[0])

  # média de cartões vermelhos do time da casa
  media_cartoes_vermelhos_casa = (cartoes_vermelhos_time_da_casa_em_casa + cartoes_vermelhos_time_da_casa_fora_de_casa) / (partidas_casa_time.shape[0] + partidas_fora_time.shape[0])
  # média de cartões vermelhos do time visitante
  media_cartoes_vermelhos_fora = (cartoes_vermelhos_time_de_fora_fora_de_casa + cartoes_vermelhos_time_de_fora_em_casa) / (partidas_casa_time.shape[0] + partidas_fora_time.shape[0])

  # média de cartões no primeiro tempo do time da casa
  media_cartoes_primeiro_tempo_casa = (cartoes_primeiro_tempo_time_da_casa_em_casa + cartoes_primeiro_tempo_time_da_casa_fora_de_casa) / (partidas_casa_time.shape[0] + partidas_fora_time.shape[0])
  # média de cartões no primeiro tempo do time visitante
  media_cartoes_primeiro_tempo_fora = (cartoes_primeiro_tempo_time_de_fora_fora_de_casa + cartoes_primeiro_tempo_time_de_fora_em_casa) / (partidas_casa_time.shape[0] + partidas_fora_time.shape[0])

  # média de cartões no segundo tempo do time da casa
  media_cartoes_segundo_tempo_casa = (cartoes_segundo_tempo_time_da_casa_em_casa + cartoes_segundo_tempo_time_da_casa_fora_de_casa) / (partidas_casa_time.shape[0] + partidas_fora_time.shape[0])
  # média de cartões no segundo tempo do time visitante
  media_cartoes_segundo_tempo_fora = (cartoes_segundo_tempo_time_de_fora_fora_de_casa + cartoes_segundo_tempo_time_de_fora_em_casa) / (partidas_casa_time.shape[0] + partidas_fora_time.shape[0])

  # média de chutes no gol pelo time da casa
  media_chutes_no_gol_time_da_casa = (chutes_no_gol_time_da_casa_em_casa + chutes_no_gol_time_da_casa_fora_de_casa) / (partidas_casa_time.shape[0] + partidas_fora_time.shape[0])
  # média de chutes no gol pelo time visitante
  media_chutes_no_gol_time_de_fora = (chutes_no_gol_time_de_fora_em_casa + chutes_no_gol_time_de_fora_fora_de_casa) / (partidas_casa_time.shape[0] + partidas_fora_time.shape[0])

  # média de chutes fora do gol pelo time da casa
  media_chutes_fora_do_gol_time_da_casa = (chutes_fora_do_gol_time_da_casa_em_casa + chutes_fora_do_gol_time_da_casa_fora_de_casa) / (partidas_casa_time.shape[0] + partidas_fora_time.shape[0])
  # média de chutes fora do gol pelo time visitante
  media_chutes_fora_do_gol_time_de_fora = (chutes_fora_do_gol_time_de_fora_em_casa + chutes_fora_do_gol_time_de_fora_fora_de_casa) / (partidas_casa_time.shape[0] + partidas_fora_time.shape[0])

  # média de faltas cometidas pelo time da casa
  media_faltas_do_time_da_casa = (faltas_do_time_da_casa_em_casa + faltas_do_time_da_casa_fora_de_casa) / (partidas_casa_time.shape[0] + partidas_fora_time.shape[0])
  # média de faltas cometidas pelo time visitante
  media_faltas_do_time_de_fora = (faltas_do_time_de_fora_em_casa + faltas_do_time_de_fora_fora_de_casa) / (partidas_casa_time.shape[0] + partidas_fora_time.shape[0])

  # média de posse de bola do time da casa
  media_posse_de_bola_do_time_da_casa = (posse_de_bola_do_time_da_casa_em_casa + posse_de_bola_do_time_da_casa_fora_de_casa) / (partidas_casa_time.shape[0] + partidas_fora_time.shape[0])
  # média de posse de bola do time visitante
  media_posse_de_bola_do_time_de_fora = (posse_de_bola_do_time_de_fora_em_casa + posse_de_bola_do_time_de_fora_fora_de_casa) / (partidas_casa_time.shape[0] + partidas_fora_time.shape[0])


  # atribuindo os valores encontrados anteriormente em um dicionário
  dados = {
    'time_casa': time_casa,
    'time_fora': time_fora,
    'arbitro': arbitro,
    'media_gols_casa': round(media_gols_casa, 2) ,
    'media_gols_fora': round(media_gols_fora, 2),
    'media_gols_primeiro_casa': round(media_gols_primeiro_casa, 2),
    'media_gols_primeiro_fora': round(media_gols_primeiro_fora, 2),
    'media_escanteios_casa': round(media_escanteios_casa, 2),
    'media_escanteios_fora': round(media_escanteios_fora, 2),
    'media_cartoes_amarelos_casa': round(media_cartoes_amarelos_casa, 2),
    'media_cartoes_amarelos_fora': round(media_cartoes_amarelos_fora, 2),
    'media_cartoes_vermelhos_casa': round(media_cartoes_vermelhos_casa, 2),
    'media_cartoes_vermelhos_fora': round(media_cartoes_vermelhos_fora, 2),
    'media_cartoes_primeiro_casa': round(media_cartoes_primeiro_tempo_casa, 2),
    'media_cartoes_primeiro_fora': round(media_cartoes_primeiro_tempo_fora, 2),
    'media_cartoes_segundo_casa': round(media_cartoes_segundo_tempo_casa, 2),
    'media_cartoes_segundo_fora': round(media_cartoes_segundo_tempo_fora, 2),
    'media_chutes_no_gol_casa': round(media_chutes_no_gol_time_da_casa, 2),
    'media_chutes_no_gol_fora': round(media_chutes_no_gol_time_de_fora, 2),
    'media_chutes_fora_do_gol_casa': round(media_chutes_fora_do_gol_time_da_casa, 2),
    'media_chutes_fora_do_gol_fora': round(media_chutes_fora_do_gol_time_de_fora, 2),
    'media_faltas_casa': round(media_faltas_do_time_da_casa, 2),
    'media_faltas_fora': round(media_faltas_do_time_de_fora, 2),
    'media_posse_de_bola_casa': round(media_posse_de_bola_do_time_da_casa, 2),
    'media_posse_de_bola_fora': round(media_posse_de_bola_do_time_de_fora, 2)
}

  # adicionando o dicionário ao dataframe de médias
  df_nova_previsao_direto = pd.DataFrame(dados, index=[0])
  
  return (df_nova_previsao_direto)

# Método que gera o dataframe que vai ser utilizado como nova previsão (partidas gerais)

def nova_previsao_partidas_gerais(ultimas_n_partidas, arbitro, n_partidas, time_casa, time_fora):

  # selecionar o nome do time da casa e do visitante
  home_team_1 = time_casa
  away_team_1 = time_fora

  gols_do_time_da_casa_em_casa = 0
  gols_do_time_da_casa_fora_de_casa = 0
  gols_do_time_de_fora_em_casa = 0
  gols_do_time_de_fora_fora_de_casa = 0

  gols_primeiro_do_time_da_casa_em_casa = 0
  gols_primeiro_do_time_da_casa_fora_de_casa = 0
  gols_primeiro_do_time_de_fora_em_casa = 0
  gols_primeiro_do_time_de_fora_fora_de_casa = 0

  escanteios_do_time_da_casa_em_casa = 0
  escanteios_do_time_da_casa_fora_de_casa = 0
  escanteios_do_time_de_fora_em_casa = 0
  escanteios_do_time_de_fora_fora_de_casa = 0

  cartoes_amarelos_time_da_casa_em_casa = 0
  cartoes_amarelos_time_da_casa_fora_de_casa = 0
  cartoes_amarelos_time_de_fora_em_casa = 0
  cartoes_amarelos_time_de_fora_fora_de_casa = 0

  cartoes_vermelhos_time_da_casa_em_casa = 0
  cartoes_vermelhos_time_da_casa_fora_de_casa = 0
  cartoes_vermelhos_time_de_fora_em_casa = 0
  cartoes_vermelhos_time_de_fora_fora_de_casa = 0

  cartoes_primeiro_tempo_time_da_casa_em_casa = 0
  cartoes_primeiro_tempo_time_da_casa_fora_de_casa = 0
  cartoes_primeiro_tempo_time_de_fora_em_casa = 0
  cartoes_primeiro_tempo_time_de_fora_fora_de_casa = 0

  cartoes_segundo_tempo_time_da_casa_em_casa = 0
  cartoes_segundo_tempo_time_da_casa_fora_de_casa = 0
  cartoes_segundo_tempo_time_de_fora_em_casa = 0
  cartoes_segundo_tempo_time_de_fora_fora_de_casa = 0

  chutes_no_gol_time_da_casa_em_casa = 0
  chutes_no_gol_time_da_casa_fora_de_casa = 0
  chutes_no_gol_time_de_fora_em_casa = 0
  chutes_no_gol_time_de_fora_fora_de_casa = 0

  chutes_fora_do_gol_time_da_casa_em_casa = 0
  chutes_fora_do_gol_time_da_casa_fora_de_casa = 0
  chutes_fora_do_gol_time_de_fora_em_casa = 0
  chutes_fora_do_gol_time_de_fora_fora_de_casa = 0

  faltas_do_time_da_casa_em_casa = 0
  faltas_do_time_da_casa_fora_de_casa = 0
  faltas_do_time_de_fora_em_casa = 0
  faltas_do_time_de_fora_fora_de_casa = 0

  posse_de_bola_do_time_da_casa_em_casa = 0
  posse_de_bola_do_time_da_casa_fora_de_casa = 0
  posse_de_bola_do_time_de_fora_em_casa = 0
  posse_de_bola_do_time_de_fora_fora_de_casa = 0

  # criar uma lista vazia para armazenar as últimas n partidas de cada time
  ultimas_partidas = []

  # iterar sobre cada linha do dataframe
  for idx, row in ultimas_n_partidas.iterrows():
      if row["home_team_name"] == home_team_1:
        gols_do_time_da_casa_em_casa += row["home_team_goal_count"]
        gols_primeiro_do_time_da_casa_em_casa += row["home_team_goal_count_half_time"]
        escanteios_do_time_da_casa_em_casa += row['home_team_corner_count']
        cartoes_amarelos_time_da_casa_em_casa += row['home_team_yellow_cards']
        cartoes_vermelhos_time_da_casa_em_casa += row['home_team_red_cards']
        cartoes_primeiro_tempo_time_da_casa_em_casa += row['home_team_first_half_cards']
        cartoes_segundo_tempo_time_da_casa_em_casa += row['home_team_second_half_cards']
        chutes_no_gol_time_da_casa_em_casa += row['home_team_shots_on_target']
        chutes_fora_do_gol_time_da_casa_em_casa += row['home_team_shots_off_target']
        faltas_do_time_da_casa_em_casa += row['home_team_fouls']
        posse_de_bola_do_time_da_casa_em_casa += row['home_team_possession']

      elif row["away_team_name"] == home_team_1:
        gols_do_time_da_casa_fora_de_casa += row["away_team_goal_count"]
        gols_primeiro_do_time_da_casa_fora_de_casa += row["away_team_goal_count_half_time"]
        escanteios_do_time_da_casa_fora_de_casa += row['away_team_corner_count']
        cartoes_amarelos_time_da_casa_fora_de_casa += row['away_team_yellow_cards']
        cartoes_vermelhos_time_da_casa_fora_de_casa += row['away_team_red_cards']
        cartoes_primeiro_tempo_time_da_casa_fora_de_casa += row['away_team_first_half_cards']
        cartoes_segundo_tempo_time_da_casa_fora_de_casa += row['away_team_second_half_cards']
        chutes_no_gol_time_da_casa_fora_de_casa += row['away_team_shots_on_target']
        chutes_fora_do_gol_time_da_casa_fora_de_casa += row['away_team_shots_off_target']
        faltas_do_time_da_casa_fora_de_casa += row['away_team_fouls']
        posse_de_bola_do_time_da_casa_fora_de_casa += row['away_team_possession']

      elif row["home_team_name"] == away_team_1:
        gols_do_time_de_fora_em_casa += row["home_team_goal_count"]
        gols_primeiro_do_time_de_fora_em_casa += row["home_team_goal_count_half_time"]
        escanteios_do_time_de_fora_em_casa += row['home_team_corner_count']
        cartoes_amarelos_time_de_fora_em_casa += row['home_team_yellow_cards']
        cartoes_vermelhos_time_de_fora_em_casa += row['home_team_red_cards']
        cartoes_primeiro_tempo_time_de_fora_em_casa += row['home_team_first_half_cards']
        cartoes_segundo_tempo_time_de_fora_em_casa += row['home_team_second_half_cards']
        chutes_no_gol_time_de_fora_em_casa += row['home_team_shots_on_target']
        chutes_fora_do_gol_time_de_fora_em_casa += row['home_team_shots_off_target']
        faltas_do_time_de_fora_em_casa += row['home_team_fouls']
        posse_de_bola_do_time_de_fora_em_casa += row['home_team_possession']

      elif row["away_team_name"] == away_team_1:
        gols_do_time_de_fora_fora_de_casa += row["away_team_goal_count"]
        gols_primeiro_do_time_de_fora_fora_de_casa += row["away_team_goal_count_half_time"]
        escanteios_do_time_de_fora_fora_de_casa += row['away_team_corner_count']
        cartoes_amarelos_time_de_fora_fora_de_casa += row['away_team_yellow_cards']
        cartoes_vermelhos_time_de_fora_fora_de_casa += row['away_team_red_cards']
        cartoes_primeiro_tempo_time_de_fora_fora_de_casa += row['away_team_first_half_cards']
        cartoes_segundo_tempo_time_de_fora_fora_de_casa += row['away_team_second_half_cards']
        chutes_no_gol_time_de_fora_fora_de_casa += row['away_team_shots_on_target']
        chutes_fora_do_gol_time_de_fora_fora_de_casa += row['away_team_shots_off_target']
        faltas_do_time_de_fora_fora_de_casa += row['away_team_fouls']
        posse_de_bola_do_time_de_fora_fora_de_casa += row['away_team_possession']


  # Médias

  # média de gols do time da casa
  media_gols_time_casa = (gols_do_time_da_casa_em_casa + gols_do_time_da_casa_fora_de_casa) / n_partidas
  # média de gols do time visitante
  media_gols_time_fora = (gols_do_time_de_fora_em_casa + gols_do_time_de_fora_fora_de_casa) / n_partidas

  # média de gols no primeiro tempo do time da casa
  media_gols_primeiro_tempo_casa = (gols_primeiro_do_time_da_casa_em_casa + gols_primeiro_do_time_da_casa_fora_de_casa) / n_partidas
  # média de gols no primeiro tempo do time visitante
  media_gols_primeiro_tempo_fora = (gols_primeiro_do_time_de_fora_em_casa + gols_primeiro_do_time_de_fora_fora_de_casa) / n_partidas

  # média de escanteios do time da casa
  media_escanteios_casa = (escanteios_do_time_da_casa_em_casa + escanteios_do_time_da_casa_fora_de_casa) / n_partidas
  # média de escanteios do time visitante
  media_escanteios_fora = (escanteios_do_time_de_fora_em_casa + escanteios_do_time_de_fora_fora_de_casa) / n_partidas

  # média de cartões amarelos do time da casa
  media_cartoes_amarelos_casa = (cartoes_amarelos_time_da_casa_em_casa + cartoes_amarelos_time_da_casa_fora_de_casa) / n_partidas
  # média de cartões amarelos do time visitante
  media_cartoes_amarelos_fora = (cartoes_amarelos_time_de_fora_fora_de_casa + cartoes_amarelos_time_de_fora_em_casa) / n_partidas

  # média de cartões vermelhos do time da casa
  media_cartoes_vermelhos_casa = (cartoes_vermelhos_time_da_casa_em_casa + cartoes_vermelhos_time_da_casa_fora_de_casa) / n_partidas
  # média de cartões vermelhos do time visitante
  media_cartoes_vermelhos_fora = (cartoes_vermelhos_time_de_fora_fora_de_casa + cartoes_vermelhos_time_de_fora_em_casa) / n_partidas

  # média de cartões no primeiro tempo do time da casa
  media_cartoes_primeiro_tempo_casa = (cartoes_primeiro_tempo_time_da_casa_em_casa + cartoes_primeiro_tempo_time_da_casa_fora_de_casa) / n_partidas
  # média de cartões no primeiro tempo do time visitante
  media_cartoes_primeiro_tempo_fora = (cartoes_primeiro_tempo_time_de_fora_fora_de_casa + cartoes_primeiro_tempo_time_de_fora_em_casa) / n_partidas

  # média de cartões no segundo tempo do time da casa
  media_cartoes_segundo_tempo_casa = (cartoes_segundo_tempo_time_da_casa_em_casa + cartoes_segundo_tempo_time_da_casa_fora_de_casa) / n_partidas
  # média de cartões no segundo tempo do time visitante
  media_cartoes_segundo_tempo_fora = (cartoes_segundo_tempo_time_de_fora_fora_de_casa + cartoes_segundo_tempo_time_de_fora_em_casa) / n_partidas

  # média de chutes no gol pelo time da casa
  media_chutes_no_gol_time_da_casa = (chutes_no_gol_time_da_casa_em_casa + chutes_no_gol_time_da_casa_fora_de_casa) / n_partidas
  # média de chutes no gol pelo time visitante
  media_chutes_no_gol_time_de_fora = (chutes_no_gol_time_de_fora_em_casa + chutes_no_gol_time_de_fora_fora_de_casa) / n_partidas

  # média de chutes fora do gol pelo time da casa
  media_chutes_fora_do_gol_time_da_casa = (chutes_fora_do_gol_time_da_casa_em_casa + chutes_fora_do_gol_time_da_casa_fora_de_casa) / n_partidas
  # média de chutes fora do gol pelo time visitante
  media_chutes_fora_do_gol_time_de_fora = (chutes_fora_do_gol_time_de_fora_em_casa + chutes_fora_do_gol_time_de_fora_fora_de_casa) / n_partidas

  # média de faltas cometidas pelo time da casa
  media_faltas_do_time_da_casa = (faltas_do_time_da_casa_em_casa + faltas_do_time_da_casa_fora_de_casa) / n_partidas
  # média de faltas cometidas pelo time visitante
  media_faltas_do_time_de_fora = (faltas_do_time_de_fora_em_casa + faltas_do_time_de_fora_fora_de_casa) / n_partidas

  # média de posse de bola do time da casa
  media_posse_de_bola_do_time_da_casa = (posse_de_bola_do_time_da_casa_em_casa + posse_de_bola_do_time_da_casa_fora_de_casa) / n_partidas
  # média de posse de bola do time visitante
  media_posse_de_bola_do_time_de_fora = (posse_de_bola_do_time_de_fora_em_casa + posse_de_bola_do_time_de_fora_fora_de_casa) / n_partidas

  # atribuindo os valores encontrados anteriormente em um dicionário
  dados = {
    'time_casa': time_casa,
    'time_fora': time_fora,
    'arbitro': arbitro,
    'media_gols_casa': round(media_gols_time_casa, 2) ,
    'media_gols_fora': round(media_gols_time_fora, 2),
    'media_gols_primeiro_casa': round(media_gols_primeiro_tempo_casa, 2),
    'media_gols_primeiro_fora': round(media_gols_primeiro_tempo_fora, 2),
    'media_escanteios_casa': round(media_escanteios_casa, 2),
    'media_escanteios_fora': round(media_escanteios_fora, 2),
    'media_cartoes_amarelos_casa': round(media_cartoes_amarelos_casa, 2),
    'media_cartoes_amarelos_fora': round(media_cartoes_amarelos_fora, 2),
    'media_cartoes_vermelhos_casa': round(media_cartoes_vermelhos_casa, 2),
    'media_cartoes_vermelhos_fora': round(media_cartoes_vermelhos_fora, 2),
    'media_cartoes_primeiro_casa': round(media_cartoes_primeiro_tempo_casa, 2),
    'media_cartoes_primeiro_fora': round(media_cartoes_primeiro_tempo_fora, 2),
    'media_cartoes_segundo_casa': round(media_cartoes_segundo_tempo_casa, 2),
    'media_cartoes_segundo_fora': round(media_cartoes_segundo_tempo_fora, 2),
    'media_chutes_no_gol_casa': round(media_chutes_no_gol_time_da_casa, 2),
    'media_chutes_no_gol_fora': round(media_chutes_no_gol_time_de_fora, 2),
    'media_chutes_fora_do_gol_casa': round(media_chutes_fora_do_gol_time_da_casa, 2),
    'media_chutes_fora_do_gol_fora': round(media_chutes_fora_do_gol_time_de_fora, 2),
    'media_faltas_casa': round(media_faltas_do_time_da_casa, 2),
    'media_faltas_fora': round(media_faltas_do_time_de_fora, 2),
    'media_posse_de_bola_casa': round(media_posse_de_bola_do_time_da_casa, 2),
    'media_posse_de_bola_fora': round(media_posse_de_bola_do_time_de_fora, 2)
}

  # adicionando o dicionário ao dataframe de médias
  df_nova_previsao_partidas_gerais = pd.DataFrame(dados, index=[0])
  
  return (df_nova_previsao_partidas_gerais)

# Método que gera o dataframe que vai ser utilizado como nova previsão (histórico do campeonato)

def historico_campeonato(partidas_casa_time_casa_df1, partidas_fora_time_fora_df1, multi_target_rfc, le):

  df1_casa = nova_previsao_partidas_casa_fora(partidas_casa_time_casa_df1, arbitro)
  df1_fora = nova_previsao_partidas_casa_fora(partidas_fora_time_fora_df1, arbitro)

  colunas_casa = ['time_casa', 'time_fora', 'arbitro', 'media_gols_casa', 'media_gols_primeiro_casa', 'media_escanteios_casa', 'media_cartoes_amarelos_casa', 'media_cartoes_vermelhos_casa', 'media_cartoes_primeiro_casa', 'media_cartoes_segundo_casa', 'media_chutes_no_gol_casa', 'media_chutes_fora_do_gol_casa', 'media_faltas_casa', 'media_posse_de_bola_casa']
  colunas_fora = ['media_gols_fora', 'media_gols_primeiro_fora', 'media_escanteios_fora', 'media_cartoes_amarelos_fora', 'media_cartoes_vermelhos_fora', 'media_cartoes_primeiro_fora', 'media_cartoes_segundo_fora', 'media_chutes_no_gol_fora', 'media_chutes_fora_do_gol_fora', 'media_faltas_fora', 'media_posse_de_bola_fora']

  df_combinado = pd.concat([df1_casa[colunas_casa], df1_fora[colunas_fora]], axis=1)

  df_combinado = df_combinado.reindex(columns=['time_casa', 'time_fora','arbitro','media_gols_casa','media_gols_fora','media_gols_primeiro_casa',
      'media_gols_primeiro_fora','media_escanteios_casa', 'media_escanteios_fora','media_cartoes_amarelos_casa', 'media_cartoes_amarelos_fora',
      'media_cartoes_vermelhos_casa', 'media_cartoes_vermelhos_fora','media_cartoes_primeiro_casa', 'media_cartoes_primeiro_fora',
      'media_cartoes_segundo_casa', 'media_cartoes_segundo_fora','media_chutes_no_gol_casa', 'media_chutes_no_gol_fora','media_chutes_fora_do_gol_casa',
      'media_chutes_fora_do_gol_fora','media_faltas_casa', 'media_faltas_fora','media_posse_de_bola_casa','media_posse_de_bola_fora'])

  df1, df2 = df_completo_partidas_casa_fora(partidas_df)

  for index, row in df1.iterrows():
    partidas_casa_time_casa_df1 = df1[(df1['time_casa'] == time_casa)]
    partidas_casa_time_fora_df1 = df1[(df1['time_fora'] == time_fora)]

    partidas_casa_fora = pd.concat([partidas_casa_time_casa_df1,partidas_casa_time_fora_df1])  

  for index, row in df2.iterrows():
    partidas_casa_time_casa_df2 = df2[(df2['time_casa'] == time_casa)]
    partidas_casa_time_fora_df2 = df2[(df2['time_fora'] == time_fora)]

    partidas_casa_fora_df2 = pd.concat([partidas_casa_time_casa_df2,partidas_casa_time_fora_df2])

  print('Histórico como mandante do ' + time_casa + ' no campeonato e o histórico como visitante do ' + time_fora + ' no campeonato \n')
  print(tabulate(partidas_casa_fora_df2, headers='keys', tablefmt='fancy_grid', numalign='center', stralign='center'))
  print('\n Média dos times no campeonato')
  print(tabulate(df_combinado, headers='keys', tablefmt='fancy_grid', numalign='center', stralign='center'))
  print('\n Previsões')
  fazer_previsao(df_combinado, multi_target_rfc, le)

# Método que retorna as novas previsões com base nas últimas partidas jogando em casa e fora

def ultimas_partidas_casa_fora(partidas_casa_time_casa_df1, partidas_casa_time_fora_df1, ultimas_n_partidas, num_partidas, time_casa, time_fora, multi_target_rfc, le):

  df1_casa = nova_previsao_partidas_casa_fora(partidas_casa_time_casa_df1, arbitro)
  df1_fora = nova_previsao_partidas_casa_fora(partidas_casa_time_fora_df1, arbitro)

  colunas_casa = ['time_casa', 'time_fora', 'arbitro', 'media_gols_casa', 'media_gols_primeiro_casa', 'media_escanteios_casa', 'media_cartoes_amarelos_casa', 'media_cartoes_vermelhos_casa', 'media_cartoes_primeiro_casa', 'media_cartoes_segundo_casa', 'media_chutes_no_gol_casa', 'media_chutes_fora_do_gol_casa', 'media_faltas_casa', 'media_posse_de_bola_casa']
  colunas_fora = ['media_gols_fora', 'media_gols_primeiro_fora', 'media_escanteios_fora', 'media_cartoes_amarelos_fora', 'media_cartoes_vermelhos_fora', 'media_cartoes_primeiro_fora', 'media_cartoes_segundo_fora', 'media_chutes_no_gol_fora', 'media_chutes_fora_do_gol_fora', 'media_faltas_fora', 'media_posse_de_bola_fora']

  df_combinado = pd.concat([df1_casa[colunas_casa], df1_fora[colunas_fora]], axis=1)

  df_combinado = df_combinado.reindex(columns=['time_casa', 'time_fora','arbitro','media_gols_casa','media_gols_fora','media_gols_primeiro_casa',
      'media_gols_primeiro_fora','media_escanteios_casa', 'media_escanteios_fora','media_cartoes_amarelos_casa', 'media_cartoes_amarelos_fora',
      'media_cartoes_vermelhos_casa', 'media_cartoes_vermelhos_fora','media_cartoes_primeiro_casa', 'media_cartoes_primeiro_fora',
      'media_cartoes_segundo_casa', 'media_cartoes_segundo_fora','media_chutes_no_gol_casa', 'media_chutes_no_gol_fora','media_chutes_fora_do_gol_casa',
      'media_chutes_fora_do_gol_fora','media_faltas_casa', 'media_faltas_fora','media_posse_de_bola_casa','media_posse_de_bola_fora'])

  df1, df2, df3 = df_completo_partidas_casa_fora(ultimas_n_partidas)

  for index, row in df1.iterrows():
    partidas_casa_time_casa_df1 = df1[(df1['time_casa'] == time_casa)]
    partidas_casa_time_fora_df1 = df1[(df1['time_fora'] == time_fora)]

    partidas_casa_fora = pd.concat([partidas_casa_time_casa_df1,partidas_casa_time_fora_df1])  

  for index, row in df2.iterrows():
    partidas_casa_time_casa_df2 = df2[(df2['time_casa'] == time_casa)]
    partidas_casa_time_fora_df2 = df2[(df2['time_fora'] == time_fora)]

    partidas_casa_fora_df2 = pd.concat([partidas_casa_time_casa_df2,partidas_casa_time_fora_df2])

  print('Últimas '+ str(num_partidas) + ' partidas do ' + time_casa + ' como mandante e as últimas ' + str(num_partidas) + ' partidas do ' + time_fora + ' como visitante \n')
  print(tabulate(partidas_casa_fora_df2, headers='keys', tablefmt='fancy_grid', numalign='center', stralign='center'))
  print('\n Média dos times nas últimas' + str(num_partidas) + ' partidas')
  print(tabulate(df_combinado, headers='keys', tablefmt='fancy_grid', numalign='center', stralign='center'))
  print('\n Previsões')
  fazer_previsao(df_combinado, multi_target_rfc, le)

# Método que retorna as novas previsões com base nas últimas partidas de modo geral

def ultimas_partidas_gerais(partidas_casa_time_casa_df1, partidas_casa_time_fora_df1, ultimas_n_partidas, num_partidas, time_casa, time_fora, multi_target_rfc, le):

  df_combinado = nova_previsao_partidas_gerais(ultimas_n_partidas, arbitro, num_partidas, time_casa, time_fora)

  df_combinado = df_combinado.reindex(columns=['time_casa', 'time_fora','arbitro','media_gols_casa','media_gols_fora','media_gols_primeiro_casa',
      'media_gols_primeiro_fora','media_escanteios_casa', 'media_escanteios_fora','media_cartoes_amarelos_casa', 'media_cartoes_amarelos_fora',
      'media_cartoes_vermelhos_casa', 'media_cartoes_vermelhos_fora','media_cartoes_primeiro_casa', 'media_cartoes_primeiro_fora',
      'media_cartoes_segundo_casa', 'media_cartoes_segundo_fora','media_chutes_no_gol_casa', 'media_chutes_no_gol_fora','media_chutes_fora_do_gol_casa',
      'media_chutes_fora_do_gol_fora','media_faltas_casa', 'media_faltas_fora','media_posse_de_bola_casa','media_posse_de_bola_fora'])

  df1, df2, df3 = df_completo_partidas_gerais(ultimas_n_partidas)

  for index, row in df1.iterrows():
    partidas_casa_time_casa_df1 = df1[(df1['time_casa'] == time_casa) | (df1['time_fora'] == time_casa)]
    partidas_casa_time_fora_df1 = df1[(df1['time_casa'] == time_fora) | (df1['time_fora'] == time_fora)]

    partidas_casa_fora = pd.concat([partidas_casa_time_casa_df1,partidas_casa_time_fora_df1])  

  for index, row in df2.iterrows():
    partidas_casa_time_casa_df2 = df2[(df2['time_casa'] == time_casa) | (df1['time_fora'] == time_casa)]
    partidas_casa_time_fora_df2 = df2[(df2['time_casa'] == time_fora) | (df1['time_fora'] == time_fora)]

    partidas_casa_fora_df2 = pd.concat([partidas_casa_time_casa_df2,partidas_casa_time_fora_df2])

  print('Últimas '+ str(num_partidas) + ' partidas do ' + time_casa + ' de modo geral e as últimas ' + str(num_partidas) + ' partidas do ' + time_fora + ' de modo geral \n')
  print(tabulate(partidas_casa_fora_df2, headers='keys', tablefmt='fancy_grid', numalign='center', stralign='center'))
  print('\n Média dos times nas últimas ' + str(num_partidas) + ' partidas')
  print(tabulate(df_combinado, headers='keys', tablefmt='fancy_grid', numalign='center', stralign='center'))
  print('\n Previsões')
  fazer_previsao(df_combinado, multi_target_rfc, le)

# Método que retorna as novas previsões com base nos confrontos diretos

def confrontos_diretos(df3, df2, multi_target_rfc, le, time_casa, time_fora):

  print('Confrontos diretos entre o ' + time_casa + ' e o ' + time_fora + '\n')
  print(tabulate(df2, headers='keys', tablefmt='fancy_grid', numalign='center', stralign='center'))
  print('\n Média dos times nos confrontos diretos')
  print(tabulate(df3, headers='keys', tablefmt='fancy_grid', numalign='center', stralign='center'))
  print('\n Previsões')
  fazer_previsao(df3, multi_target_rfc, le)

# Método que treina e testa o modelo de machine learning

def modelo_ml(partidas_df, data_da_partida):

  # Passando o arquivo das partidas para o método df_completo
  df1_ml, df2_ml, df3_ml = df_completo_partidas_casa_fora(partidas_df)

  # Convertendo a data de treino para o formato datetime e definindo o valor da data de treino
  data_treino = data_da_partida
  data_treino = datetime.strptime(data_treino, '%Y-%m-%d').date()

  # Convertendo a data de teste para o formato datetime e definindo o valor da data de teste
  data_teste = data_da_partida
  data_teste = datetime.strptime(data_teste, '%Y-%m-%d').date()

  # Criação de um objeto LabelEncoder para transformar dados categóricos em numéricos
  le = LabelEncoder()

  # Transformando as colunas 'time_casa', 'time_fora' e 'arbitro' em dados numéricos
  df1_ml['time_casa'] = le.fit_transform(df1_ml['time_casa'])
  df1_ml['time_fora'] = le.fit_transform(df1_ml['time_fora'])
  df1_ml['arbitro'] = le.fit_transform(df1_ml['arbitro'])

  # Separando os dados em conjunto de treino e teste, com base nas datas de treino e teste definidas acima
  treino = df1_ml[df1_ml['data_partida'] < data_treino]
  teste = df1_ml[df1_ml['data_partida'] >= data_teste]

  # Separando as colunas que serão utilizadas para treinar o modelo, excluindo as colunas que representam o resultado da partida
  X_train = treino.drop(['resultado_partida', 'resultado_intervalo', 'resultado_num_gols_over_under', 'resultado_ambas_equipes_marcaram','resultado_num_cartoes_amarelos','resultado_num_cartoes_vermelhos','resultado_num_cartoes_totais','resultado_ambas_equipes_receberam_cartoes','resultado_cartoes_ambos_tempos','resultado_num_escanteios','resultado_num_cartoes_primeiro', 'resultado_num_cartoes_segundo', 'data_partida'], axis=1)

  # Separando as colunas que representam o resultado da partida para treinar o modelo
  y_train = treino[['resultado_partida', 'resultado_intervalo', 'resultado_num_gols_over_under',  'resultado_ambas_equipes_marcaram','resultado_num_cartoes_amarelos','resultado_num_cartoes_vermelhos','resultado_num_cartoes_totais','resultado_ambas_equipes_receberam_cartoes','resultado_cartoes_ambos_tempos','resultado_num_escanteios','resultado_num_cartoes_primeiro', 'resultado_num_cartoes_segundo']]

  # Separando as colunas que serão utilizadas para testar o modelo, excluindo as colunas que representam o resultado da partida
  X_test = teste.drop(['resultado_partida', 'resultado_intervalo', 'resultado_num_gols_over_under', 'resultado_ambas_equipes_marcaram','resultado_num_cartoes_amarelos','resultado_num_cartoes_vermelhos','resultado_num_cartoes_totais','resultado_ambas_equipes_receberam_cartoes','resultado_cartoes_ambos_tempos','resultado_num_escanteios','resultado_num_cartoes_primeiro', 'resultado_num_cartoes_segundo', 'data_partida'], axis=1)

  # Separando as colunas que representam o resultado da partida para testar o modelo
  y_test = teste[['resultado_partida', 'resultado_intervalo', 'resultado_num_gols_over_under', 'resultado_ambas_equipes_marcaram','resultado_num_cartoes_amarelos','resultado_num_cartoes_vermelhos','resultado_num_cartoes_totais','resultado_ambas_equipes_receberam_cartoes','resultado_cartoes_ambos_tempos','resultado_num_escanteios','resultado_num_cartoes_primeiro', 'resultado_num_cartoes_segundo']]

  # Define o classificador random forest com 100 estimadores, profundidade máxima de 10, divisão mínima de amostras de 2, folha mínima de amostras de 1, número máximo de features a considerar é 'auto' e semente aleatória de 42
  rfc = RandomForestClassifier(n_estimators=100, max_depth=30, min_samples_split=5, min_samples_leaf=1, max_features='auto', random_state=42)
  multi_target_rfc = MultiOutputClassifier(rfc, n_jobs=-1)
  multi_target_rfc.fit(X_train, y_train)

  # Realizando as previsões para o conjunto de teste
  y_pred = multi_target_rfc.predict(X_test)

  return (multi_target_rfc, le, y_test, y_pred, df1_ml)

def modelo_ml(partidas_df, data_da_partida):

  # Passando o arquivo das partidas para o método df_completo
  df1_ml, df2_ml, df3_ml = df_completo_partidas_casa_fora(partidas_df)

  # Convertendo a data de treino para o formato datetime e definindo o valor da data de treino
  data_treino = data_da_partida
  data_treino = datetime.strptime(data_treino, '%Y-%m-%d').date()

  # Convertendo a data de teste para o formato datetime e definindo o valor da data de teste
  data_teste = data_da_partida
  data_teste = datetime.strptime(data_teste, '%Y-%m-%d').date()

  # Criação de um objeto LabelEncoder para transformar dados categóricos em numéricos
  le = LabelEncoder()

  # Transformando as colunas 'time_casa', 'time_fora' e 'arbitro' em dados numéricos
  df1_ml['time_casa'] = le.fit_transform(df1_ml['time_casa'])
  df1_ml['time_fora'] = le.fit_transform(df1_ml['time_fora'])
  df1_ml['arbitro'] = le.fit_transform(df1_ml['arbitro'])

  # Selecionando as partidas anteriores a data da partida
  partidas_antes_da_data = df1_ml[df1_ml['data_partida'] < data_treino]

  # Selecionando a última partida antes da data passada
  ultima_partida_antes_da_data = partidas_antes_da_data.iloc[-1]

  # Obtendo a rodada correspondente à última partida antes da data passada
  rodada_da_ultima_partida = ultima_partida_antes_da_data['rodada']

  # Encontrar o número da próxima rodada
  proxima_rodada = rodada_da_ultima_partida + 1

  # Filtrar o DataFrame original para obter todas as partidas na próxima rodada
  partidas_proxima_rodada = df1_ml[df1_ml['rodada'] == proxima_rodada]

  # Separando as colunas que serão utilizadas para treinar o modelo, excluindo as colunas que representam o resultado da partida
  X_train = partidas_antes_da_data.drop(['resultado_partida', 'resultado_intervalo', 'resultado_num_gols_over_under', 'resultado_ambas_equipes_marcaram','resultado_num_cartoes_amarelos','resultado_num_cartoes_vermelhos','resultado_num_cartoes_totais','resultado_ambas_equipes_receberam_cartoes','resultado_cartoes_ambos_tempos','resultado_num_escanteios','resultado_num_cartoes_primeiro', 'resultado_num_cartoes_segundo', 'data_partida', 'rodada'], axis=1)

  # Separando as colunas que representam o resultado da partida para treinar o modelo
  y_train = partidas_antes_da_data[['resultado_partida', 'resultado_intervalo', 'resultado_num_gols_over_under',  'resultado_ambas_equipes_marcaram','resultado_num_cartoes_amarelos','resultado_num_cartoes_vermelhos','resultado_num_cartoes_totais','resultado_ambas_equipes_receberam_cartoes','resultado_cartoes_ambos_tempos','resultado_num_escanteios','resultado_num_cartoes_primeiro', 'resultado_num_cartoes_segundo']]

  # Separando as colunas que serão utilizadas para testar o modelo, excluindo as colunas que representam o resultado da partida
  X_test = partidas_proxima_rodada.drop(['resultado_partida', 'resultado_intervalo', 'resultado_num_gols_over_under', 'resultado_ambas_equipes_marcaram','resultado_num_cartoes_amarelos','resultado_num_cartoes_vermelhos','resultado_num_cartoes_totais','resultado_ambas_equipes_receberam_cartoes','resultado_cartoes_ambos_tempos','resultado_num_escanteios','resultado_num_cartoes_primeiro', 'resultado_num_cartoes_segundo', 'data_partida', 'rodada'], axis=1)

  # Separando as colunas que representam o resultado da partida para testar o modelo
  y_test = partidas_proxima_rodada[['resultado_partida', 'resultado_intervalo', 'resultado_num_gols_over_under', 'resultado_ambas_equipes_marcaram','resultado_num_cartoes_amarelos','resultado_num_cartoes_vermelhos','resultado_num_cartoes_totais','resultado_ambas_equipes_receberam_cartoes','resultado_cartoes_ambos_tempos','resultado_num_escanteios','resultado_num_cartoes_primeiro', 'resultado_num_cartoes_segundo']]

  # Define o classificador random forest com 100 estimadores, profundidade máxima de 10, divisão mínima de amostras de 2, folha mínima de amostras de 1, número máximo de features a considerar é 'auto' e semente aleatória de 42
  rfc = RandomForestClassifier(n_estimators=100, max_depth=30, min_samples_split=5, min_samples_leaf=1, max_features='auto', random_state=42)
  multi_target_rfc = MultiOutputClassifier(rfc, n_jobs=-1)
  multi_target_rfc.fit(X_train, y_train)

  # Realizando as previsões para o conjunto de teste
  y_pred = multi_target_rfc.predict(X_test)

  return (multi_target_rfc, le, y_test, y_pred, df1_ml)

# Método de avaliação do modelo de machine learning

def avaliacao_modelo(y_test, y_pred):

  # Passo 6: Calcular a acurácia do modelo para as variáveis de saída
  accuracy_partida = accuracy_score(y_test['resultado_partida'], y_pred[:, 0])
  accuracy_intervalo = accuracy_score(y_test['resultado_intervalo'], y_pred[:, 1])
  accuracy_num_gols_over_under = accuracy_score(y_test['resultado_num_gols_over_under'], y_pred[:,2])
  accuracy_ambas_equipes_marcaram = accuracy_score(y_test['resultado_ambas_equipes_marcaram'], y_pred[:,3])
  accuracy_cartoes_amarelos = accuracy_score(y_test['resultado_num_cartoes_amarelos'], y_pred[:, 4])
  accuracy_cartoes_vermelhos = accuracy_score(y_test['resultado_num_cartoes_vermelhos'], y_pred[:, 5])
  accuracy_cartoes_totais = accuracy_score(y_test['resultado_num_cartoes_totais'], y_pred[:,6])
  accuracy_ambas_equipes_cartoes = accuracy_score(y_test['resultado_ambas_equipes_receberam_cartoes'], y_pred[:,7])
  accuracy_ambos_tempos_cartoes = accuracy_score(y_test['resultado_cartoes_ambos_tempos'], y_pred[:,8])
  accuracy_escanteios = accuracy_score(y_test['resultado_num_escanteios'], y_pred[:,9])
  accuracy_cartoes_primeiro = accuracy_score(y_test['resultado_num_cartoes_primeiro'], y_pred[:,10])
  accuracy_cartoes_segundo = accuracy_score(y_test['resultado_num_cartoes_segundo'], y_pred[:,11])

  # dados do DataFrame
  data = {'Acurácia': [f'{accuracy_partida:.2%}', f'{accuracy_intervalo:.2%}', f'{accuracy_num_gols_over_under:.2%}', f'{accuracy_ambas_equipes_marcaram:.2%}', f'{accuracy_cartoes_amarelos:.2%}', f'{accuracy_cartoes_vermelhos:.2%}', f'{accuracy_cartoes_totais:.2%}', f'{accuracy_ambas_equipes_cartoes:.2%}', f'{accuracy_ambos_tempos_cartoes:.2%}', f'{accuracy_escanteios:.2%}', f'{accuracy_cartoes_primeiro:.2%}', f'{accuracy_cartoes_segundo:.2%}']}

  
  data = {'Acurácia': [f'{accuracy_partida:.2%}', f'{accuracy_intervalo:.2%}', f'{accuracy_num_gols_over_under:.2%}', f'{accuracy_ambas_equipes_marcaram:.2%}', f'{accuracy_cartoes_amarelos:.2%}', f'{accuracy_cartoes_vermelhos:.2%}', f'{accuracy_cartoes_totais:.2%}', f'{accuracy_ambas_equipes_cartoes:.2%}', f'{accuracy_ambos_tempos_cartoes:.2%}', f'{accuracy_escanteios:.2%}', f'{accuracy_cartoes_primeiro:.2%}', f'{accuracy_cartoes_segundo:.2%}']}

  df_final = pd.DataFrame(data, columns=['Acurácia'])

  return(df_final)

# Método que faz as novas previsões

def fazer_previsao(df_para_previsao, multi_target_rfc, le):

  # Passo 3: Codificar as variáveis categóricas
  df_para_previsao['time_casa'] = le.fit_transform(df_para_previsao['time_casa'])
  df_para_previsao['time_fora'] = le.fit_transform(df_para_previsao['time_fora'])
  df_para_previsao['arbitro'] = le.fit_transform(df_para_previsao['arbitro'])

  nova_previsao = multi_target_rfc.predict(df_para_previsao)

  # criar uma lista vazia para armazenar os dados da tabela
  nova_tabela = []

  # adicionar os cabeçalhos da primeira linha
  nova_tabela.append(['Variáveis-alvo', 'Previsões'])

  # cabeçalho da tabela
  headers = ['Resultado da partida', 'Resultado do intervalo', 'Número de gols over under', 'Ambas equipes marcaram',
             'Número de cartões amarelos', 'Número de cartões vermelhos', 'Número de cartões totais',
             'Ambas equipes receberam cartões', 'Cartões aconteceram em ambos os tempos', 'Número de escanteios', 'Número de cartões no primeiro tempo',
             'Número de cartões no segundo tempo']

  # para cada cabeçalho em headers, adicione um novo item à lista
  for i, header in enumerate(headers):
      nova_tabela.append([header, nova_previsao[0, i]])

  # imprimir tabela com dados formatados
  print(tabulate(nova_tabela, tablefmt='fancy_grid', numalign="center", stralign="center"))

# Método para estilizar os dataframes resultantes

def estilizar_df(df_concatenado_time_casa, df_concatenado_time_fora, df_resultados_confrontos_diretos, df_info_confrontos_diretos, time_casa, time_fora, tabela, legenda):
  
  # Define uma classe que herda de pd.DataFrame para criar um DataFrame customizado
  class SubclassedDataFrame(pd.DataFrame):
      
      # Propriedade normal
      _metadata = ['description']
      
      # Sobrescreve o construtor para sempre retornar uma instância da classe customizada
      @property
      def _constructor(self):
          return SubclassedDataFrame

  # Define o DataFrame para os últimos resultados do time da casa, e personaliza a formatação
  data_casa = df_concatenado_time_casa
  df_casa = SubclassedDataFrame(data_casa)
  title_casa = 'Últimos resultados do {}'.format(time_casa)
  
  # Define um estilo para a tabela usando os seletores e propriedades do CSS
  df_casa = (df_casa.style
        .hide_index() # Esconde a coluna do índice
        .set_caption(title_casa) # Define o título da tabela
        .set_table_styles([{
            'selector': 'caption', # Seletor CSS para o título da tabela
            'props': [
                ('color', '#FFFFFF'),
                ('font-size', '18px'),
                ('font-style', 'normal'),
                ('font-weight', 'bold'),
                ('text-align', 'center'),
                ('background-color', '#126e51'),
                ('border', '1px solid gray')
            ]
        },
        {
            'selector': 'th', # Seletor CSS para as células do cabeçalho
            'props': [
                ('background-color', '#126e51'),
                ('color', 'black'),
                ('font-size', '15px'),
                ('font-weight', 'bold'),
                ('text-align', 'center'),
                ('border', '1px solid gray'),
                ('white-space', 'pre-wrap')
            ]
        },
        {
            'selector': 'td', # Seletor CSS para as células de dados
            'props': [
                ('background-color', '#283734'),
                ('color', 'white'),
                ('font-size', '15px'),
                ('font-weight', 'normal'),
                ('text-align', 'center'),
                ('border', '1px solid gray'),
                ('white-space', 'pre-wrap')
            ]
        },
        {
            'selector': 'tr:nth-child(3) td', # Seletor CSS para a terceira linha de células de dados
            'props': [
                ('white-space', 'pre-wrap'),
            ]
        },
        {
            'selector': 'tr:nth-child(2) th', # Seletor CSS para a segunda linha de células do cabeçalho
            'props': [
                ('background-color', '#126e51'),
                ('color', 'black'),
                ('font-size', '15px'),
                ('font-weight', 'bold'),
                ('text-align', 'center'),
                ('border', '1px solid gray')
            ]
        },
        {
            'selector': 'tr:nth-child(2) td', # Seletor CSS para a segunda linha de células de dados
            'props': [
                ('white-space', 'pre-wrap'),
                ('background-color', '#126e51'),
                ('color', 'black'),
                ('font-size', '15px'),
                ('font-weight', 'bold'),
                ('text-align', 'center'),
                ('border', '1px solid gray')
            ]
        }
        ])
      )
  
  data_fora = df_concatenado_time_fora

  df_fora = SubclassedDataFrame(data_fora)
  title_fora = 'Últimos resultados do {}'.format(time_fora)
  header = 'header'

  # Define um estilo para a tabela usando os seletores e propriedades do CSS
  df_fora = (df_fora.style
        .hide_index() # Esconde a coluna do índice
        .set_caption(title_fora) # Define o título da tabela
        .set_table_styles([{
            'selector': 'caption', # Seletor CSS para o título da tabela
            'props': [
                ('color', '#FFFFFF'),
                ('font-size', '18px'),
                ('font-style', 'normal'),
                ('font-weight', 'bold'),
                ('text-align', 'center'),
                ('background-color', '#126e51'),
                ('border', '1px solid gray')
            ]
        },
        {
            'selector': 'th', # Seletor CSS para as células do cabeçalho
            'props': [
                ('background-color', '#126e51'),
                ('color', 'black'),
                ('font-size', '15px'),
                ('font-weight', 'bold'),
                ('text-align', 'center'),
                ('border', '1px solid gray'),
                ('white-space', 'pre-wrap')
            ]
        },
        {
            'selector': 'td', # Seletor CSS para as células de dados
            'props': [
                ('background-color', '#283734'),
                ('color', 'white'),
                ('font-size', '15px'),
                ('font-weight', 'normal'),
                ('text-align', 'center'),
                ('border', '1px solid gray'),
                ('white-space', 'pre-wrap')
            ]
        },
        {
            'selector': 'tr:nth-child(3) td', # Seletor CSS para a terceira linha de células de dados
            'props': [
                ('white-space', 'pre-wrap'),
            ]
        },
        {
            'selector': 'tr:nth-child(2) th', # Seletor CSS para a segunda linha de células do cabeçalho
            'props': [
                ('background-color', '#126e51'),
                ('color', 'black'),
                ('font-size', '15px'),
                ('font-weight', 'bold'),
                ('text-align', 'center'),
                ('border', '1px solid gray')
            ]
        },
        {
            'selector': 'tr:nth-child(2) td', # Seletor CSS para a segunda linha de células de dados
            'props': [
                ('white-space', 'pre-wrap'),
                ('background-color', '#126e51'),
                ('color', 'black'),
                ('font-size', '15px'),
                ('font-weight', 'bold'),
                ('text-align', 'center'),
                ('border', '1px solid gray')
            ]
        }
        ])
      )
  
  # Define a variável 'data_inf' como a DataFrame 'df_info_confrontos_diretos'
  data_inf = df_info_confrontos_diretos

  # Cria uma nova instância de DataFrame personalizada, 'df_inf', passando 'data_inf' como parâmetro
  df_inf = SubclassedDataFrame(data_inf)

  # Define o título da tabela de informações com base nos nomes dos times
  title_inf = 'Informações dos confrontos diretos entre {} e {}'.format(time_casa, time_fora)

  # Define o cabeçalho da tabela como 'header'
  header = 'header'

  # Define o estilo da tabela de informações, aplicando as configurações de estilo para a caption, th e td
  df_inf = (df_inf.style
            .hide_index()
            .set_caption(title_inf)
            .set_table_styles([{
                'selector': 'caption',
                'props': [
                    ('color', '#FFFFFF'),
                    ('font-size', '18px'),
                    ('font-style', 'normal'),
                    ('font-weight', 'bold'),
                    ('text-align', 'center'),
                    ('background-color', '#126e51'),
                    ('border', '1px solid gray')
                ]
            },
            {
                'selector': 'th',
                'props': [
                    ('background-color', '#126e51'),
                    ('color', 'black'),
                    ('font-size', '15px'),
                    ('font-weight', 'bold'),
                    ('text-align', 'center'),
                    ('border', '1px solid gray'),
                    ('white-space', 'pre-wrap')
                ]
            },
            {
                'selector': 'td',
                'props': [
                    ('background-color', '#283734'),
                    ('color', 'white'),
                    ('font-size', '15px'),
                    ('font-weight', 'normal'),
                    ('text-align', 'center'),
                    ('border', '1px solid gray'),
                    ('white-space', 'pre-wrap')
                ]
            },
            {
                'selector': 'tr:nth-child(3) td',
                'props': [
                    ('white-space', 'pre-wrap'),
                ]
            },
            {
                'selector': 'tr:nth-child(2) th',
                'props': [
                    ('background-color', '#126e51'),
                    ('color', 'black'),
                    ('font-size', '15px'),
                    ('font-weight', 'bold'),
                    ('text-align', 'center'),
                    ('border', '1px solid gray')
                ]
            },
            
            ])
          )

  # Define a variável 'data_res' como a DataFrame 'df_resultados_confrontos_diretos'
  data_res = df_resultados_confrontos_diretos

  # Cria uma nova instância de DataFrame personalizada, 'df_res', passando 'data_res' como parâmetro
  df_res = SubclassedDataFrame(data_res)

  # Define o título da tabela de resultados com base nos nomes dos times
  title_res = 'Confrontos diretos entre {} e {}'.format(time_casa, time_fora)

  # Define o cabeçalho da tabela como 'header'
  header = 'header'

  # Define o estilo da tabela de resultados, aplicando as configurações de estilo para a caption, th e td
  df_res = (df_res.style
    .hide_index()
    .set_caption(title_res)
    .set_table_styles([{
        'selector': 'caption',
        'props': [
            ('color', '#FFFFFF'),
            ('font-size', '18px'),
            ('font-style', 'normal'),
            ('font-weight', 'bold'),
            ('text-align', 'center'),
            ('background-color', '#126e51'),
            ('border', '1px solid gray')
        ]
    },
    {
        'selector': 'th',
        'props': [
            ('background-color', '#126e51'),
            ('color', 'black'),
            ('font-size', '15px'),
            ('font-weight', 'bold'),
            ('text-align', 'center'),
            ('border', '1px solid gray'),
            ('white-space', 'pre-wrap')
        ]
    },
    {
        'selector': 'td',
        'props': [
            ('background-color', '#283734'),
            ('color', 'white'),
            ('font-size', '15px'),
            ('font-weight', 'normal'),
            ('text-align', 'center'),
            ('border', '1px solid gray'),
            ('white-space', 'pre-wrap')
        ]
    },
    
    ])
    )
  
  # Define o DataFrame para as tabelas de previsões, e personaliza a formatação
  data_tabela = tabela
  df_tabela = SubclassedDataFrame(data_tabela)
  title_tabela = 'Previsões para {} e {}'.format(time_casa, time_fora)

  # Define um estilo para a tabela usando os seletores e propriedades do CSS
  df_tabela = (df_tabela.style
        .hide_index() # Esconde a coluna do índice
        .set_caption(title_tabela) # Define o título da tabela
        .set_table_styles([{
            'selector': 'caption', # Seletor CSS para o título da tabela
            'props': [
                ('color', '#FFFFFF'),
                ('font-size', '18px'),
                ('font-style', 'normal'),
                ('font-weight', 'bold'),
                ('text-align', 'center'),
                ('background-color', '#126e51'),
                ('border', '1px solid gray')
            ]
        },
        {
            'selector': 'th', # Seletor CSS para as células do cabeçalho
            'props': [
                ('background-color', '#126e51'),
                ('color', 'black'),
                ('font-size', '15px'),
                ('font-weight', 'bold'),
                ('text-align', 'center'),
                ('border', '1px solid gray'),
                ('white-space', 'pre-wrap')
            ]
        },
        {
            'selector': 'td', # Seletor CSS para as células de dados
            'props': [
                ('background-color', '#283734'),
                ('color', 'white'),
                ('font-size', '15px'),
                ('font-weight', 'normal'),
                ('text-align', 'center'),
                ('border', '1px solid gray'),
                ('white-space', 'pre-wrap')
            ]
        },
        ])
    )
  
  # Define o DataFrame para as legendas, e personaliza a formatação
  data_legenda = legenda
  df_legenda = SubclassedDataFrame(data_legenda)
  title_legenda = 'Legenda dos Padrões'

  # Define um estilo para a tabela usando os seletores e propriedades do CSS
  df_legenda = (df_legenda.style
        .hide_index() # Esconde a coluna do índice
        .set_caption(title_legenda) # Define o título da tabela
        .set_table_styles([{
            'selector': 'caption', # Seletor CSS para o título da tabela
            'props': [
                ('color', '#FFFFFF'),
                ('font-size', '18px'),
                ('font-style', 'normal'),
                ('font-weight', 'bold'),
                ('text-align', 'center'),
                ('background-color', '#126e51'),
                ('border', '1px solid gray')
            ]
        },
        {
            'selector': 'th', # Seletor CSS para as células do cabeçalho
            'props': [
                ('background-color', '#126e51'),
                ('color', 'black'),
                ('font-size', '15px'),
                ('font-weight', 'bold'),
                ('text-align', 'center'),
                ('border', '1px solid gray'),
                ('white-space', 'pre-wrap')
            ]
        },
        {
            'selector': 'td', # Seletor CSS para as células de dados
            'props': [
                ('background-color', '#283734'),
                ('color', 'white'),
                ('font-size', '15px'),
                ('font-weight', 'normal'),
                ('text-align', 'center'),
                ('border', '1px solid gray'),
                ('white-space', 'pre-wrap')
            ]
        },
        ])
    )
  

  display(df_tabela)
  print('\n')
  display(df_legenda)
  print('\n')
  display(df_casa)
  print('\n')
  display(df_fora)
  print('\n')
  display(df_res)
  print('\n')
  display(df_inf)

# Método que gera uma tabela de resultados e médias para o time da casa e o time de fora

def tabela_resultados_medias(partidas_anteriores, time_casa, time_fora, multi_target_rfc, le, num_partidas, arbitro):  
  # agrupando as partidas por time da casa e time de fora
  grupos_casa = partidas_anteriores.groupby('home_team_name')
  grupos_fora = partidas_anteriores.groupby('away_team_name')

  # salvando os nomes dos times da casa em uma lista para futura verificação
  times_da_casa = partidas_anteriores['home_team_name'].unique()
  # salvando os nomes dos times de fora em uma lista para futura verificação
  times_de_fora = partidas_anteriores['away_team_name'].unique()
  # salvando os nomes dos árbitros em uma lista para futura verificação
  arbitros = partidas_anteriores['referee'].unique()

  # PARTIDAS GERAIS #

  # Criar um dataframe vazio para armazenar as últimas n partidas de cada time
  df_ultimas_partidas_gerais = pd.DataFrame()

  # Iterar sobre cada time
  for time in partidas_anteriores['home_team_name'].unique():
      # Filtrar as partidas do time em questão
      partidas_time = partidas_anteriores[(partidas_anteriores['home_team_name'] == time) | (partidas_anteriores['away_team_name'] == time)]  
      # Ordenar as partidas do time pela data em ordem decrescente e selecionar as últimas n partidas
      ultimas_partidas_gerais = partidas_time.sort_values(by='data', ascending=False).head(num_partidas) 
      # Adicionar as últimas n partidas do time ao dataframe final
      df_ultimas_partidas_gerais = pd.concat([df_ultimas_partidas_gerais, ultimas_partidas_gerais])

  # Ajustar o índice do dataframe final
  df_ultimas_partidas_gerais.reset_index(drop=True, inplace=True)

  for index, row in df_ultimas_partidas_gerais.iterrows():
    partidas_gerais_time_casa = df_ultimas_partidas_gerais[(df_ultimas_partidas_gerais['home_team_name'] == time_casa) | (df_ultimas_partidas_gerais['away_team_name'] == time_casa)]
    partidas_gerais_time_fora = df_ultimas_partidas_gerais[(df_ultimas_partidas_gerais['home_team_name'] == time_fora) | (df_ultimas_partidas_gerais['away_team_name'] == time_fora)]

    # Remover linhas duplicadas em partidas_gerais_time_casa
    partidas_gerais_time_casa = partidas_gerais_time_casa.drop_duplicates()

    # Remover linhas duplicadas em partidas_gerais_time_fora
    partidas_gerais_time_fora = partidas_gerais_time_fora.drop_duplicates()

    ultimas_n_partidas_gerais = pd.concat([partidas_gerais_time_casa,partidas_gerais_time_fora])

  df_combinado_partidas_gerais = nova_previsao_partidas_gerais(ultimas_n_partidas_gerais, arbitro, num_partidas, time_casa, time_fora)

  df_combinado_partidas_gerais = df_combinado_partidas_gerais.reindex(columns=['time_casa', 'time_fora','arbitro','media_gols_casa','media_gols_fora','media_gols_primeiro_casa',
      'media_gols_primeiro_fora','media_escanteios_casa', 'media_escanteios_fora','media_cartoes_amarelos_casa', 'media_cartoes_amarelos_fora',
      'media_cartoes_vermelhos_casa', 'media_cartoes_vermelhos_fora','media_cartoes_primeiro_casa', 'media_cartoes_primeiro_fora',
      'media_cartoes_segundo_casa', 'media_cartoes_segundo_fora','media_chutes_no_gol_casa', 'media_chutes_no_gol_fora','media_chutes_fora_do_gol_casa',
      'media_chutes_fora_do_gol_fora','media_faltas_casa', 'media_faltas_fora','media_posse_de_bola_casa','media_posse_de_bola_fora'])

  # MÉDIAS NAS PARTIDAS GERAIS DO TIME DA CASA #

  # Criar dataframe df_combinado_casa com base nas colunas relevantes
  colunas_partidas_gerais_time_casa = ['media_gols_casa', 'media_gols_primeiro_casa',
                  'media_escanteios_casa', 'media_cartoes_amarelos_casa',
                  'media_cartoes_vermelhos_casa', 'media_cartoes_primeiro_casa',
                  'media_cartoes_segundo_casa']
  df_combinado_partidas_gerais_time_casa = df_combinado_partidas_gerais[colunas_partidas_gerais_time_casa]

  # Renomear as colunas
  novo_nome_colunas_partidas_gerais = ['Gols', 'Gols no primeiro tempo',
                      'Escanteios', 'Cartões amarelos',
                      'Cartões vermelhos', 'Cartões no primeiro tempo',
                      'Cartões no segundo tempo']

  df_combinado_partidas_gerais_time_casa.columns = novo_nome_colunas_partidas_gerais

  # MÉDIAS NAS PARTIDAS GERAIS DO TIME DE FORA #

  # Criar dataframe df_combinado_fora com base nas colunas relevantes
  colunas_partidas_time_fora = ['media_gols_fora', 'media_gols_primeiro_fora',
                  'media_escanteios_fora', 'media_cartoes_amarelos_fora',
                  'media_cartoes_vermelhos_fora', 'media_cartoes_primeiro_fora',
                  'media_cartoes_segundo_fora']
  df_combinado_partidas_gerais_time_fora = df_combinado_partidas_gerais[colunas_partidas_time_fora]

  df_combinado_partidas_gerais_time_fora.columns = novo_nome_colunas_partidas_gerais

  # PARTIDAS GERAIS PROPRIAMENTE DITAS #

  _, _, df3_partidas = df_completo_partidas_gerais(ultimas_n_partidas_gerais)

  for index, row in df3_partidas.iterrows():
    partidas_gerais_time_casa_df2 = df3_partidas[(df3_partidas['time_casa'] == time_casa) | (df3_partidas['time_fora'] == time_casa)]
    partidas_gerais_time_fora_df2 = df3_partidas[(df3_partidas['time_casa'] == time_fora) | (df3_partidas['time_fora'] == time_fora)]

  partidas_gerais_time_casa_df2 = partidas_gerais_time_casa_df2.drop(["time_casa", "time_fora"], axis=1)
  partidas_gerais_time_fora_df2 = partidas_gerais_time_fora_df2.drop(["time_casa", "time_fora"], axis=1)

  # CONCATENANDO O DATAFRAMES DE PARTIDAS COM O DE MÉDIAS

  # Converte o DataFrame para uma única linha
  df_combinado_partidas_gerais_time_casa = df_combinado_partidas_gerais_time_casa.T.reset_index()
  df_combinado_partidas_gerais_time_fora = df_combinado_partidas_gerais_time_fora.T.reset_index()

  # Renomeia as colunas
  df_combinado_partidas_gerais_time_casa.columns = ['coluna', 'valor']
  df_combinado_partidas_gerais_time_fora.columns = ['coluna', 'valor']

  # Concatena o nome da coluna com o valor em uma única coluna
  df_combinado_partidas_gerais_time_casa['coluna_valor'] = df_combinado_partidas_gerais_time_casa['coluna'] + ': ' + df_combinado_partidas_gerais_time_casa['valor'].astype(str)
  df_combinado_partidas_gerais_time_fora['coluna_valor'] = df_combinado_partidas_gerais_time_fora['coluna'] + ': ' + df_combinado_partidas_gerais_time_fora['valor'].astype(str)

  # Exclui as colunas desnecessárias
  df_combinado_partidas_gerais_time_casa = df_combinado_partidas_gerais_time_casa[['coluna_valor']]
  df_combinado_partidas_gerais_time_fora = df_combinado_partidas_gerais_time_fora[['coluna_valor']]

  # Médias das últimas partidas gerais
  df_medias_partidas_gerais_time_casa = pd.DataFrame({'Médias': ['\n'.join(df_combinado_partidas_gerais_time_casa['coluna_valor'])]})
  df_medias_partidas_gerais_time_fora = pd.DataFrame({'Médias': ['\n'.join(df_combinado_partidas_gerais_time_fora['coluna_valor'])]})

  # Resultados das últimas partidas gerais
  df_resultados_partidas_gerais_casa = pd.DataFrame({'Geral': ['\n'.join(partidas_gerais_time_casa_df2['Geral'])]})
  df_resultados_partidas_gerais_fora = pd.DataFrame({'Geral': ['\n'.join(partidas_gerais_time_fora_df2['Geral'])]})

  # Crie um dataframe intermediário com uma única linha e uma única coluna
  df_intermediario = pd.DataFrame({'Resultados': ['Médias']})

  # Transponha os dataframes do time da casa
  df_resultados_partidas_gerais_casa_t = df_resultados_partidas_gerais_casa.transpose()
  df_medias_partidas_gerais_time_casa_t = df_medias_partidas_gerais_time_casa.transpose()
  df_intermediario_t = df_intermediario.transpose()

  # Reset o índice dos dataframes transpostos do time da casa
  df_resultados_partidas_gerais_casa_t = df_resultados_partidas_gerais_casa_t.reset_index(drop=True)
  df_medias_partidas_gerais_time_casa_t = df_medias_partidas_gerais_time_casa_t.reset_index(drop=True)
  df_intermediario_t = df_intermediario_t.reset_index(drop=True)

  # Transponha os dataframes do time da casa
  df_resultados_partidas_gerais_fora_t = df_resultados_partidas_gerais_fora.transpose()
  df_medias_partidas_gerais_time_fora_t = df_medias_partidas_gerais_time_fora.transpose()
  df_intermediario_t = df_intermediario.transpose()

  # Reset o índice dos dataframes transpostos do time da casa
  df_resultados_partidas_gerais_fora_t = df_resultados_partidas_gerais_fora_t.reset_index(drop=True)
  df_medias_partidas_gerais_time_fora_t = df_medias_partidas_gerais_time_fora_t.reset_index(drop=True)
  df_intermediario_t = df_intermediario_t.reset_index(drop=True)

  # Concatene os dataframes transpostos verticalmente
  df_concatenado_resultados_medias_gerais_time_casa = pd.concat([df_resultados_partidas_gerais_casa_t, df_intermediario_t ,df_medias_partidas_gerais_time_casa_t], axis=0)
  df_concatenado_resultados_medias_gerais_time_fora = pd.concat([df_resultados_partidas_gerais_fora_t, df_intermediario_t ,df_medias_partidas_gerais_time_fora_t], axis=0)

  # Reset o índice do dataframe resultante
  df_concatenado_resultados_medias_gerais_time_casa = df_concatenado_resultados_medias_gerais_time_casa.reset_index(drop=True)
  df_concatenado_resultados_medias_gerais_time_fora = df_concatenado_resultados_medias_gerais_time_fora.reset_index(drop=True)

  # Renomeie a coluna do dataframe resultante
  # DATAFRAMES COM OS RESULTADOS E MÉDIAS DAS ÚLTIMAS 5 PARTIDAS EM GERAL DOS TIMES
  df_concatenado_resultados_medias_gerais_time_casa.columns = ['Geral']
  df_concatenado_resultados_medias_gerais_time_fora.columns = ['Geral']

  # PARTIDAS CASA E FORA

  dataframe_partidas_casa_fora_time_casa_mandante = pd.DataFrame()
  dataframe_partidas_casa_fora_time_casa_visitante = pd.DataFrame()
  dataframe_partidas_casa_fora_time_fora_mandante = pd.DataFrame()
  dataframe_partidas_casa_fora_time_fora_visitante = pd.DataFrame()

  # para cada time, seleciona as últimas n partidas e adiciona no dataframe final
  for time, grupo in grupos_casa:
    ultimas_partidas_time = grupo.sort_values('data').tail(num_partidas)
    dataframe_partidas_casa_fora_time_casa_mandante = pd.concat([dataframe_partidas_casa_fora_time_casa_mandante, ultimas_partidas_time])
    dataframe_partidas_casa_fora_time_fora_mandante = pd.concat([dataframe_partidas_casa_fora_time_fora_mandante, ultimas_partidas_time])

  for time, grupo in grupos_fora:
    ultimas_partidas_time = grupo.sort_values('data').tail(num_partidas)
    dataframe_partidas_casa_fora_time_fora_visitante = pd.concat([dataframe_partidas_casa_fora_time_fora_visitante, ultimas_partidas_time])
    dataframe_partidas_casa_fora_time_casa_visitante = pd.concat([dataframe_partidas_casa_fora_time_casa_visitante, ultimas_partidas_time])

  for index, row in partidas_anteriores.iterrows():
    partidas_casa_fora_time_casa_mandante = dataframe_partidas_casa_fora_time_casa_mandante[(dataframe_partidas_casa_fora_time_casa_mandante['home_team_name'] == time_casa)]
    partidas_casa_fora_time_casa_visitante = dataframe_partidas_casa_fora_time_casa_visitante[(dataframe_partidas_casa_fora_time_casa_visitante['away_team_name'] == time_casa)]
    partidas_casa_fora_time_fora_mandante = dataframe_partidas_casa_fora_time_fora_mandante[(dataframe_partidas_casa_fora_time_fora_mandante['home_team_name'] == time_fora)]
    partidas_casa_fora_time_fora_visitante = dataframe_partidas_casa_fora_time_fora_visitante[(dataframe_partidas_casa_fora_time_fora_visitante['away_team_name'] == time_fora)]

    ultimas_n_partidas_time_casa = pd.concat([partidas_casa_fora_time_casa_mandante,partidas_casa_fora_time_casa_visitante])
    ultimas_n_partidas_time_fora = pd.concat([partidas_casa_fora_time_fora_mandante,partidas_casa_fora_time_fora_visitante])

  df1_casa_mandante = nova_previsao_partidas_casa_fora(partidas_casa_fora_time_casa_mandante, arbitro, time_casa, time_fora)
  df1_casa_visitante = nova_previsao_partidas_casa_fora(partidas_casa_fora_time_casa_visitante, arbitro, time_casa, time_fora)
  df1_fora_mandante = nova_previsao_partidas_casa_fora(partidas_casa_fora_time_fora_mandante, arbitro, time_casa, time_fora)
  df1_fora_visitante = nova_previsao_partidas_casa_fora(partidas_casa_fora_time_fora_visitante, arbitro, time_casa, time_fora)

  colunas_casa = ['time_casa', 'time_fora', 'arbitro', 'media_gols_casa', 'media_gols_primeiro_casa', 'media_escanteios_casa', 'media_cartoes_amarelos_casa', 'media_cartoes_vermelhos_casa', 'media_cartoes_primeiro_casa', 'media_cartoes_segundo_casa', 'media_chutes_no_gol_casa', 'media_chutes_fora_do_gol_casa', 'media_faltas_casa', 'media_posse_de_bola_casa']
  colunas_fora = ['media_gols_fora', 'media_gols_primeiro_fora', 'media_escanteios_fora', 'media_cartoes_amarelos_fora', 'media_cartoes_vermelhos_fora', 'media_cartoes_primeiro_fora', 'media_cartoes_segundo_fora', 'media_chutes_no_gol_fora', 'media_chutes_fora_do_gol_fora', 'media_faltas_fora', 'media_posse_de_bola_fora']

  df_combinado_time_casa = pd.concat([df1_casa_mandante[colunas_casa], df1_casa_visitante[colunas_fora]], axis=1)
  df_combinado_time_fora = pd.concat([df1_fora_mandante[colunas_casa], df1_fora_visitante[colunas_fora]], axis=1)

  df_combinado_time_casa = df_combinado_time_casa.reindex(columns=['time_casa', 'time_fora','arbitro','media_gols_casa','media_gols_fora','media_gols_primeiro_casa',
      'media_gols_primeiro_fora','media_escanteios_casa', 'media_escanteios_fora','media_cartoes_amarelos_casa', 'media_cartoes_amarelos_fora',
      'media_cartoes_vermelhos_casa', 'media_cartoes_vermelhos_fora','media_cartoes_primeiro_casa', 'media_cartoes_primeiro_fora',
      'media_cartoes_segundo_casa', 'media_cartoes_segundo_fora','media_chutes_no_gol_casa', 'media_chutes_no_gol_fora','media_chutes_fora_do_gol_casa',
      'media_chutes_fora_do_gol_fora','media_faltas_casa', 'media_faltas_fora','media_posse_de_bola_casa','media_posse_de_bola_fora'])

  df_combinado_time_fora = df_combinado_time_fora.reindex(columns=['time_casa', 'time_fora','arbitro','media_gols_casa','media_gols_fora','media_gols_primeiro_casa',
      'media_gols_primeiro_fora','media_escanteios_casa', 'media_escanteios_fora','media_cartoes_amarelos_casa', 'media_cartoes_amarelos_fora',
      'media_cartoes_vermelhos_casa', 'media_cartoes_vermelhos_fora','media_cartoes_primeiro_casa', 'media_cartoes_primeiro_fora',
      'media_cartoes_segundo_casa', 'media_cartoes_segundo_fora','media_chutes_no_gol_casa', 'media_chutes_no_gol_fora','media_chutes_fora_do_gol_casa',
      'media_chutes_fora_do_gol_fora','media_faltas_casa', 'media_faltas_fora','media_posse_de_bola_casa','media_posse_de_bola_fora'])

  # MÉDIAS NAS PARTIDAS EM CASA DO TIME DA CASA #

  # Renomear as colunas
  novo_nome_colunas = ['Gols', 'Gols no primeiro tempo',
                      'Escanteios', 'Cartões amarelos',
                      'Cartões vermelhos', 'Cartões no primeiro tempo',
                      'Cartões no segundo tempo']

  # Criar dataframe df_combinado_casa com base nas colunas relevantes
  colunas_casa_mandante = ['media_gols_casa', 'media_gols_primeiro_casa',
                  'media_escanteios_casa', 'media_cartoes_amarelos_casa',
                  'media_cartoes_vermelhos_casa', 'media_cartoes_primeiro_casa',
                  'media_cartoes_segundo_casa']

  df_combinado_casa_mandante = df_combinado_time_casa[colunas_casa_mandante]

  df_combinado_casa_mandante.columns = novo_nome_colunas

  # MÉDIAS NAS PARTIDAS FORA DE CASA DO TIME DA CASA #

  # Criar dataframe df_combinado_fora com base nas colunas relevantes
  colunas_casa_visitante = ['media_gols_fora', 'media_gols_primeiro_fora',
                  'media_escanteios_fora', 'media_cartoes_amarelos_fora',
                  'media_cartoes_vermelhos_fora', 'media_cartoes_primeiro_fora',
                  'media_cartoes_segundo_fora']

  df_combinado_casa_visitante = df_combinado_time_casa[colunas_casa_visitante]

  df_combinado_casa_visitante.columns = novo_nome_colunas

  # MÉDIAS NAS PARTIDAS EM CASA DO TIME DE FORA #

  # Criar dataframe df_combinado_casa com base nas colunas relevantes
  colunas_fora_mandante = ['media_gols_casa', 'media_gols_primeiro_casa',
                  'media_escanteios_casa', 'media_cartoes_amarelos_casa',
                  'media_cartoes_vermelhos_casa', 'media_cartoes_primeiro_casa',
                  'media_cartoes_segundo_casa']

  df_combinado_fora_mandante = df_combinado_time_fora[colunas_fora_mandante]

  df_combinado_fora_mandante.columns = novo_nome_colunas

  # MÉDIAS NAS PARTIDAS FORA DE CASA DO TIME DE FORA #

  # Criar dataframe df_combinado_fora com base nas colunas relevantes
  colunas_fora_visitante = ['media_gols_fora', 'media_gols_primeiro_fora',
                  'media_escanteios_fora', 'media_cartoes_amarelos_fora',
                  'media_cartoes_vermelhos_fora', 'media_cartoes_primeiro_fora',
                  'media_cartoes_segundo_fora']

  df_combinado_fora_visitante = df_combinado_time_fora[colunas_fora_visitante]

  df_combinado_fora_visitante.columns = novo_nome_colunas

  # PARTIDAS EM CASA E FORA PROPRIAMENTE DITAS

  _, _, df3_partidas_time_casa = df_completo_partidas_casa_fora(ultimas_n_partidas_time_casa)
  _, _, df3_partidas_time_fora = df_completo_partidas_casa_fora(ultimas_n_partidas_time_fora)
      
  for index, row in df3_partidas_time_casa.iterrows():
    partidas_time_casa_mandante = df3_partidas_time_casa[(df3_partidas_time_casa['time_casa'] == time_casa)]
    partidas_time_casa_visitante = df3_partidas_time_casa[(df3_partidas_time_casa['time_fora'] == time_casa)]

  for index, row in df3_partidas_time_fora.iterrows():
    partidas_time_fora_mandante = df3_partidas_time_fora[(df3_partidas_time_fora['time_casa'] == time_fora)]
    partidas_time_fora_visitante = df3_partidas_time_fora[(df3_partidas_time_fora['time_fora'] == time_fora)]

  partidas_time_casa_mandante = partidas_time_casa_mandante.drop(["time_casa", "time_fora"], axis=1)
  partidas_time_casa_visitante = partidas_time_casa_visitante.drop(["time_casa", "time_fora"], axis=1)
  partidas_time_fora_mandante = partidas_time_fora_mandante.drop(["time_casa", "time_fora"], axis=1)
  partidas_time_fora_visitante = partidas_time_fora_visitante.drop(["time_casa", "time_fora"], axis=1)

  # Converte o DataFrame para uma única linha e reseta o índice
  df_combinado_casa_mandante = df_combinado_casa_mandante.T.reset_index()
  df_combinado_casa_visitante = df_combinado_casa_visitante.T.reset_index()
  df_combinado_fora_mandante = df_combinado_fora_mandante.T.reset_index()
  df_combinado_fora_visitante = df_combinado_fora_visitante.T.reset_index()

  # Renomeia as colunas
  df_combinado_casa_mandante.columns = ['coluna', 'valor']
  df_combinado_casa_visitante.columns = ['coluna', 'valor']
  df_combinado_fora_mandante.columns = ['coluna', 'valor']
  df_combinado_fora_visitante.columns = ['coluna', 'valor']

  # Concatena o nome da coluna com o valor em uma única coluna
  df_combinado_casa_mandante['coluna_valor'] = df_combinado_casa_mandante['coluna'] + ': ' + df_combinado_casa_mandante['valor'].astype(str)
  df_combinado_casa_visitante['coluna_valor'] = df_combinado_casa_visitante['coluna'] + ': ' + df_combinado_casa_visitante['valor'].astype(str)
  df_combinado_fora_mandante['coluna_valor'] = df_combinado_fora_mandante['coluna'] + ': ' + df_combinado_fora_mandante['valor'].astype(str)
  df_combinado_fora_visitante['coluna_valor'] = df_combinado_fora_visitante['coluna'] + ': ' + df_combinado_fora_visitante['valor'].astype(str)

  # Exclui as colunas desnecessárias
  df_combinado_casa_mandante = df_combinado_casa_mandante[['coluna_valor']]
  df_combinado_casa_visitante = df_combinado_casa_visitante[['coluna_valor']]
  df_combinado_fora_mandante = df_combinado_fora_mandante[['coluna_valor']]
  df_combinado_fora_visitante = df_combinado_fora_visitante[['coluna_valor']]

  # Médias das últimas partidas gerais
  df_medias_time_casa_mandante = pd.DataFrame({'Médias': ['\n'.join(df_combinado_casa_mandante['coluna_valor'])]})
  df_medias_time_casa_visitante = pd.DataFrame({'Médias': ['\n'.join(df_combinado_casa_visitante['coluna_valor'])]})
  df_medias_time_fora_mandante = pd.DataFrame({'Médias': ['\n'.join(df_combinado_fora_mandante['coluna_valor'])]})
  df_medias_time_fora_visitante = pd.DataFrame({'Médias': ['\n'.join(df_combinado_fora_visitante['coluna_valor'])]})

  # Resultados das últimas partidas em casa e fora
  df_resultados_partidas_casa_mandante = pd.DataFrame({'Casa': ['\n'.join(partidas_time_casa_mandante['Geral'])]})
  df_resultados_partidas_casa_visitante = pd.DataFrame({'Fora': ['\n'.join(partidas_time_casa_visitante['Geral'])]})
  df_resultados_partidas_fora_mandante = pd.DataFrame({'Casa': ['\n'.join(partidas_time_fora_mandante['Geral'])]})
  df_resultados_partidas_fora_visitante = pd.DataFrame({'Fora': ['\n'.join(partidas_time_fora_visitante['Geral'])]})

  # Crie um dataframe intermediário com uma única linha e uma única coluna
  df_intermediario_casa_fora = pd.DataFrame({'Resultados': ['Médias']})

  # Transponha os dataframes do time da casa como mandante
  df_resultados_partidas_casa_mandante_t = df_resultados_partidas_casa_mandante.transpose()
  df_medias_time_casa_mandante_t = df_medias_time_casa_mandante.transpose()
  df_intermediario_casa_fora_t = df_intermediario_casa_fora.transpose()

  # Reset o índice dos dataframes transpostos do time da casa como mandante
  df_resultados_partidas_casa_mandante_t = df_resultados_partidas_casa_mandante_t.reset_index(drop=True)
  df_medias_time_casa_mandante_t = df_medias_time_casa_mandante_t.reset_index(drop=True)
  df_intermediario_casa_fora_t = df_intermediario_casa_fora_t.reset_index(drop=True)

  # Transponha os dataframes do time da casa como visitante
  df_resultados_partidas_casa_visitante_t = df_resultados_partidas_casa_visitante.transpose()
  df_medias_time_casa_visitante_t = df_medias_time_casa_visitante.transpose()
  df_intermediario_casa_fora_t = df_intermediario_casa_fora.transpose()

  # Reset o índice dos dataframes transpostos do time da casa como visitante
  df_resultados_partidas_casa_visitante_t = df_resultados_partidas_casa_visitante_t.reset_index(drop=True)
  df_medias_time_casa_visitante_t = df_medias_time_casa_visitante_t.reset_index(drop=True)
  df_intermediario_casa_fora_t = df_intermediario_casa_fora_t.reset_index(drop=True)

  # Transponha os dataframes do time de fora como mandante
  df_resultados_partidas_fora_mandante_t = df_resultados_partidas_fora_mandante.transpose()
  df_medias_time_fora_mandante_t = df_medias_time_fora_mandante.transpose()
  df_intermediario_casa_fora_t = df_intermediario_casa_fora.transpose()

  # Reset o índice dos dataframes transpostos do time de fora como mandante
  df_resultados_partidas_fora_mandante_t = df_resultados_partidas_fora_mandante_t.reset_index(drop=True)
  df_medias_time_fora_mandante_t = df_medias_time_fora_mandante_t.reset_index(drop=True)
  df_intermediario_casa_fora_t = df_intermediario_casa_fora_t.reset_index(drop=True)

  # Transponha os dataframes do time de fora como visitante
  df_resultados_partidas_fora_visitante_t = df_resultados_partidas_fora_visitante.transpose()
  df_medias_time_fora_visitante_t = df_medias_time_fora_visitante.transpose()
  df_intermediario_casa_fora_t = df_intermediario_casa_fora.transpose()

  # Reset o índice dos dataframes transpostos do time de fora como visitante
  df_resultados_partidas_fora_visitante_t = df_resultados_partidas_fora_visitante_t.reset_index(drop=True)
  df_medias_time_fora_visitante_t = df_medias_time_fora_visitante_t.reset_index(drop=True)
  df_intermediario_casa_fora_t = df_intermediario_casa_fora_t.reset_index(drop=True)

  # Concatene os dataframes transpostos verticalmente
  df_concatenado_casa_mandante = pd.concat([df_resultados_partidas_casa_mandante_t, df_intermediario_casa_fora_t, df_medias_time_casa_mandante_t], axis=0)
  df_concatenado_casa_visitante = pd.concat([df_resultados_partidas_casa_visitante_t, df_intermediario_casa_fora_t, df_medias_time_casa_visitante_t], axis=0)
  df_concatenado_fora_mandante = pd.concat([df_resultados_partidas_fora_mandante_t, df_intermediario_casa_fora_t, df_medias_time_fora_mandante_t], axis=0)
  df_concatenado_fora_visitante = pd.concat([df_resultados_partidas_fora_visitante_t, df_intermediario_casa_fora_t, df_medias_time_fora_visitante_t], axis=0)

  # Reset o índice do dataframe resultante
  df_concatenado_casa_mandante = df_concatenado_casa_mandante.reset_index(drop=True)
  df_concatenado_casa_visitante = df_concatenado_casa_visitante.reset_index(drop=True)
  df_concatenado_fora_mandante = df_concatenado_fora_mandante.reset_index(drop=True)
  df_concatenado_fora_visitante = df_concatenado_fora_visitante.reset_index(drop=True)

  # Renomeie a coluna do dataframe resultante
  df_concatenado_casa_mandante.columns = ['Casa']
  df_concatenado_casa_visitante.columns = ['Fora']
  df_concatenado_fora_mandante.columns = ['Casa']
  df_concatenado_fora_visitante.columns = ['Fora']

  # GERAL, CASA e FORA
  df_concatenado_time_casa = pd.concat([df_concatenado_resultados_medias_gerais_time_casa, df_concatenado_casa_mandante, df_concatenado_casa_visitante], axis=1)
  df_concatenado_time_fora = pd.concat([df_concatenado_resultados_medias_gerais_time_fora, df_concatenado_fora_mandante, df_concatenado_fora_visitante], axis=1)

  # CONFRONTOS DIRETOS

  # filtra apenas os confrontos diretos entre o time da casa e o time de fora
  partidas_filtradas = partidas_anteriores.loc[((partidas_anteriores['home_team_name'] == time_casa) & (partidas_anteriores['away_team_name'] == time_fora)) | ((partidas_anteriores['home_team_name'] == time_fora) & (partidas_anteriores['away_team_name'] == time_casa))]
  df1_diretos, df2_diretos, df3_diretos = df_completo_partidas_gerais(partidas_filtradas)

  # calcula o número de vitórias de cada time e o número de empates
  vitorias_time_casa = 0
  vitorias_time_fora = 0
  empates = 0
  for index, row in df2_diretos.iterrows():
      if row['gols_time_casa'] > row['gols_time_fora']:
          if row['time_casa'] == time_casa:
              vitorias_time_casa += 1
          else:
              vitorias_time_fora += 1
      elif row['gols_time_casa'] < row['gols_time_fora']:
          if row['time_casa'] == time_casa:
              vitorias_time_fora += 1
          else:
              vitorias_time_casa += 1
      else:
          empates += 1

  # calcula as porcentagens de vitórias e empates
  total_partidas = vitorias_time_casa + vitorias_time_fora + empates

  # inicialização das variáveis
  porcentagem_vitorias_time_casa = 0
  porcentagem_vitorias_time_fora = 0
  porcentagem_empates = 0
  partidas_over_gols_15 = pd.DataFrame({})
  percentual_over_gols_15 = 0
  partidas_over_gols_25 = pd.DataFrame({})
  percentual_over_gols_25 = 0
  partidas_over_gols_35 = pd.DataFrame({})
  percentual_over_gols_35 = 0
  partidas_over_escanteios_75 = pd.DataFrame({})
  percentual_escanteios_over_75 = 0
  partidas_over_escanteios_85 = pd.DataFrame({})
  percentual_escanteios_over_85 = 0
  partidas_over_escanteios_95 = pd.DataFrame({})
  percentual_escanteios_over_95 = 0
  partidas_over_cartoes_amarelos_15 = pd.DataFrame({})
  percentual_over_cartoes_amarelos_15 = 0
  partidas_over_cartoes_amarelos_25 = pd.DataFrame({})
  percentual_over_cartoes_amarelos_25 = 0
  partidas_over_cartoes_amarelos_35 = pd.DataFrame({})
  percentual_over_cartoes_amarelos_35 = 0
  partidas_over_cartoes_vermelhos_05 = pd.DataFrame({})
  percentual_over_cartoes_vermelhos_05 = 0
  partidas_over_cartoes_vermelhos_15 = pd.DataFrame({})
  percentual_over_cartoes_vermelhos_15 = 0
  partidas_over_cartoes_vermelhos_25 = pd.DataFrame({})
  percentual_over_cartoes_vermelhos_25 = 0
  df_resultados_confrontos_diretos = pd.DataFrame({})
  df_info_confrontos_diretos = pd.DataFrame({})

  if total_partidas != 0:
    porcentagem_vitorias_time_casa = (vitorias_time_casa / total_partidas) * 100
    porcentagem_vitorias_time_fora = (vitorias_time_fora / total_partidas) * 100
    porcentagem_empates = (empates / total_partidas) * 100

    # cria um novo dataframe com as informações de vitórias e empates
    df_resultados_confrontos_diretos = pd.DataFrame({'Vitórias do ' + time_casa: [f'{vitorias_time_casa} ({porcentagem_vitorias_time_casa:.2f}%)'],
                                  'Vitórias do ' + time_fora: [f'{vitorias_time_fora} ({porcentagem_vitorias_time_fora:.2f}%)'],
                                  'Empates': [f'{empates} ({porcentagem_empates:.2f}%)']})

    # Filtrando apenas os confrontos diretos entre os times A e B
    df_confrontos_diretos = df2_diretos

    # Obtendo o total de partidas
    total_partidas = len(df_confrontos_diretos)

    # GOLS

    # Número de partidas com mais de 1.5 gols
    partidas_over_gols_15 = df_confrontos_diretos[(df_confrontos_diretos['gols_time_casa'] +
                                                  df_confrontos_diretos['gols_time_fora']) > 1.5]

    percentual_over_gols_15 = len(partidas_over_gols_15) / total_partidas * 100

    # Número de partidas com mais de 2.5 gols
    partidas_over_gols_25 = df_confrontos_diretos[(df_confrontos_diretos['gols_time_casa'] +
                                                  df_confrontos_diretos['gols_time_fora']) > 2.5]

    percentual_over_gols_25 = len(partidas_over_gols_25) / total_partidas * 100

    # Número de partidas com mais de 3.5 gols
    partidas_over_gols_35 = df_confrontos_diretos[(df_confrontos_diretos['gols_time_casa'] +
                                                  df_confrontos_diretos['gols_time_fora']) > 3.5]
                                                  
    percentual_over_gols_35 = len(partidas_over_gols_35) / total_partidas * 100

    # ESCANTEIOS

    # Número de partidas com mais de 7.5 escanteios
    partidas_over_escanteios_75 = df_confrontos_diretos[(df_confrontos_diretos['escanteios_casa'] +
                                                  df_confrontos_diretos['escanteios_fora']) > 7.5]

    percentual_escanteios_over_75 = len(partidas_over_escanteios_75) / total_partidas * 100

    # Número de partidas com mais de 8.5 escanteios
    partidas_over_escanteios_85 = df_confrontos_diretos[(df_confrontos_diretos['escanteios_casa'] +
                                                  df_confrontos_diretos['escanteios_fora']) > 8.5]

    percentual_escanteios_over_85 = len(partidas_over_escanteios_85) / total_partidas * 100

    # Número de partidas com mais de 9.5 escanteios
    partidas_over_escanteios_95 = df_confrontos_diretos[(df_confrontos_diretos['escanteios_casa'] +
                                                  df_confrontos_diretos['escanteios_fora']) > 9.5]

    percentual_escanteios_over_95 = len(partidas_over_escanteios_95) / total_partidas * 100

    # CARTÕES AMARELOS

    # Número de partidas com mais de 1.5 cartões amarelos
    partidas_over_cartoes_amarelos_15 = df_confrontos_diretos[(df_confrontos_diretos['cartoes_amarelos_casa'] + df_confrontos_diretos['cartoes_amarelos_fora']) > 1.5]
    percentual_over_cartoes_amarelos_15 = len(partidas_over_cartoes_amarelos_15) / total_partidas * 100

    # Número de partidas com mais de 2.5 cartões amarelos
    partidas_over_cartoes_amarelos_25 = df_confrontos_diretos[(df_confrontos_diretos['cartoes_amarelos_casa'] + df_confrontos_diretos['cartoes_amarelos_fora']) > 2.5]
    percentual_over_cartoes_amarelos_25 = len(partidas_over_cartoes_amarelos_25) / total_partidas * 100

    # Número de partidas com mais de 3.5 cartões amarelos
    partidas_over_cartoes_amarelos_35 = df_confrontos_diretos[(df_confrontos_diretos['cartoes_amarelos_casa'] + df_confrontos_diretos['cartoes_amarelos_fora']) > 3.5]
    percentual_over_cartoes_amarelos_35 = len(partidas_over_cartoes_amarelos_35) / total_partidas * 100

    # CARTÕES VERMELHOS

    #Número de partidas com mais de 0.5 cartões vermelhos
    partidas_over_cartoes_vermelhos_05 = df_confrontos_diretos[(df_confrontos_diretos['cartoes_vermelhos_casa'] + df_confrontos_diretos['cartoes_vermelhos_fora']) > 0.5]
    percentual_over_cartoes_vermelhos_05 = len(partidas_over_cartoes_vermelhos_05) / total_partidas * 100

    #Número de partidas com mais de 1.5 cartões vermelhos
    partidas_over_cartoes_vermelhos_15 = df_confrontos_diretos[(df_confrontos_diretos['cartoes_vermelhos_casa'] + df_confrontos_diretos['cartoes_vermelhos_fora']) > 1.5]
    percentual_over_cartoes_vermelhos_15 = len(partidas_over_cartoes_vermelhos_15) / total_partidas * 100

    #Número de partidas com mais de 2.5 cartões vermelhos
    partidas_over_cartoes_vermelhos_25 = df_confrontos_diretos[(df_confrontos_diretos['cartoes_vermelhos_casa'] + df_confrontos_diretos['cartoes_vermelhos_fora']) > 2.5]
    percentual_over_cartoes_vermelhos_25 = len(partidas_over_cartoes_vermelhos_25) / total_partidas * 100
  
  # Criando o novo dataframe
  df_info_confrontos_diretos = pd.DataFrame({'Gols': [f"{percentual_over_gols_15:.2f}% Over 1.5 ({len(partidas_over_gols_15)}/{total_partidas} partidas)",
                                                      '{}% Over 2.5 ({}/{}) partidas'.format(percentual_over_gols_25, len(partidas_over_gols_25), total_partidas),
  '{}% Over 3.5 ({}/{}) partidas'.format(percentual_over_gols_35, len(partidas_over_gols_35), total_partidas)],
  'Escanteios': [f"{percentual_escanteios_over_75:.2f}% Over 7.5 ({len(partidas_over_escanteios_75)}/{total_partidas} partidas)",
  f"{percentual_escanteios_over_85:.2f}% Over 8.5 ({len(partidas_over_escanteios_85)}/{total_partidas} partidas)",
  f"{percentual_escanteios_over_95:.2f}% Over 9.5 ({len(partidas_over_escanteios_95)}/{total_partidas} partidas)"],
  'Cartões Amarelos': [f"{percentual_over_cartoes_amarelos_15:.2f}% Over 1.5 ({len(partidas_over_cartoes_amarelos_15)}/{total_partidas} partidas)",
  f"{percentual_over_cartoes_amarelos_25:.2f}% Over 2.5 ({len(partidas_over_cartoes_amarelos_25)}/{total_partidas} partidas)",
  f"{percentual_over_cartoes_amarelos_35:.2f}% Over 3.5 ({len(partidas_over_cartoes_amarelos_35)}/{total_partidas} partidas)"],
  'Cartões Vermelhos': [f"{percentual_over_cartoes_vermelhos_05:.2f}% Over 0.5 ({len(partidas_over_cartoes_vermelhos_05)}/{total_partidas} partidas)",
  f"{percentual_over_cartoes_vermelhos_15:.2f}% Over 1.5 ({len(partidas_over_cartoes_vermelhos_15)}/{total_partidas} partidas)",
  f"{percentual_over_cartoes_vermelhos_25:.2f}% Over 2.5 ({len(partidas_over_cartoes_vermelhos_25)}/{total_partidas} partidas)"]
  })

  # Imprimindo os dataframes
  return (df_concatenado_time_casa, df_concatenado_time_fora, df_resultados_confrontos_diretos, df_info_confrontos_diretos, time_casa, time_fora, total_partidas)

# Método que gera o dataframe que vai ser utilizado como nova previsão (histórico do campeonato) para a tabela de previsões

def historico_campeonato_previsao_tabela(partidas_casa_time_casa_df1, partidas_casa_time_fora_df1, multi_target_rfc, le, arbitro, time_casa, time_fora):

  df1_casa = nova_previsao_partidas_casa_fora(partidas_casa_time_casa_df1, arbitro, time_casa, time_fora)
  df1_fora = nova_previsao_partidas_casa_fora(partidas_casa_time_fora_df1, arbitro, time_casa, time_fora)

  colunas_casa = ['time_casa', 'time_fora', 'arbitro', 'media_gols_casa', 'media_gols_primeiro_casa', 'media_escanteios_casa', 'media_cartoes_amarelos_casa', 'media_cartoes_vermelhos_casa', 'media_cartoes_primeiro_casa', 'media_cartoes_segundo_casa', 'media_chutes_no_gol_casa', 'media_chutes_fora_do_gol_casa', 'media_faltas_casa', 'media_posse_de_bola_casa']
  colunas_fora = ['media_gols_fora', 'media_gols_primeiro_fora', 'media_escanteios_fora', 'media_cartoes_amarelos_fora', 'media_cartoes_vermelhos_fora', 'media_cartoes_primeiro_fora', 'media_cartoes_segundo_fora', 'media_chutes_no_gol_fora', 'media_chutes_fora_do_gol_fora', 'media_faltas_fora', 'media_posse_de_bola_fora']

  df_combinado = pd.concat([df1_casa[colunas_casa], df1_fora[colunas_fora]], axis=1)

  df_combinado = df_combinado.reindex(columns=['time_casa', 'time_fora','arbitro','media_gols_casa','media_gols_fora','media_gols_primeiro_casa',
      'media_gols_primeiro_fora','media_escanteios_casa', 'media_escanteios_fora','media_cartoes_amarelos_casa', 'media_cartoes_amarelos_fora',
      'media_cartoes_vermelhos_casa', 'media_cartoes_vermelhos_fora','media_cartoes_primeiro_casa', 'media_cartoes_primeiro_fora',
      'media_cartoes_segundo_casa', 'media_cartoes_segundo_fora','media_chutes_no_gol_casa', 'media_chutes_no_gol_fora','media_chutes_fora_do_gol_casa',
      'media_chutes_fora_do_gol_fora','media_faltas_casa', 'media_faltas_fora','media_posse_de_bola_casa','media_posse_de_bola_fora'])

  return(fazer_previsao_tabela(df_combinado, multi_target_rfc, le))

# Método que retorna as novas previsões com base nas últimas partidas jogando em casa e fora para a tabela de previsões

def ultimas_partidas_casa_fora_previsao_tabela(partidas_casa_time_casa_df1, partidas_casa_time_fora_df1, ultimas_n_partidas, num_partidas, time_casa, time_fora, multi_target_rfc, le, arbitro):

  df1_casa = nova_previsao_partidas_casa_fora(partidas_casa_time_casa_df1, arbitro, time_casa, time_fora)
  df1_fora = nova_previsao_partidas_casa_fora(partidas_casa_time_fora_df1, arbitro, time_casa, time_fora)

  colunas_casa = ['time_casa', 'time_fora', 'arbitro', 'media_gols_casa', 'media_gols_primeiro_casa', 'media_escanteios_casa', 'media_cartoes_amarelos_casa', 'media_cartoes_vermelhos_casa', 'media_cartoes_primeiro_casa', 'media_cartoes_segundo_casa', 'media_chutes_no_gol_casa', 'media_chutes_fora_do_gol_casa', 'media_faltas_casa', 'media_posse_de_bola_casa']
  colunas_fora = ['media_gols_fora', 'media_gols_primeiro_fora', 'media_escanteios_fora', 'media_cartoes_amarelos_fora', 'media_cartoes_vermelhos_fora', 'media_cartoes_primeiro_fora', 'media_cartoes_segundo_fora', 'media_chutes_no_gol_fora', 'media_chutes_fora_do_gol_fora', 'media_faltas_fora', 'media_posse_de_bola_fora']

  df_combinado = pd.concat([df1_casa[colunas_casa], df1_fora[colunas_fora]], axis=1)

  df_combinado = df_combinado.reindex(columns=['time_casa', 'time_fora','arbitro','media_gols_casa','media_gols_fora','media_gols_primeiro_casa',
      'media_gols_primeiro_fora','media_escanteios_casa', 'media_escanteios_fora','media_cartoes_amarelos_casa', 'media_cartoes_amarelos_fora',
      'media_cartoes_vermelhos_casa', 'media_cartoes_vermelhos_fora','media_cartoes_primeiro_casa', 'media_cartoes_primeiro_fora',
      'media_cartoes_segundo_casa', 'media_cartoes_segundo_fora','media_chutes_no_gol_casa', 'media_chutes_no_gol_fora','media_chutes_fora_do_gol_casa',
      'media_chutes_fora_do_gol_fora','media_faltas_casa', 'media_faltas_fora','media_posse_de_bola_casa','media_posse_de_bola_fora'])

  return(fazer_previsao_tabela(df_combinado, multi_target_rfc, le))

# Método que retorna as novas previsões com base nas últimas partidas de modo geral para a tabela de previsões

def ultimas_partidas_gerais_previsao_tabela(partidas_casa_time_casa_df1, partidas_casa_time_fora_df1, ultimas_n_partidas, num_partidas, time_casa, time_fora, multi_target_rfc, le, arbitro):

  df_combinado = nova_previsao_partidas_gerais(ultimas_n_partidas, arbitro, num_partidas, time_casa, time_fora)

  df_combinado = df_combinado.reindex(columns=['time_casa', 'time_fora','arbitro','media_gols_casa','media_gols_fora','media_gols_primeiro_casa',
      'media_gols_primeiro_fora','media_escanteios_casa', 'media_escanteios_fora','media_cartoes_amarelos_casa', 'media_cartoes_amarelos_fora',
      'media_cartoes_vermelhos_casa', 'media_cartoes_vermelhos_fora','media_cartoes_primeiro_casa', 'media_cartoes_primeiro_fora',
      'media_cartoes_segundo_casa', 'media_cartoes_segundo_fora','media_chutes_no_gol_casa', 'media_chutes_no_gol_fora','media_chutes_fora_do_gol_casa',
      'media_chutes_fora_do_gol_fora','media_faltas_casa', 'media_faltas_fora','media_posse_de_bola_casa','media_posse_de_bola_fora'])

  return(fazer_previsao_tabela(df_combinado, multi_target_rfc, le))

# Método que retorna as novas previsões com base nos confrontos diretos para a tabela de previsões

def confrontos_diretos_previsao_tabela(df3, multi_target_rfc, le):

  return(fazer_previsao_tabela(df3, multi_target_rfc, le))

# Método que faz as novas previsões para a tabela de previsões

def fazer_previsao_tabela(df_para_previsao, multi_target_rfc, le):

  # Passo 3: Codificar as variáveis categóricas
  df_para_previsao['time_casa'] = le.fit_transform(df_para_previsao['time_casa'])
  df_para_previsao['time_fora'] = le.fit_transform(df_para_previsao['time_fora'])
  df_para_previsao['arbitro'] = le.fit_transform(df_para_previsao['arbitro'])

  nova_previsao = multi_target_rfc.predict(df_para_previsao)
  return (nova_previsao)

# Método que gera a tabela de previsões

def gerar_tabela(time_casa, time_fora, arbitro, multi_target_rfc, le, partidas_anteriores, acuracia):

  # agrupando as partidas por time da casa e time de fora
  grupos_casa = partidas_anteriores.groupby('home_team_name')
  grupos_fora = partidas_anteriores.groupby('away_team_name')

  # salvando os nomes dos times da casa em uma lista para futura verificação
  times_da_casa = partidas_anteriores['home_team_name'].unique()
  # salvando os nomes dos times de fora em uma lista para futura verificação
  times_de_fora = partidas_anteriores['away_team_name'].unique()
  # salvando os nomes dos árbitros em uma lista para futura verificação
  arbitros = partidas_anteriores['referee'].unique()

  # criando um dataframe vazio que vai armazenar as últimas n partidas de cada time da casa
  dataframe_time_casa = pd.DataFrame(columns=partidas_anteriores.columns)

  # criando um dataframe vazio que vai armazenar as últimas n partidas de cada time de fora
  dataframe_time_fora = pd.DataFrame(columns=partidas_anteriores.columns)

  # número de partidas para o padrão 3, 4 e 5
  num_partidas_3_casa_fora = 3
  num_partidas_5_casa_fora = 5
  num_partidas_10_casa_fora = 10

  # número de partidas para o padrão 6, 7 e 8
  num_partidas_3_geral = 3
  num_partidas_5_geral = 5
  num_partidas_10_geral = 10

  # Padrão 1 - Confrontos Diretos
  partidas_filtradas_padrao_1 = partidas_anteriores.loc[((partidas_anteriores['home_team_name'] == time_casa) & (partidas_anteriores['away_team_name'] == time_fora)) | ((partidas_anteriores['home_team_name'] == time_fora) & (partidas_anteriores['away_team_name'] == time_casa))]
  df1_padrao_1, df2_padrao_1, df3_padrao_1 = df_completo_partidas_gerais(partidas_filtradas_padrao_1)
  df4_padrao_1 = nova_previsao_direto(partidas_filtradas_padrao_1, arbitro, time_casa, time_fora)

  tabela_padrao_1 = confrontos_diretos_previsao_tabela(df4_padrao_1, multi_target_rfc, le)

  # Padrão 2 - Histórico no campeonato
  for index, row in partidas_anteriores.iterrows():
    partidas_casa_time_casa_df1_padrao_2 = partidas_anteriores[(partidas_anteriores['home_team_name'] == time_casa)]
    partidas_casa_time_fora_df1_padrao_2 = partidas_anteriores[(partidas_anteriores['away_team_name'] == time_fora)]

  tabela_padrao_2 = historico_campeonato_previsao_tabela(partidas_casa_time_casa_df1_padrao_2,partidas_casa_time_fora_df1_padrao_2, multi_target_rfc, le, arbitro, time_casa, time_fora)

  # Padrão 3 - Últimas 3 partidas em casa e últimas 3 partidas fora de casa  
  dataframe_time_casa_padrao_3 = pd.DataFrame()
  dataframe_time_fora_padrao_3 = pd.DataFrame()

  for time, grupo in grupos_casa:
    ultimas_partidas_time_padrao_3 = grupo.sort_values('data').tail(num_partidas_3_casa_fora)
    dataframe_time_casa_padrao_3 = pd.concat([dataframe_time_casa_padrao_3, ultimas_partidas_time_padrao_3])

  for time, grupo in grupos_fora:
    ultimas_partidas_time_padrao_3 = grupo.sort_values('data').tail(num_partidas_3_casa_fora)
    dataframe_time_fora_padrao_3 = pd.concat([dataframe_time_fora_padrao_3, ultimas_partidas_time_padrao_3])

  for index, row in partidas_anteriores.iterrows():
    partidas_casa_time_casa_df1_padrao_3 = dataframe_time_casa_padrao_3[(dataframe_time_casa_padrao_3['home_team_name'] == time_casa)]
    partidas_casa_time_fora_df1_padrao_3 = dataframe_time_fora_padrao_3[(dataframe_time_fora_padrao_3['away_team_name'] == time_fora)]

    ultimas_n_partidas_padrao_3 = pd.concat([partidas_casa_time_casa_df1_padrao_3,partidas_casa_time_fora_df1_padrao_3])

  tabela_padrao_3 =ultimas_partidas_casa_fora_previsao_tabela(partidas_casa_time_casa_df1_padrao_3,partidas_casa_time_fora_df1_padrao_3,ultimas_n_partidas_padrao_3,num_partidas_3_casa_fora,time_casa,time_fora, multi_target_rfc, le, arbitro)

  # Padrão 4 - Últimas 5 partidas em casa e últimas 5 partidas fora de casa
  dataframe_time_casa_padrao_4 = pd.DataFrame()
  dataframe_time_fora_padrao_4 = pd.DataFrame()

  for time, grupo in grupos_casa:
    ultimas_partidas_time_padrao_4 = grupo.sort_values('data').tail(num_partidas_5_casa_fora)
    dataframe_time_casa_padrao_4 = pd.concat([dataframe_time_casa_padrao_4, ultimas_partidas_time_padrao_4])

  for time, grupo in grupos_fora:
    ultimas_partidas_time_padrao_4 = grupo.sort_values('data').tail(num_partidas_5_casa_fora)
    dataframe_time_fora_padrao_4 = pd.concat([dataframe_time_fora_padrao_4, ultimas_partidas_time_padrao_4])

  for index, row in partidas_anteriores.iterrows():
    partidas_casa_time_casa_df1_padrao_4 = dataframe_time_casa_padrao_4[(dataframe_time_casa_padrao_4['home_team_name'] == time_casa)]
    partidas_casa_time_fora_df1_padrao_4 = dataframe_time_fora_padrao_4[(dataframe_time_fora_padrao_4['away_team_name'] == time_fora)]

    ultimas_n_partidas_padrao_4 = pd.concat([partidas_casa_time_casa_df1_padrao_4,partidas_casa_time_fora_df1_padrao_4])

  tabela_padrao_4 = ultimas_partidas_casa_fora_previsao_tabela(partidas_casa_time_casa_df1_padrao_4,partidas_casa_time_fora_df1_padrao_4,ultimas_n_partidas_padrao_4,num_partidas_5_casa_fora,time_casa,time_fora, multi_target_rfc, le, arbitro)

  # Padrão 5 - Últimas 10 partidas em casa e últimas 10 partidas fora de casa
  dataframe_time_casa_padrao_5 = pd.DataFrame()
  dataframe_time_fora_padrao_5 = pd.DataFrame()

  for time, grupo in grupos_casa:
    ultimas_partidas_time_padrao_5 = grupo.sort_values('data').tail(num_partidas_10_casa_fora)
    dataframe_time_casa_padrao_5 = pd.concat([dataframe_time_casa_padrao_5, ultimas_partidas_time_padrao_5])

  for time, grupo in grupos_fora:
    ultimas_partidas_time_padrao_5 = grupo.sort_values('data').tail(num_partidas_10_casa_fora)
    dataframe_time_fora_padrao_5 = pd.concat([dataframe_time_fora_padrao_5, ultimas_partidas_time_padrao_5])

  for index, row in partidas_anteriores.iterrows():
    partidas_casa_time_casa_df1_padrao_5 = dataframe_time_casa_padrao_5[(dataframe_time_casa_padrao_5['home_team_name'] == time_casa)]
    partidas_casa_time_fora_df1_padrao_5 = dataframe_time_fora_padrao_5[(dataframe_time_fora_padrao_5['away_team_name'] == time_fora)]

    ultimas_n_partidas_padrao_5 = pd.concat([partidas_casa_time_casa_df1_padrao_5,partidas_casa_time_fora_df1_padrao_5])

  tabela_padrao_5 = ultimas_partidas_casa_fora_previsao_tabela(partidas_casa_time_casa_df1_padrao_5,partidas_casa_time_fora_df1_padrao_5,ultimas_n_partidas_padrao_5,num_partidas_10_casa_fora,time_casa,time_fora, multi_target_rfc, le, arbitro)

  # Padrão 6 - Últimas 3 partidas de modo geral

  # Criar um dataframe vazio para armazenar as últimas n partidas de cada time
  df_ultimas_partidas_padrao_6 = pd.DataFrame()

  # Iterar sobre cada time
  for time in partidas_anteriores['home_team_name'].unique():
      # Filtrar as partidas do time em questão
      partidas_time_padrao_6 = partidas_anteriores[(partidas_anteriores['home_team_name'] == time) | (partidas_anteriores['away_team_name'] == time)]  
      # Ordenar as partidas do time pela data em ordem decrescente e selecionar as últimas n partidas
      ultimas_partidas_padrao_6 = partidas_time_padrao_6.sort_values(by='data', ascending=False).head(num_partidas_3_geral) 
      # Adicionar as últimas n partidas do time ao dataframe final
      df_ultimas_partidas_padrao_6 = pd.concat([df_ultimas_partidas_padrao_6, ultimas_partidas_padrao_6])

  # Ajustar o índice do dataframe final
  df_ultimas_partidas_padrao_6.reset_index(drop=True, inplace=True)

  for index, row in df_ultimas_partidas_padrao_6.iterrows():
    partidas_casa_time_casa_df1_padrao_6 = df_ultimas_partidas_padrao_6[(df_ultimas_partidas_padrao_6['home_team_name'] == time_casa) | (df_ultimas_partidas_padrao_6['away_team_name'] == time_casa)]
    partidas_casa_time_fora_df1_padrao_6 = df_ultimas_partidas_padrao_6[(df_ultimas_partidas_padrao_6['home_team_name'] == time_fora) | (df_ultimas_partidas_padrao_6['away_team_name'] == time_fora)]

    partidas_casa_time_casa_df1_padrao_6 = partidas_casa_time_casa_df1_padrao_6.drop(partidas_casa_time_casa_df1_padrao_6.index[num_partidas_3_geral:])
    partidas_casa_time_fora_df1_padrao_6 = partidas_casa_time_fora_df1_padrao_6.drop(partidas_casa_time_fora_df1_padrao_6.index[:num_partidas_3_geral])

    ultimas_n_partidas_padrao_6 = pd.concat([partidas_casa_time_casa_df1_padrao_6,partidas_casa_time_fora_df1_padrao_6])

  tabela_padrao_6 = ultimas_partidas_gerais_previsao_tabela(partidas_casa_time_casa_df1_padrao_6,partidas_casa_time_fora_df1_padrao_6,ultimas_n_partidas_padrao_6,num_partidas_3_geral,time_casa,time_fora, multi_target_rfc, le, arbitro)

  # Padrão 7 - Últimas 5 partidas de modo geral

  # Criar um dataframe vazio para armazenar as últimas n partidas de cada time
  df_ultimas_partidas_padrao_7 = pd.DataFrame()

  # Iterar sobre cada time
  for time in partidas_anteriores['home_team_name'].unique():
      # Filtrar as partidas do time em questão
      partidas_time_padrao_7 = partidas_anteriores[(partidas_anteriores['home_team_name'] == time) | (partidas_anteriores['away_team_name'] == time)]  
      # Ordenar as partidas do time pela data em ordem decrescente e selecionar as últimas n partidas
      ultimas_partidas_padrao_7 = partidas_time_padrao_7.sort_values(by='data', ascending=False).head(num_partidas_5_geral) 
      # Adicionar as últimas n partidas do time ao dataframe final
      df_ultimas_partidas_padrao_7 = pd.concat([df_ultimas_partidas_padrao_7, ultimas_partidas_padrao_7])

  # Ajustar o índice do dataframe final
  df_ultimas_partidas_padrao_7.reset_index(drop=True, inplace=True)

  for index, row in df_ultimas_partidas_padrao_7.iterrows():
    partidas_casa_time_casa_df1_padrao_7 = df_ultimas_partidas_padrao_7[(df_ultimas_partidas_padrao_7['home_team_name'] == time_casa) | (df_ultimas_partidas_padrao_7['away_team_name'] == time_casa)]
    partidas_casa_time_fora_df1_padrao_7 = df_ultimas_partidas_padrao_7[(df_ultimas_partidas_padrao_7['home_team_name'] == time_fora) | (df_ultimas_partidas_padrao_7['away_team_name'] == time_fora)]

    partidas_casa_time_casa_df1_padrao_7 = partidas_casa_time_casa_df1_padrao_7.drop(partidas_casa_time_casa_df1_padrao_7.index[num_partidas_5_geral:])
    partidas_casa_time_fora_df1_padrao_7 = partidas_casa_time_fora_df1_padrao_7.drop(partidas_casa_time_fora_df1_padrao_7.index[:num_partidas_5_geral])

    ultimas_n_partidas_padrao_7 = pd.concat([partidas_casa_time_casa_df1_padrao_7,partidas_casa_time_fora_df1_padrao_7])

  tabela_padrao_7 = ultimas_partidas_gerais_previsao_tabela(partidas_casa_time_casa_df1_padrao_7,partidas_casa_time_fora_df1_padrao_7,ultimas_n_partidas_padrao_7,num_partidas_5_geral,time_casa,time_fora, multi_target_rfc, le, arbitro)

  # Padrão 8 - Últimas 10 partidas de modo geral

  # Criar um dataframe vazio para armazenar as últimas n partidas de cada time
  df_ultimas_partidas_padrao_8 = pd.DataFrame()

  # Iterar sobre cada time
  for time in partidas_anteriores['home_team_name'].unique():
      # Filtrar as partidas do time em questão
      partidas_time_padrao_8 = partidas_anteriores[(partidas_anteriores['home_team_name'] == time) | (partidas_anteriores['away_team_name'] == time)]  
      # Ordenar as partidas do time pela data em ordem decrescente e selecionar as últimas n partidas
      ultimas_partidas_padrao_8 = partidas_time_padrao_8.sort_values(by='data', ascending=False).head(num_partidas_10_geral) 
      # Adicionar as últimas n partidas do time ao dataframe final
      df_ultimas_partidas_padrao_8 = pd.concat([df_ultimas_partidas_padrao_8, ultimas_partidas_padrao_8])

  # Ajustar o índice do dataframe final
  df_ultimas_partidas_padrao_8.reset_index(drop=True, inplace=True)

  for index, row in df_ultimas_partidas_padrao_8.iterrows():
    partidas_casa_time_casa_df1_padrao_8 = df_ultimas_partidas_padrao_8[(df_ultimas_partidas_padrao_8['home_team_name'] == time_casa) | (df_ultimas_partidas_padrao_8['away_team_name'] == time_casa)]
    partidas_casa_time_fora_df1_padrao_8 = df_ultimas_partidas_padrao_8[(df_ultimas_partidas_padrao_8['home_team_name'] == time_fora) | (df_ultimas_partidas_padrao_8['away_team_name'] == time_fora)]

    partidas_casa_time_casa_df1_padrao_8 = partidas_casa_time_casa_df1_padrao_8.drop(partidas_casa_time_casa_df1_padrao_8.index[num_partidas_10_geral:])
    partidas_casa_time_fora_df1_padrao_8 = partidas_casa_time_fora_df1_padrao_8.drop(partidas_casa_time_fora_df1_padrao_8.index[:num_partidas_10_geral])

    ultimas_n_partidas_padrao_8 = pd.concat([partidas_casa_time_casa_df1_padrao_8,partidas_casa_time_fora_df1_padrao_8])

  tabela_padrao_8 = ultimas_partidas_gerais_previsao_tabela(partidas_casa_time_casa_df1_padrao_8,partidas_casa_time_fora_df1_padrao_8,ultimas_n_partidas_padrao_8,num_partidas_10_geral,time_casa,time_fora, multi_target_rfc, le, arbitro)

  # inicializa as listas vazias
  elemento_1 = []
  elemento_2 = []
  elemento_3 = []
  elemento_4 = []
  elemento_5 = []
  elemento_6 = []
  elemento_7 = []
  elemento_8 = []

  # laço para percorrer cada elemento dos objetos
  for i in range(12):
      elemento_1.append(tabela_padrao_1[0][i])
      elemento_2.append(tabela_padrao_2[0][i])
      elemento_3.append(tabela_padrao_3[0][i])
      elemento_4.append(tabela_padrao_4[0][i])
      elemento_5.append(tabela_padrao_5[0][i])
      elemento_6.append(tabela_padrao_6[0][i])
      elemento_7.append(tabela_padrao_7[0][i])
      elemento_8.append(tabela_padrao_8[0][i])

  # Crie uma lista com todas as listas
  todas_listas = [elemento_1, elemento_2, elemento_3, elemento_4, elemento_5, elemento_6, elemento_7, elemento_8]

  # Crie um dicionário com as listas e suas respectivas chaves
  dicionario = {}
  for i, lista in enumerate(todas_listas):
      dicionario[f'elemento_{i+1}'] = lista

  # Crie o dataframe a partir do dicionário
  df = pd.DataFrame(dicionario)
  df.columns = ['Padrão 1 - Confrontos diretos', 'Padrão 2 - Histórico do campeonato',
                'Padrão 3 - Últimas 3 partidas em casa e últimas 3 partidas fora',
                'Padrão 4 - Últimas 5 partidas em casa e últimas 5 partidas fora',
                'Padrão 5 - Últimas 10 partidas em casa e últimas 10 partidas fora',
                'Padrão 6 - Últimas 3 partidas de modo geral',
                'Padrão 7 - Últimas 5 partidas de modo geral',
                'Padrão 8 - Últimas 10 partidas de modo geral']
  
  # definir uma função para contar as ocorrências de valores em uma linha e concatenar os resultados em uma string
  def contar_e_concatenar_linha(row):
      contagem = row.value_counts().to_dict()
      contagem_string = ''
      for valor, ocorrencias in contagem.items():
          contagem_string += f'{valor.capitalize()} apareceu {ocorrencias} vezes nos padrões, '
      contagem_string = contagem_string[:-2] + '.'
      return contagem_string

  # Função para verificar a diferença de ocorrências entre os valores de uma linha
  def verificar_diferenca_ocorrencias(row):
    if 'Resultado - Padrões' in row.index:
        row = row.drop('Resultado - Padrões')  # Remover a coluna "Resultados - Padrões" do objeto row

    contagem = row.value_counts().to_dict()  # Conta a ocorrência dos valores na linha e cria um dicionário com o resultado
    valores = []  # Lista para armazenar os valores encontrados
    ocorrencias = []  # Lista para armazenar as ocorrências dos valores encontrados
    for valor, ocorrencia in contagem.items():
        valores.append(valor.capitalize())  # Adiciona o valor à lista de valores, convertendo a primeira letra para maiúscula
        ocorrencias.append(ocorrencia)  # Adiciona a ocorrência do valor à lista de ocorrências

    diff_max = max(ocorrencias) - min(ocorrencias)  # Calcula a diferença máxima entre as ocorrências dos valores
    if diff_max >= 4:  # Verifica se a diferença máxima é maior ou igual a 4
        resultado = ''  # String para armazenar o resultado
        for i in range(len(valores)):
            if ocorrencias[i] == 0:  # Verifica se a ocorrência do valor é 0
                continue  # Se for 0, pula para o próximo valor
            resultado += f'{valores[i]} ({ocorrencias[i]}), '  # Adiciona os valores e suas ocorrências à string de resultado
        resultado = resultado[:-2]  # Remove a vírgula e o espaço extra no final da string
    else:
        resultado = '--------------------------'  # String de resultado para diferenças menores que 4

    # Verifica se a ocorrência de um valor é igual a 8 e atualiza a string de resultado
    for i in range(len(valores)):
        if ocorrencias[i] == 8:
            resultado = f'{valores[i]} ({ocorrencias[i]})'

    return resultado

  def verificar_diferenca_ocorrencias_padroes(row):
    # Remove as colunas 'Resultado - Padrões' e 'Diferença de Ocorrências' do DataFrame 'row', se existirem
    row = row.drop(['Resultado - Padrões', 'Diferença de Ocorrências'], errors='ignore')

    # Cria um dicionário contendo a contagem de ocorrências de cada valor na coluna 'row'
    contagem = row.value_counts().to_dict()

    # Inicializa listas vazias para armazenar os valores e suas ocorrências
    valores = []
    ocorrencias = []

    # Itera sobre os itens do dicionário de contagem para preencher as listas de valores e ocorrências
    for valor, ocorrencia in contagem.items():
        valores.append(valor.capitalize())  # Adiciona o valor capitalizado à lista de valores
        ocorrencias.append(ocorrencia)  # Adiciona a ocorrência à lista de ocorrências

    # Calcula a diferença máxima entre as ocorrências
    diff_max = max(ocorrencias) - min(ocorrencias)

    # Verifica se a diferença máxima entre as ocorrências é maior ou igual a 4
    if diff_max >= 4:
        resultado = ''  # Inicializa a string de resultado
        # Ordena os valores com base nas ocorrências em ordem decrescente
        valores_ordenados = [valor for _, valor in sorted(zip(ocorrencias, valores), key=lambda pair: pair[0], reverse=True)]

        # Itera sobre os valores ordenados para construir a string de resultado
        for valor in valores_ordenados:
            padroes = []  # Inicializa uma lista vazia para armazenar os números de padrão
            # Itera sobre as colunas do DataFrame 'row' para verificar se o valor corresponde a algum padrão
            for coluna in row.index:
                if 'Padrão' in coluna and row[coluna].lower() == valor.lower():
                    padrao_num = int(coluna.split(' ')[1])  # Extrai o número de padrão do nome da coluna
                    padroes.append(str(padrao_num))  # Adiciona o número de padrão à lista de padrões
            resultado += f'{valor} ({",".join(padroes)}), '  # Adiciona o valor e seus padrões à string de resultado
        resultado = resultado.rstrip(', ')  # Remove a última vírgula e espaços em branco da string de resultado
    else:
        resultado = '--------------------------'  # Atribui uma string com traços para a diferença de ocorrências menor que 4

    # Verifica se a ocorrência de um valor é igual a 8 e atualiza a string de resultado, se necessário
    for i in range(len(valores)):
        if ocorrencias[i] == 8:
            resultado = f'{valores[i]} ({ocorrencias[i]})'

    return resultado  # Retorna a string de resultado

  # aplicar a função em cada linha do dataframe
  df['Resultado - Padrões'] = df.apply(contar_e_concatenar_linha, axis=1)

  # Aplicar a função em cada linha do dataframe
  df['Diferença de Ocorrências'] = df.apply(verificar_diferenca_ocorrencias, axis=1)

  # Aplicar a função em cada linha do dataframe
  df['Ocorrências - Padrões'] = df.apply(verificar_diferenca_ocorrencias_padroes, axis=1)

  # Lista de novos elementos
  new_cols = ['Resultado da partida', 'Resultado do intervalo', 'Número de gols over under', 'Ambas equipes marcaram',
              'Número de cartões amarelos', 'Número de cartões vermelhos', 'Número de cartões totais',
              'Ambas equipes receberam cartões', 'Cartões aconteceram em ambos os tempos', 'Número de escanteios',
              'Número de cartões no primeiro tempo', 'Número de cartões no segundo tempo']

  # Adicionando as novas colunas no início do DataFrame
  df.insert(0, 'Variáveis-alvo', new_cols)

  novo_df = df[['Variáveis-alvo', 'Ocorrências - Padrões']].copy()

  df_acuracia = pd.concat([novo_df, acuracia], axis=1)

  legenda = ['Confrontos diretos', 
           'Histórico do campeonato',
           'Últimas 3 partidas em casa e últimas 3 partidas fora',
           'Últimas 5 partidas em casa e últimas 5 partidas fora',
           'Últimas 10 partidas em casa e últimas 10 partidas fora',
           'Últimas 3 partidas de modo geral',
           'Últimas 5 partidas de modo geral',
           'Últimas 10 partidas de modo geral']

  codigos = list(range(1, 9))

  legenda_df = pd.DataFrame({'Código': codigos, 'Descrição': legenda})

  return(df_acuracia, legenda_df)

# Método que calcula os padrões mais assertivos

def padroes_assertivos(partidas_df, data_da_partida):

  partidas_anteriores = partidas_df[partidas_df['data'] < data_da_partida]

  # treinando o modelo
  multi_target_rfc, le, y_test, y_pred, df1_ml = modelo_ml(partidas_df, data_da_partida)

  # Passando o arquivo das partidas para o método df_completo
  df1_ml, df2_ml, df3_ml = df_completo_partidas_casa_fora(partidas_df)

  # Convertendo a data de treino para o formato datetime e definindo o valor da data de treino
  data_treino = data_da_partida
  data_treino = datetime.strptime(data_treino, '%Y-%m-%d').date()

  # Convertendo a data de teste para o formato datetime e definindo o valor da data de teste
  data_teste = data_da_partida
  data_teste = datetime.strptime(data_teste, '%Y-%m-%d').date()

  # Selecionando as partidas anteriores a data da partida
  partidas_antes_da_data = df1_ml[df1_ml['data_partida'] < data_treino]

  # Selecionando a última partida antes da data passada
  ultima_partida_antes_da_data = partidas_antes_da_data.iloc[-1]

  # Obtendo a rodada correspondente à última partida antes da data passada
  rodada_da_ultima_partida = ultima_partida_antes_da_data['rodada']

  # Encontrar o número da próxima rodada
  proxima_rodada = rodada_da_ultima_partida + 1

  # Filtrar o DataFrame original para obter todas as partidas na próxima rodada
  partidas_proxima_rodada = df1_ml[df1_ml['rodada'] == proxima_rodada]

  # Criação de uma lista vazia para armazenar as listas das partidas
  lista_partidas = []

  # agrupando as partidas por time da casa e time de fora
  grupos_casa = partidas_anteriores.groupby('home_team_name')
  grupos_fora = partidas_anteriores.groupby('away_team_name')

  # Itera sobre cada linha do DataFrame
  for index, row in partidas_proxima_rodada.iterrows():
      # Extrai as informações de cada linha
      time_casa = row['time_casa']
      time_fora = row['time_fora']
      arbitro = row['arbitro']
      
      # Cria uma lista com as informações da partida atual
      partida = [time_casa, time_fora, arbitro]
      
      # Adiciona a lista da partida à lista maior
      lista_partidas.append(partida)

  # número de partidas para o padrão 3, 4 e 5
  num_partidas_3_casa_fora = 3
  num_partidas_5_casa_fora = 5
  num_partidas_10_casa_fora = 10

  # número de partidas para o padrão 6, 7 e 8
  num_partidas_3_geral = 3
  num_partidas_5_geral = 5
  num_partidas_10_geral = 10

  def gerar_tabelas_padrao(lista_partidas):

      tabelas_padrao_1 = []
      tabelas_padrao_2 = []
      tabelas_padrao_3 = []
      tabelas_padrao_4 = []
      tabelas_padrao_5 = []
      tabelas_padrao_6 = []
      tabelas_padrao_7 = []
      tabelas_padrao_8 = []

      for partida in lista_partidas:
          time_casa, time_fora, arbitro = partida
          
          # Padrão 1 - Confrontos Diretos

          partidas_filtradas_padrao_1 = partidas_anteriores.loc[((partidas_anteriores['home_team_name'] == time_casa) & (partidas_anteriores['away_team_name'] == time_fora)) | ((partidas_anteriores['home_team_name'] == time_fora) & (partidas_anteriores['away_team_name'] == time_casa))]
          df1_padrao_1, df2_padrao_1, df3_padrao_1 = df_completo_partidas_gerais(partidas_filtradas_padrao_1)
          df4_padrao_1 = nova_previsao_direto(partidas_filtradas_padrao_1, arbitro, time_casa, time_fora)

          tabela_padrao_1 = confrontos_diretos_previsao_tabela(df4_padrao_1, multi_target_rfc, le)

          tabelas_padrao_1.append(tabela_padrao_1)
          
          # Padrão 2 - Histórico no campeonato

          partidas_casa_time_casa_df1_padrao_2 = partidas_anteriores[(partidas_anteriores['home_team_name'] == time_casa)]
          partidas_casa_time_fora_df1_padrao_2 = partidas_anteriores[(partidas_anteriores['away_team_name'] == time_fora)]

          tabela_padrao_2 = historico_campeonato_previsao_tabela(partidas_casa_time_casa_df1_padrao_2, partidas_casa_time_fora_df1_padrao_2, multi_target_rfc, le, arbitro, time_casa, time_fora)

          tabelas_padrao_2.append(tabela_padrao_2)
          
          # Padrão 3 - Últimas 3 partidas em casa e últimas 3 partidas fora de casa  

          dataframe_time_casa_padrao_3 = pd.DataFrame()
          dataframe_time_fora_padrao_3 = pd.DataFrame()

          for time, grupo in grupos_casa:
              ultimas_partidas_time_padrao_3 = grupo.sort_values('data').tail(num_partidas_3_casa_fora)
              dataframe_time_casa_padrao_3 = pd.concat([dataframe_time_casa_padrao_3, ultimas_partidas_time_padrao_3])

          for time, grupo in grupos_fora:
              ultimas_partidas_time_padrao_3 = grupo.sort_values('data').tail(num_partidas_3_casa_fora)
              dataframe_time_fora_padrao_3 = pd.concat([dataframe_time_fora_padrao_3, ultimas_partidas_time_padrao_3])

          for index, row in partidas_anteriores.iterrows():
              partidas_casa_time_casa_df1_padrao_3 = dataframe_time_casa_padrao_3[(dataframe_time_casa_padrao_3['home_team_name'] == time_casa)]
              partidas_casa_time_fora_df1_padrao_3 = dataframe_time_fora_padrao_3[(dataframe_time_fora_padrao_3['away_team_name'] == time_fora)]

              ultimas_n_partidas_padrao_3 = pd.concat([partidas_casa_time_casa_df1_padrao_3, partidas_casa_time_fora_df1_padrao_3])

          tabela_padrao_3 = ultimas_partidas_casa_fora_previsao_tabela(partidas_casa_time_casa_df1_padrao_3, partidas_casa_time_fora_df1_padrao_3, ultimas_n_partidas_padrao_3, num_partidas_3_casa_fora, time_casa, time_fora, multi_target_rfc, le, arbitro)

          tabelas_padrao_3.append(tabela_padrao_3)
          
          # Padrão 4 - Últimas 5 partidas em casa e últimas 5 partidas fora de casa

          dataframe_time_casa_padrao_4 = pd.DataFrame()
          dataframe_time_fora_padrao_4 = pd.DataFrame()

          for time, grupo in grupos_casa:
              ultimas_partidas_time_padrao_4 = grupo.sort_values('data').tail(num_partidas_5_casa_fora)
              dataframe_time_casa_padrao_4 = pd.concat([dataframe_time_casa_padrao_4, ultimas_partidas_time_padrao_4])

          for time, grupo in grupos_fora:
              ultimas_partidas_time_padrao_4 = grupo.sort_values('data').tail(num_partidas_5_casa_fora)
              dataframe_time_fora_padrao_4 = pd.concat([dataframe_time_fora_padrao_4, ultimas_partidas_time_padrao_4])

          for index, row in partidas_anteriores.iterrows():
              partidas_casa_time_casa_df1_padrao_4 = dataframe_time_casa_padrao_4[(dataframe_time_casa_padrao_4['home_team_name'] == time_casa)]
              partidas_casa_time_fora_df1_padrao_4 = dataframe_time_fora_padrao_4[(dataframe_time_fora_padrao_4['away_team_name'] == time_fora)]

              ultimas_n_partidas_padrao_4 = pd.concat([partidas_casa_time_casa_df1_padrao_4, partidas_casa_time_fora_df1_padrao_4])

          tabela_padrao_4 = ultimas_partidas_casa_fora_previsao_tabela(partidas_casa_time_casa_df1_padrao_4, partidas_casa_time_fora_df1_padrao_4, ultimas_n_partidas_padrao_4, num_partidas_5_casa_fora, time_casa, time_fora, multi_target_rfc, le, arbitro)

          tabelas_padrao_4.append(tabela_padrao_4)

          # Padrão 5 - Últimas 10 partidas em casa e últimas 10 partidas fora de casa

          dataframe_time_casa_padrao_5 = pd.DataFrame()
          dataframe_time_fora_padrao_5 = pd.DataFrame()

          for time, grupo in grupos_casa:
            ultimas_partidas_time_padrao_5 = grupo.sort_values('data').tail(num_partidas_10_casa_fora)
            dataframe_time_casa_padrao_5 = pd.concat([dataframe_time_casa_padrao_5, ultimas_partidas_time_padrao_5])

          for time, grupo in grupos_fora:
            ultimas_partidas_time_padrao_5 = grupo.sort_values('data').tail(num_partidas_10_casa_fora)
            dataframe_time_fora_padrao_5 = pd.concat([dataframe_time_fora_padrao_5, ultimas_partidas_time_padrao_5])

          for index, row in partidas_anteriores.iterrows():
            partidas_casa_time_casa_df1_padrao_5 = dataframe_time_casa_padrao_5[(dataframe_time_casa_padrao_5['home_team_name'] == time_casa)]
            partidas_casa_time_fora_df1_padrao_5 = dataframe_time_fora_padrao_5[(dataframe_time_fora_padrao_5['away_team_name'] == time_fora)]

            ultimas_n_partidas_padrao_5 = pd.concat([partidas_casa_time_casa_df1_padrao_5,partidas_casa_time_fora_df1_padrao_5])

          tabela_padrao_5 = ultimas_partidas_casa_fora_previsao_tabela(partidas_casa_time_casa_df1_padrao_5,partidas_casa_time_fora_df1_padrao_5,ultimas_n_partidas_padrao_5,num_partidas_10_casa_fora,time_casa,time_fora, multi_target_rfc, le, arbitro)

          tabelas_padrao_5.append(tabela_padrao_5)

          # Padrão 6 - Últimas 3 partidas de modo geral

          # Criar um dataframe vazio para armazenar as últimas n partidas de cada time
          df_ultimas_partidas_padrao_6 = pd.DataFrame()

          # Iterar sobre cada time
          for time in partidas_anteriores['home_team_name'].unique():
              # Filtrar as partidas do time em questão
              partidas_time_padrao_6 = partidas_anteriores[(partidas_anteriores['home_team_name'] == time) | (partidas_anteriores['away_team_name'] == time)]  
              # Ordenar as partidas do time pela data em ordem decrescente e selecionar as últimas n partidas
              ultimas_partidas_padrao_6 = partidas_time_padrao_6.sort_values(by='data', ascending=False).head(num_partidas_3_geral) 
              # Adicionar as últimas n partidas do time ao dataframe final
              df_ultimas_partidas_padrao_6 = pd.concat([df_ultimas_partidas_padrao_6, ultimas_partidas_padrao_6])

          # Ajustar o índice do dataframe final
          df_ultimas_partidas_padrao_6.reset_index(drop=True, inplace=True)

          for index, row in df_ultimas_partidas_padrao_6.iterrows():
            partidas_casa_time_casa_df1_padrao_6 = df_ultimas_partidas_padrao_6[(df_ultimas_partidas_padrao_6['home_team_name'] == time_casa) | (df_ultimas_partidas_padrao_6['away_team_name'] == time_casa)]
            partidas_casa_time_fora_df1_padrao_6 = df_ultimas_partidas_padrao_6[(df_ultimas_partidas_padrao_6['home_team_name'] == time_fora) | (df_ultimas_partidas_padrao_6['away_team_name'] == time_fora)]

            partidas_casa_time_casa_df1_padrao_6 = partidas_casa_time_casa_df1_padrao_6.drop(partidas_casa_time_casa_df1_padrao_6.index[num_partidas_3_geral:])
            partidas_casa_time_fora_df1_padrao_6 = partidas_casa_time_fora_df1_padrao_6.drop(partidas_casa_time_fora_df1_padrao_6.index[:num_partidas_3_geral])

            ultimas_n_partidas_padrao_6 = pd.concat([partidas_casa_time_casa_df1_padrao_6,partidas_casa_time_fora_df1_padrao_6])

          tabela_padrao_6 = ultimas_partidas_gerais_previsao_tabela(partidas_casa_time_casa_df1_padrao_6,partidas_casa_time_fora_df1_padrao_6,ultimas_n_partidas_padrao_6,num_partidas_3_geral,time_casa,time_fora, multi_target_rfc, le, arbitro)

          tabelas_padrao_6.append(tabela_padrao_6)

          # Padrão 7 - Últimas 5 partidas de modo geral

          # Criar um dataframe vazio para armazenar as últimas n partidas de cada time
          df_ultimas_partidas_padrao_7 = pd.DataFrame()

          # Iterar sobre cada time
          for time in partidas_anteriores['home_team_name'].unique():
              # Filtrar as partidas do time em questão
              partidas_time_padrao_7 = partidas_anteriores[(partidas_anteriores['home_team_name'] == time) | (partidas_anteriores['away_team_name'] == time)]  
              # Ordenar as partidas do time pela data em ordem decrescente e selecionar as últimas n partidas
              ultimas_partidas_padrao_7 = partidas_time_padrao_7.sort_values(by='data', ascending=False).head(num_partidas_5_geral) 
              # Adicionar as últimas n partidas do time ao dataframe final
              df_ultimas_partidas_padrao_7 = pd.concat([df_ultimas_partidas_padrao_7, ultimas_partidas_padrao_7])

          # Ajustar o índice do dataframe final
          df_ultimas_partidas_padrao_7.reset_index(drop=True, inplace=True)

          for index, row in df_ultimas_partidas_padrao_7.iterrows():
            partidas_casa_time_casa_df1_padrao_7 = df_ultimas_partidas_padrao_7[(df_ultimas_partidas_padrao_7['home_team_name'] == time_casa) | (df_ultimas_partidas_padrao_7['away_team_name'] == time_casa)]
            partidas_casa_time_fora_df1_padrao_7 = df_ultimas_partidas_padrao_7[(df_ultimas_partidas_padrao_7['home_team_name'] == time_fora) | (df_ultimas_partidas_padrao_7['away_team_name'] == time_fora)]

            partidas_casa_time_casa_df1_padrao_7 = partidas_casa_time_casa_df1_padrao_7.drop(partidas_casa_time_casa_df1_padrao_7.index[num_partidas_5_geral:])
            partidas_casa_time_fora_df1_padrao_7 = partidas_casa_time_fora_df1_padrao_7.drop(partidas_casa_time_fora_df1_padrao_7.index[:num_partidas_5_geral])

            ultimas_n_partidas_padrao_7 = pd.concat([partidas_casa_time_casa_df1_padrao_7,partidas_casa_time_fora_df1_padrao_7])

          tabela_padrao_7 = ultimas_partidas_gerais_previsao_tabela(partidas_casa_time_casa_df1_padrao_7,partidas_casa_time_fora_df1_padrao_7,ultimas_n_partidas_padrao_7,num_partidas_5_geral,time_casa,time_fora, multi_target_rfc, le, arbitro)

          tabelas_padrao_7.append(tabela_padrao_7)
          
          # Padrão 8 - Últimas 10 partidas de modo geral

          # Criar um dataframe vazio para armazenar as últimas n partidas de cada time
          df_ultimas_partidas_padrao_8 = pd.DataFrame()

          # Iterar sobre cada time
          for time in partidas_anteriores['home_team_name'].unique():
              # Filtrar as partidas do time em questão
              partidas_time_padrao_8 = partidas_anteriores[(partidas_anteriores['home_team_name'] == time) | (partidas_anteriores['away_team_name'] == time)]  
              # Ordenar as partidas do time pela data em ordem decrescente e selecionar as últimas n partidas
              ultimas_partidas_padrao_8 = partidas_time_padrao_8.sort_values(by='data', ascending=False).head(num_partidas_10_geral) 
              # Adicionar as últimas n partidas do time ao dataframe final
              df_ultimas_partidas_padrao_8 = pd.concat([df_ultimas_partidas_padrao_8, ultimas_partidas_padrao_8])

          # Ajustar o índice do dataframe final
          df_ultimas_partidas_padrao_8.reset_index(drop=True, inplace=True)

          for index, row in df_ultimas_partidas_padrao_8.iterrows():
            partidas_casa_time_casa_df1_padrao_8 = df_ultimas_partidas_padrao_8[(df_ultimas_partidas_padrao_8['home_team_name'] == time_casa) | (df_ultimas_partidas_padrao_8['away_team_name'] == time_casa)]
            partidas_casa_time_fora_df1_padrao_8 = df_ultimas_partidas_padrao_8[(df_ultimas_partidas_padrao_8['home_team_name'] == time_fora) | (df_ultimas_partidas_padrao_8['away_team_name'] == time_fora)]

            partidas_casa_time_casa_df1_padrao_8 = partidas_casa_time_casa_df1_padrao_8.drop(partidas_casa_time_casa_df1_padrao_8.index[num_partidas_10_geral:])
            partidas_casa_time_fora_df1_padrao_8 = partidas_casa_time_fora_df1_padrao_8.drop(partidas_casa_time_fora_df1_padrao_8.index[:num_partidas_10_geral])

            ultimas_n_partidas_padrao_8 = pd.concat([partidas_casa_time_casa_df1_padrao_8,partidas_casa_time_fora_df1_padrao_8])

          tabela_padrao_8 = ultimas_partidas_gerais_previsao_tabela(partidas_casa_time_casa_df1_padrao_8,partidas_casa_time_fora_df1_padrao_8,ultimas_n_partidas_padrao_8,num_partidas_10_geral,time_casa,time_fora, multi_target_rfc, le, arbitro)

          tabelas_padrao_8.append(tabela_padrao_8)

      primeira_partida = []
      segunda_partida = []
      terceira_partida = []
      quarta_partida = []
      quinta_partida = []
      sexta_partida = []
      setima_partida = []
      oitava_partida = []
      nona_partida = []
      decima_partida = []

      for partida in tabelas_padrao_1, tabelas_padrao_2, tabelas_padrao_3, tabelas_padrao_4, tabelas_padrao_5, tabelas_padrao_6, tabelas_padrao_7, tabelas_padrao_8:
          primeira_partida.append(partida[0])
          segunda_partida.append(partida[1])
          terceira_partida.append(partida[2])
          quarta_partida.append(partida[3])
          quinta_partida.append(partida[4])
          sexta_partida.append(partida[5])
          setima_partida.append(partida[6])
          oitava_partida.append(partida[7])
          nona_partida.append(partida[8])
          decima_partida.append(partida[9])

      return primeira_partida, segunda_partida, terceira_partida, quarta_partida, quinta_partida, sexta_partida, setima_partida, oitava_partida, nona_partida, decima_partida

  primeira_partida, segunda_partida, terceira_partida, quarta_partida, quinta_partida, sexta_partida, setima_partida, oitava_partida, nona_partida, decima_partida = gerar_tabelas_padrao(lista_partidas)

  # Criação do dataframe
  colunas = ['Variáveis-alvo', '1','2','3','4','5','6','7','8']

  linhas = ['resultado_partida', 'resultado_intervalo', 'resultado_num_gols_over_under',
            'resultado_ambas_equipes_marcaram', 'resultado_num_cartoes_amarelos',
            'resultado_num_cartoes_vermelhos', 'resultado_num_cartoes_totais',
            'resultado_ambas_equipes_receberam_cartoes', 'resultado_cartoes_ambos_tempos',
            'resultado_num_escanteios', 'resultado_num_cartoes_primeiro', 'resultado_num_cartoes_segundo']

  # Criação dos dataframes
  df1 = pd.DataFrame(columns=colunas)
  df1['Variáveis-alvo'] = linhas

  df2 = pd.DataFrame(columns=colunas)
  df2['Variáveis-alvo'] = linhas

  df3 = pd.DataFrame(columns=colunas)
  df3['Variáveis-alvo'] = linhas

  df4 = pd.DataFrame(columns=colunas)
  df4['Variáveis-alvo'] = linhas

  df5 = pd.DataFrame(columns=colunas)
  df5['Variáveis-alvo'] = linhas

  df6 = pd.DataFrame(columns=colunas)
  df6['Variáveis-alvo'] = linhas

  df7 = pd.DataFrame(columns=colunas)
  df7['Variáveis-alvo'] = linhas

  df8 = pd.DataFrame(columns=colunas)
  df8['Variáveis-alvo'] = linhas

  df9 = pd.DataFrame(columns=colunas)
  df9['Variáveis-alvo'] = linhas

  df10 = pd.DataFrame(columns=colunas)
  df10['Variáveis-alvo'] = linhas

  # Preenchimento dos dataframes com os dados das partidas
  for i in range(len(primeira_partida)):
      partida_1 = primeira_partida[i]
      partida_2 = segunda_partida[i]
      partida_3 = terceira_partida[i]
      partida_4 = quarta_partida[i]
      partida_5 = quinta_partida[i]
      partida_6 = sexta_partida[i]
      partida_7 = setima_partida[i]
      partida_8 = oitava_partida[i]
      partida_9 = nona_partida[i]
      partida_10 = decima_partida[i]

      for j in range(len(partida_1[0])):
          df1.iloc[j, i+1] = partida_1[0][j]
          df2.iloc[j, i+1] = partida_2[0][j]
          df3.iloc[j, i+1] = partida_3[0][j]
          df4.iloc[j, i+1] = partida_4[0][j]
          df5.iloc[j, i+1] = partida_5[0][j]
          df6.iloc[j, i+1] = partida_6[0][j]
          df7.iloc[j, i+1] = partida_7[0][j]
          df8.iloc[j, i+1] = partida_8[0][j]
          df9.iloc[j, i+1] = partida_9[0][j]
          df10.iloc[j, i+1] = partida_10[0][j]   

  df1_dict = df1.to_dict(orient='list')
  df2_dict = df2.to_dict(orient='list')
  df3_dict = df3.to_dict(orient='list')
  df4_dict = df4.to_dict(orient='list')
  df5_dict = df5.to_dict(orient='list')
  df6_dict = df6.to_dict(orient='list')
  df7_dict = df7.to_dict(orient='list')
  df8_dict = df8.to_dict(orient='list')
  df9_dict = df9.to_dict(orient='list')
  df10_dict = df10.to_dict(orient='list')

  resultados_reais = y_test
  resultados_reais_dict = resultados_reais.to_dict(orient='list')

  chaves_resultados = list(resultados_reais_dict.keys())

  dataframes = []

  for k, df_dict in enumerate([df1_dict, df2_dict, df3_dict, df4_dict, df5_dict, df6_dict, df7_dict, df8_dict, df9_dict, df10_dict]):
      partida = [k+1] * 12
      variaveis_alvo = df_dict['Variáveis-alvo']
      padroes = []
      acuracia = []

      for i in range(12):
          chave_resultados = chaves_resultados[i]
          resultados = resultados_reais_dict[chave_resultados]
          valor_comparacao = resultados[k]
          padroes_iguais = []

          for j in range(1, 9):
              valores_df = df_dict[str(j)]

              if valores_df[i] == valor_comparacao:
                  padroes_iguais.append(j)

          padroes.append(f"{padroes_iguais} ({len(padroes_iguais)})")
          acuracia.append(len(padroes_iguais) / 8 * 100)

      data = {'Partida': partida, 'Variáveis-alvo': variaveis_alvo, 'Padrões': padroes, 'Acurácia': acuracia}
      df = pd.DataFrame(data)
      dataframes.append(df)

      # Adicionar uma linha com "-" após cada bloco de 12 linhas, exceto na última iteração
      if k < len([df1_dict, df2_dict, df3_dict, df4_dict, df5_dict, df6_dict, df7_dict, df8_dict, df9_dict, df10_dict]) - 1:
          linha_separadora = pd.DataFrame({'Partida': ['---'], 'Variáveis-alvo': ['---'], 'Padrões': ['---'], 'Acurácia': ['---']})
          dataframes.append(linha_separadora)

  df_final = pd.concat(dataframes, ignore_index=True)

  # Define um estilo para a tabela usando os seletores e propriedades do CSS
  df_final = (df_final.style
        .set_table_styles([{
            'selector': 'caption', # Seletor CSS para o título da tabela
            'props': [
                ('color', '#FFFFFF'),
                ('font-size', '18px'),
                ('font-style', 'normal'),
                ('font-weight', 'bold'),
                ('text-align', 'center'),
                ('background-color', '#126e51'),
                ('border', '1px solid gray')
            ]
        },
        {
            'selector': 'th', # Seletor CSS para as células do cabeçalho
            'props': [
                ('background-color', '#126e51'),
                ('color', 'black'),
                ('font-size', '15px'),
                ('font-weight', 'bold'),
                ('text-align', 'center'),
                ('border', '1px solid gray'),
                ('white-space', 'pre-wrap')
            ]
        },
        {
            'selector': 'td', # Seletor CSS para as células de dados
            'props': [
                ('background-color', '#283734'),
                ('color', 'white'),
                ('font-size', '15px'),
                ('font-weight', 'normal'),
                ('text-align', 'center'),
                ('border', '1px solid gray'),
                ('white-space', 'pre-wrap')
            ]
        },
        ])
        
    )

  return (df_final)

# Interação com o usuário

def main():
    st.set_page_config(page_title="InPES bet", page_icon=":soccer:")

    st.title('**Previsão de resultados de futebol**')

    st.subheader('**Selecione o arquivo csv com os dados das partidas de futebol**')

    arquivo = st.file_uploader("", type="csv")

    if arquivo is not None:
        partidas_df = pd.read_csv(arquivo)

        partidas_df['data'] =  pd.to_datetime(partidas_df['date_GMT'])

        # salvando os nomes dos times da casa em uma lista para futura verificação
        times_da_casa = sorted(partidas_df['home_team_name'].unique())
        # salvando os nomes dos times de fora em uma lista para futura verificação
        times_de_fora = sorted(partidas_df['away_team_name'].unique())
        # salvando os nomes dos árbitros em uma lista para futura verificação
        arbitros = sorted(partidas_df['referee'].unique())

        num_partidas = 0
        time_casa_widget = st.selectbox('**Time da casa:**', options=times_da_casa)
        time_fora_widget = st.selectbox('**Time de fora:**', options=times_de_fora)
        arbitro_widget = st.selectbox('**Árbitro:**', options=arbitros)
        data_widget = st.date_input('**Data da partida:**')

        # Define as opções do multiselect
        opcoes = ['Padrão 1 - Confrontos diretos',
                  'Padrão 2 - Histórico do campeonato',
                  'Padrão 3 - Últimas 3 partidas em casa e últimas 3 partidas fora',
                  'Padrão 4 - Últimas 5 partidas em casa e últimas 5 partidas fora',
                  'Padrão 5 - Últimas 10 partidas em casa e últimas 10 partidas fora',
                  'Padrão 6 - Últimas 3 partidas de modo geral',
                  'Padrão 7 - Últimas 5 partidas de modo geral',
                  'Padrão 8 - Últimas 10 partidas de modo geral']

        # Define a variável do checkbox
        considerar_todos = st.checkbox('**Considerar todos os padrões**')

        # Verifica se o checkbox está marcado
        if considerar_todos:
            # Define a lista de padrões como sendo todos
            padroes_selecionados = opcoes
        else:
            # Define a lista de padrões com base no multiselect
            padroes_selecionados = st.multiselect('**Selecione os padrões de análise:**', options=opcoes)
 
        # Adicionando botões de ação
        if st.button('**Gerar previsões**'):
            if time_fora_widget == time_casa_widget:
                st.error('**O time visitante não pode ser o mesmo que o time mandante!**')
            else:
                try:
                    # Converte a data para o formato desejado
                    data_da_partida = data_widget.strftime("%Y-%m-%d")

                    # gerando um dataframe com todas as partidas antes da data passada
                    partidas_anteriores = partidas_df[partidas_df['data'] < data_da_partida]

                    # treinando o modelo
                    multi_target_rfc, le, y_test, y_pred, _ = modelo_ml(partidas_df, data_da_partida)

                    # avaliando o modelo
                    acuracia = avaliacao_modelo(y_test, y_pred)

                    num_partidas = 5

                    df_concatenado_time_casa, df_concatenado_time_fora, df_resultados_confrontos_diretos, df_info_confrontos_diretos, _, _, total_partidas = tabela_resultados_medias(partidas_anteriores, time_casa_widget, time_fora_widget, multi_target_rfc, le, num_partidas, arbitro_widget)
                    if total_partidas != 0:
                      if considerar_todos == True:
                        tabela, legenda = gerar_tabela(time_casa_widget, time_fora_widget, arbitro_widget, multi_target_rfc, le, partidas_anteriores, acuracia)
                        df_tabela, df_legenda, df_casa, df_fora, df_res, df_inf = estilizar_df(df_concatenado_time_casa, df_concatenado_time_fora, df_resultados_confrontos_diretos, df_info_confrontos_diretos, time_casa_widget, time_fora_widget, tabela, legenda)
                        df_final = padroes_assertivos(partidas_df, data_da_partida, partidas_anteriores, multi_target_rfc, le, y_test)
                        st.header('**Previsões para a partida**')
                        st.subheader(f"{time_casa_widget} x {time_fora_widget}")
                        st.write(f'**Árbitro: {arbitro_widget}**')
                        st.write(f'**Data da partida: {data_da_partida}**')

                        st.table(df_tabela)
                        st.write('**Legenda dos Padrões**')
                        st.table(df_legenda)
                        st.write('**Últimos resultados do {}**'.format(time_casa_widget))
                        st.table(df_casa)
                        st.write('**Últimos resultados do {}**'.format(time_fora_widget))
                        st.table(df_fora)
                        st.write('**Confrontos diretos entre {} e {}**'.format(time_casa_widget, time_fora_widget))
                        st.table(df_res)
                        st.write('**Informações dos confrontos diretos entre {} e {}**'.format(time_casa_widget, time_fora_widget))
                        st.table(df_inf)
                        st.write('**Padrões assertivos**')
                        st.table(df_final)
                      else:
                        df = padroes_usuario(time_casa_widget, time_fora_widget, arbitro_widget, multi_target_rfc, le, partidas_anteriores, acuracia, padroes_selecionados)
                        tabela, legenda = gerar_tabela(time_casa_widget, time_fora_widget, arbitro_widget, multi_target_rfc, le, partidas_anteriores, acuracia)
                        df_tabela, df_legenda, df_casa, df_fora, df_res, df_inf = estilizar_df(df_concatenado_time_casa, df_concatenado_time_fora, df_resultados_confrontos_diretos, df_info_confrontos_diretos, time_casa_widget, time_fora_widget, tabela, legenda)
                        st.header('**Previsões para a partida**')
                        st.subheader(f"{time_casa_widget} x {time_fora_widget}")
                        st.write(f'**Árbitro: {arbitro_widget}**')
                        st.write(f'**Data da partida: {data_widget}**')
                        
                        st.table(df)
                        st.write('Últimos resultados do {}'.format(time_casa_widget))
                        st.write('**Últimos resultados do {}**'.format(time_casa_widget))
                        st.table(df_casa)
                        st.write('**Últimos resultados do {}**'.format(time_fora_widget))
                        st.table(df_fora)
                        st.write('**Confrontos diretos entre {} e {}**'.format(time_casa_widget, time_fora_widget))
                        st.table(df_res)
                        st.write('**Informações dos confrontos diretos entre {} e {}**'.format(time_casa_widget, time_fora_widget))
                        st.table(df_inf)

                except ValueError:
                    st.error('**Data inválida. Por favor, selecione outra data.**')

if __name__ == '__main__':
    main()