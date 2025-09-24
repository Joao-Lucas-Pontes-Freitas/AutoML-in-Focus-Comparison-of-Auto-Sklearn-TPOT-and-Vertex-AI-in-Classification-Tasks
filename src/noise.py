import math

import numpy as np
import pandas as pd
from pandas.api.types import is_categorical_dtype, is_float_dtype, is_integer_dtype


# ============================================================
# 1) LABEL NOISE (flip de rótulos estratificado no TREINO)
# ============================================================
def label_noise(
    y_train: pd.DataFrame, p: float = 0.10, random_state: int = 42
) -> pd.DataFrame:
    """
    Aplica ruído de rótulo (flip) estratificado no conjunto de TREINO.
    - Seleciona ceil(p * n_da_classe) instâncias por classe.
    - Troca o rótulo dessas instâncias por outra classe escolhida uniformemente ao acaso.
    - Imprime metadados de auditoria (quantos flips por classe).

    Parâmetros:
        y_train: DataFrame com uma única coluna de rótulos.
        p: Fração por classe a ser corrompida (default=0.10).
        random_state: Semente para reprodutibilidade (default=42).

    Retorna:
        y_train com rótulos flipados conforme especificação.
    """
    if not isinstance(y_train, pd.DataFrame) or y_train.shape[1] != 1:
        raise ValueError("y_train deve ser um DataFrame com exatamente 1 coluna.")

    rng = np.random.default_rng(random_state)
    nome_coluna = y_train.columns[0]
    y_original = y_train[nome_coluna]
    y_novo = y_original.copy()

    classes = pd.unique(y_original)
    contagem_flips = {}

    for cls in classes:
        index = y_original.index[y_original == cls]
        n_flip = math.ceil(p * len(index))
        if n_flip == 0:
            contagem_flips[cls] = 0
            continue

        index_escolhidos = rng.choice(index.to_numpy(), size=n_flip, replace=False)
        contagem_flips[cls] = len(index_escolhidos)

        # Para cada índice escolhido, escolher novo rótulo != cls
        outras = [c for c in classes if c != cls]
        for i in index_escolhidos:
            y_novo.at[i] = rng.choice(outras)

    # Auditoria
    total_flips = sum(contagem_flips.values())
    print("=== AUDITORIA: label_noise ===")
    print(f"p (fração por classe): {p}, random_state: {random_state}")
    print("Flips por classe:")
    for cls in classes:
        print(f"  - {cls}: {contagem_flips.get(cls, 0)}")
    print(f"Total de rótulos alterados: {total_flips}")

    # Retornar no mesmo formato (DataFrame)
    return pd.DataFrame({nome_coluna: y_novo}, index=y_train.index)


# ============================================================
# 2) DATA NOISE (IMAGENS) - SALT & PEPPER estratificado
# ============================================================
def data_noise_images(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    dataset_name: str,
    p: float = 0.10,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Aplica ruído SALT & PEPPER em dados de imagem (MNIST ou DIGITS) no TREINO.
    - Estratificado por classe: seleciona ceil(p * n_instâncias_da_classe).
    - Para cada instância selecionada, escolhe ceil(p * n_colunas) pixels (colunas) DIFERENTES por instância.
    - Define metade desses pixels como 0 (pepper) e metade como max (salt).
      * MNIST: max = 255
      * DIGITS: max = 16
    - Retorna X_train modificado (mesmo formato) e imprime metadados.

    Parâmetros:
        X_train: DataFrame (linhas=instâncias, colunas=pixels).
        y_train: DataFrame 1 coluna com rótulos (para estratificação).
        dataset_name: 'mnist_784' ou 'digits'.
        p: Fração de instâncias por classe e fração de colunas por instância (default=0.10).
        random_state: Semente para reprodutibilidade (default=42).

    Retorna:
        X_train com ruído adicionado.
    """
    if dataset_name.lower() not in {"mnist_784", "digits"}:
        raise ValueError("dataset_name deve ser 'mnist_784' ou 'digits'.")

    if not isinstance(y_train, pd.DataFrame) or y_train.shape[1] != 1:
        raise ValueError("y_train deve ser um DataFrame com exatamente 1 coluna.")

    rng = np.random.default_rng(random_state)
    X_novo = X_train.copy()
    y_series = y_train.iloc[:, 0]

    n_cols = X_train.shape[1]
    n_cols_ruidosas = math.ceil(p * n_cols)

    # Valor máximo de pixel conforme dataset
    valor_maximo = 255 if dataset_name.lower() == "mnist_784" else 16
    valor_minimo = 0

    contagem_por_classe = {}
    classes = pd.unique(y_series)

    for cls in classes:
        index = y_series.index[y_series == cls]
        n_inst = len(index)
        n_ruidosas = math.ceil(p * n_inst)
        if n_ruidosas == 0:
            contagem_por_classe[cls] = 0
            continue

        index_escolhidos = rng.choice(index.to_numpy(), size=n_ruidosas, replace=False)
        contagem_por_classe[cls] = len(index_escolhidos)

        for i in index_escolhidos:
            # Sorteio de colunas (pixels) específicos desta instância
            colunas_escolhidas = rng.choice(
                X_novo.columns.to_numpy(), size=n_cols_ruidosas, replace=False
            )

            # Metade pepper (0), metade salt (max). Se ímpar, extra vai para salt.
            rng.shuffle(colunas_escolhidas)
            metade = len(colunas_escolhidas) // 2
            col_pepper = colunas_escolhidas[:metade]
            col_salt = colunas_escolhidas[metade:]

            # Atribuição
            X_novo.loc[i, col_pepper] = valor_minimo
            X_novo.loc[i, col_salt] = valor_maximo

    # Auditoria
    print("=== AUDITORIA: data_noise_images (salt & pepper) ===")
    print(f"Dataset: {dataset_name}, p: {p}, random_state: {random_state}")
    print(
        f"Nº de colunas (pixels) totais: {n_cols}, nº de colunas ruidosas por instância: {n_cols_ruidosas}"
    )
    print("Instâncias alteradas por classe:")
    for cls in classes:
        print(f"  - {cls}: {contagem_por_classe.get(cls, 0)}")
    print("Proporção salt/pepper por instância: 50/50 (extra para 'salt' quando ímpar)")

    return X_novo


# ============================================================
# 3) DATA NOISE (TABULAR) - numérico (gaussiano) + categórico (troca)
#     - 10% instâncias por classe
#     - 10% colunas numéricas por instância
#     - 10% colunas categóricas por instância
# ============================================================
def data_noise_tabular(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    numerical_columns: list,
    categorical_columns: list,
    p: float = 0.10,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Aplica ruído em dados TABULARES no TREINO, de forma estratificada:
    - Seleciona ceil(p * n_instâncias_da_classe) por classe.
    - Para CADA instância selecionada:
        * Sorteia ceil(p * nº de colunas numéricas) e aplica ruído Gaussiano aditivo:
          - ruído ~ N(0, std_col), usando std calculado na coluna inteira antes do ruído;
          - clip dos valores em [min_col, max_col] (calculados antes do ruído);
          - se coluna é inteira, arredonda e preserva dtype original.
        * Sorteia ceil(p * nº de colunas categóricas) e troca o valor por outro válido (≠ valor atual).

    Observação:
        - Colunas sorteadas são por instância (não fixas globalmente).
        - NaN em categóricas permanece inalterado.

    Parâmetros:
        X_train: dados tabulares de treino.
        y_train: DataFrame 1 coluna com rótulos (p/ estratificação).
        numerical_columns: nomes de colunas numéricas.
        categorical_columns: nomes de colunas categóricas.
        p: fração (default=0.10).
        random_state: semente (default=42).

    Retorna:
        X_train modificado.
    """
    if not isinstance(y_train, pd.DataFrame) or y_train.shape[1] != 1:
        raise ValueError("y_train deve ser um DataFrame com exatamente 1 coluna.")

    rng = np.random.default_rng(random_state)
    X_novo = X_train.copy()
    y_series = y_train.iloc[:, 0]

    # Preparar estatísticas das colunas numéricas (antes do ruído)
    # std pode ser 0 (coluna constante); nesse caso não alteramos essa coluna quando sorteada.
    medias = {}
    desvios = {}
    mins = {}
    maxs = {}
    for col in numerical_columns:
        col_series = X_novo[col]
        medias[col] = col_series.mean()
        desvios[col] = col_series.std(ddof=0)  # ddof=0 (populacional) p/ estabilidade
        mins[col] = col_series.min()
        maxs[col] = col_series.max()

    # Preparar domínios das categóricas (valores possíveis, excluindo NaN)
    dominios_categoricos = {}
    for col in categorical_columns:
        valores = pd.unique(X_novo[col].dropna())
        dominios_categoricos[col] = list(valores)

    # Tamanhos a sortear por instância
    n_colunas_numericas = (
        math.ceil(p * len(numerical_columns)) if len(numerical_columns) > 0 else 0
    )
    n_colunas_categoricas = (
        math.ceil(p * len(categorical_columns)) if len(categorical_columns) > 0 else 0
    )

    contagem_instancias = {}
    classes = pd.unique(y_series)

    for cls in classes:
        index = y_series.index[y_series == cls]
        n_inst = len(index)
        n_ruidosas = math.ceil(p * n_inst)
        contagem_instancias[cls] = n_ruidosas

        if n_ruidosas == 0:
            continue

        index_escolhidos = rng.choice(index.to_numpy(), size=n_ruidosas, replace=False)

        for i in index_escolhidos:
            # --- NUMÉRICAS: escolher colunas desta instância ---
            if n_colunas_numericas > 0:
                cols_num_escolhidas = rng.choice(
                    numerical_columns,
                    size=min(n_colunas_numericas, len(numerical_columns)),
                    replace=False,
                )
                for col in cols_num_escolhidas:
                    std_col = desvios[col]
                    if std_col is None or np.isnan(std_col) or std_col == 0:
                        # Coluna constante (ou std inválido): não altera
                        continue

                    valor_atual = X_novo.at[i, col]
                    ruido = rng.normal(0.0, std_col)
                    novo_valor = valor_atual + ruido

                    # Clipar para [min, max] observados
                    novo_valor = max(mins[col], min(maxs[col], novo_valor))

                    # Se a coluna for inteira, arredondar e preservar dtype
                    if is_integer_dtype(X_novo[col].dtype):
                        # lidar com nulos em inteiros (pandas 'Int64') — aqui não geramos NaN
                        novo_valor = int(round(novo_valor))

                    X_novo.at[i, col] = novo_valor

            # --- CATEGÓRICAS: escolher colunas desta instância ---
            if n_colunas_categoricas > 0:
                cols_cat_escolhidas = rng.choice(
                    categorical_columns,
                    size=min(n_colunas_categoricas, len(categorical_columns)),
                    replace=False,
                )
                for col in cols_cat_escolhidas:
                    valor_atual = X_novo.at[i, col]
                    if pd.isna(valor_atual):
                        # Mantém NaN
                        continue

                    dominio = dominios_categoricos.get(col, [])
                    # Se não houver alternativa diferente, não altera
                    alternativas = [v for v in dominio if v != valor_atual]
                    if len(alternativas) == 0:
                        continue

                    X_novo.at[i, col] = rng.choice(alternativas)

    # Auditoria
    print("=== AUDITORIA: data_noise_tabular ===")
    print(f"p: {p}, random_state: {random_state}")
    print(
        f"Num colunas numéricas: {len(numerical_columns)} | sorteadas por instância: {n_colunas_numericas}"
    )
    print(
        f"Num colunas categóricas: {len(categorical_columns)} | sorteadas por instância: {n_colunas_categoricas}"
    )
    print("Instâncias alteradas por classe:")
    for cls in classes:
        print(f"  - {cls}: {contagem_instancias.get(cls, 0)}")

    return X_novo


# ============================================================
# 4) DATA NOISE (TEXTO) - remoção de palavras
#     - 10% instâncias por classe
#     - 20% de chance de remoção por palavra
# ============================================================
def data_noise_text(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    p: float = 0.10,
    random_state: int = 42,
    col_name: str = None,
) -> pd.DataFrame:
    """
    Aplica ruído textual por REMOÇÃO DE PALAVRAS no TREINO, estratificado:
    - Seleciona ceil(p * n_instâncias_da_classe) por classe.
    - Para cada texto selecionado:
        * Tokeniza por espaço (split simples).
        * Remove cada palavra com probabilidade 0.20.
        * Garante pelo menos 1 palavra (se esvaziar, mantém a primeira).
        * Tenta garantir que a linha selecionada seja de fato alterada
            (se nenhuma palavra foi removida e houver >= 2 tokens, remove 1 token aleatório).
    - Preserva caixa e pontuação (apenas remove tokens inteiros).
    - Usa a coluna de texto passada como parâmetro.

    Parâmetros:
        X_train (pd.DataFrame): DataFrame que tem uma coluna de texto.
        y_train (pd.DataFrame): DataFrame 1 coluna com rótulos (p/ estratificação).
        p (float): fração de instâncias por classe a serem afetadas (default=0.10).
        random_state (int): semente.

    Retorna:
        pd.DataFrame: X_train com textos modificados.
    """

    if not isinstance(y_train, pd.DataFrame) or y_train.shape[1] != 1:
        raise ValueError("y_train deve ser um DataFrame com exatamente 1 coluna.")
    
    if col_name is None or col_name not in X_train.columns:
        raise ValueError("col_name deve ser o nome de uma coluna existente em X_train.")

    rng = np.random.default_rng(random_state)
    X_novo = X_train.copy()
    y_series = y_train.iloc[:, 0]

    prob_remocao = 0.20
    contagem_por_classe = {}
    classes = pd.unique(y_series)

    for cls in classes:
        idx_cls = y_series.index[y_series == cls]
        n_inst = len(idx_cls)
        n_ruidosas = math.ceil(p * n_inst)
        contagem_por_classe[cls] = n_ruidosas

        if n_ruidosas == 0:
            continue

        idx_instancias = rng.choice(idx_cls.to_numpy(), size=n_ruidosas, replace=False)

        for i in idx_instancias:
            texto = X_novo.at[i, col_name]

            # Se não for string (ex: NaN), não altera
            if not isinstance(texto, str):
                continue

            tokens = texto.split()
            if len(tokens) == 0:
                continue

            # Remoção independente com probabilidade 0.20 por token
            novos_tokens = []
            for tk in tokens:
                if rng.random() < prob_remocao:
                    continue
                novos_tokens.append(tk)

            # >>> GARANTIR MUDANÇA QUANDO POSSÍVEL:
            # Se nada foi removido e há pelo menos 2 tokens, remover 1 token aleatório
            if len(novos_tokens) == len(tokens) and len(tokens) >= 2:
                idx_drop = rng.integers(0, len(tokens))
                novos_tokens = tokens[:idx_drop] + tokens[idx_drop + 1 :]

            # Garante ao menos 1 token
            if len(novos_tokens) == 0:
                novos_tokens = [tokens[0]]

            X_novo.at[i, col_name] = " ".join(novos_tokens)

    # Auditoria
    print("=== AUDITORIA: data_noise_text ===")
    print(
        f"p: {p}, random_state: {random_state}, prob_remocao_por_palavra: {prob_remocao}"
    )
    print("Instâncias alteradas por classe:")
    for cls in classes:
        print(f"  - {cls}: {contagem_por_classe.get(cls, 0)}")

    return X_novo


import keyword
import re
import unicodedata


def normalize_cols(nomes_colunas):
    vistos = {}  # base -> contador (para sufixos _1, _2, ...)
    saida = []

    for nome in nomes_colunas:
        s = str(nome).strip().lower()

        # normaliza acentos e remove marcas diacríticas
        s = unicodedata.normalize("NFKD", s)
        s = "".join(ch for ch in s if not unicodedata.combining(ch))

        # padroniza aspas curvas e troca qualquer sequência não [a-z0-9] por "_"
        s = s.replace("'", "'")
        s = re.sub(r"[^a-z0-9]+", "_", s)

        # colapsa múltiplos "_" e remove "_" nos extremos
        s = re.sub(r"_+", "_", s).strip("_")

        # vazio -> "col"
        if not s:
            s = "col"

        # evitar iniciar por dígito
        if s[0].isdigit():
            s = f"col_{s}"

        # evitar palavras reservadas do Python
        if keyword.iskeyword(s):
            s = f"{s}_"

        # garantir unicidade (_1, _2, ...) apenas se já existir
        base = s
        k = vistos.get(base, 0)
        if k > 0:
            s = f"{base}_{k}"
        vistos[base] = k + 1

        saida.append(s)

    return saida
