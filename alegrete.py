import numpy as np


def compute_mse(b, w, data):
    """
    Calcula o erro quadratico medio
    :param b: float - bias (intercepto da reta)
    :param w: float - peso (inclinacao da reta)
    :param data: np.array - matriz com o conjunto de dados, x na coluna 0 e y na coluna 1
    :return: float - o erro quadratico medio
    """
    total_error = 0
    for i in range(len(data)):
        x = data[i, 0]
        y = data[i, 1]
        total_error += (y - (w * x + b)) ** 2
    return total_error / len(data)


def step_gradient(b, w, data, alpha):
    """
    Executa uma atualização por descida do gradiente  e retorna os valores atualizados de b e w.
    :param b: float - bias (intercepto da reta)
    :param w: float - peso (inclinacao da reta)
    :param data: np.array - matriz com o conjunto de dados, x na coluna 0 e y na coluna 1
    :param alpha: float - taxa de aprendizado (a.k.a. tamanho do passo)
    :return: float,float - os novos valores de b e w, respectivamente
    """
    b_gradient = 0
    w_gradient = 0
    N = len(data)

    for i in range(N):
        x = data[i, 0]
        y = data[i, 1]
        b_gradient += -(2 / N) * (y - (w * x + b))
        w_gradient += -(2 / N) * x * (y - (w * x + b))

    new_b = b - (alpha * b_gradient)
    new_w = w - (alpha * w_gradient)
    
    return new_b, new_w


def fit(data, b, w, alpha, num_iterations):
    """
    Para cada época/iteração, executa uma atualização por descida de
    gradiente e registra os valores atualizados de b e w.
    Ao final, retorna duas listas, uma com os b e outra com os w
    obtidos ao longo da execução (o último valor das listas deve
    corresponder à última época/iteração).

    :param data: np.array - matriz com o conjunto de dados, x na coluna 0 e y na coluna 1
    :param b: float - bias (intercepto da reta)
    :param w: float - peso (inclinacao da reta)
    :param alpha: float - taxa de aprendizado (a.k.a. tamanho do passo)
    :param num_iterations: int - numero de épocas/iterações para executar a descida de gradiente
    :return: list,list - uma lista com os b e outra com os w obtidos ao longo da execução
    """
    b_values = [b]
    w_values = [w]

    for i in range(num_iterations):
        b, w = step_gradient(b, w, data, alpha)
        b_values.append(b)
        w_values.append(w)

    return b_values, w_values
