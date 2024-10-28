import numpy as np


def compute_mse(b, w, data):
    total_error = np.mean((data[:, 1] - (w * data[:, 0] + b)) ** 2)
    return total_error

def step_gradient(b, w, data, alpha):
    N = len(data)  # número de amostras no conjunto de dados
    x = data[:, 0]  # áreas do terreno
    y = data[:, 1]  # preços das fazendas
    
    # Calcular as previsões atuais
    y_pred = w * x + b
    
    # Gradientes em relação a b e w
    b_gradient = -(2 / N) * np.sum(y - y_pred)
    w_gradient = -(2 / N) * np.sum((y - y_pred) * x)
    
    # Atualizar b e w
    new_b = b - alpha * b_gradient
    new_w = w - alpha * w_gradient
    
    return new_b, new_w

def fit(data, b, w, alpha, num_iterations):
    b_history = [b]  # lista para armazenar o histórico de b
    w_history = [w]  # lista para armazenar o histórico de w
    
    for _ in range(num_iterations):
        b, w = step_gradient(b, w, data, alpha)
        b_history.append(b)
        w_history.append(w)
    
    return b_history, w_history
