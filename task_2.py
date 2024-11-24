import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def simulate_wiener_process(T, dt, num_paths):
    """
    Симулює вінерівський випадковий процес.
    
    Parameters:
        T (float): Кінцевий час симуляції
        dt (float): Часовий крок
        num_paths (int): Кількість реалізацій процесу
    
    Returns:
        tuple: (t, W) - масиви часу та значень процесу
    """
    num_steps = int(T / dt)
    dW = np.sqrt(dt) * np.random.normal(0, 1, (num_paths, num_steps))
    W = np.cumsum(dW, axis=1)
    W = np.hstack((np.zeros((num_paths, 1)), W))  # Додаємо початкову точку W(0) = 0
    t = np.linspace(0, T, num_steps + 1)
    return t, W

def compute_statistics(W):
    """
    Обчислює середнє значення та дисперсію процесу.
    
    Parameters:
        W (numpy.ndarray): Масив реалізацій процесу
    
    Returns:
        tuple: (mean_W, var_W) - масиви середніх значень та дисперсій
    """
    mean_W = np.mean(W, axis=0)
    var_W = np.var(W, axis=0)
    return mean_W, var_W

def plot_mean_variance(t, mean_W, var_W):
    """
    Візуалізує середнє значення та дисперсію процесу.
    
    Parameters:
        t (numpy.ndarray): Масив часових точок
        mean_W (numpy.ndarray): Масив середніх значень
        var_W (numpy.ndarray): Масив значень дисперсії
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Графік середнього значення
    color = 'tab:blue'
    ax1.set_xlabel('Час')
    ax1.set_ylabel('Середнє W(t)', color=color)
    ax1.plot(t, mean_W, color=color, label='Середнє')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')

    # Графік дисперсії на другій осі
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Дисперсія', color=color)
    ax2.plot(t, var_W, color=color, label='Дисперсія')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')

    plt.title('Середнє та дисперсія вінерівського процесу')
    plt.grid(True)
    plt.show()

def find_first_exit_times(W, level=1.0):
    """
    Знаходить час першого виходу процесу за заданий рівень.
    
    Parameters:
        W (numpy.ndarray): Масив реалізацій процесу
        level (float): Рівень виходу
    
    Returns:
        numpy.ndarray: Масив часів першого виходу
    """
    num_paths, num_steps = W.shape
    first_exit_times = np.full(num_paths, np.nan)

    for i in range(num_paths):
        exit_indices = np.where(np.abs(W[i, :]) >= level)[0]
        if exit_indices.size > 0:
            first_exit_times[i] = exit_indices[0]
    return first_exit_times

def plot_first_exit_distribution(first_exit_times, dt, level=1.0):
    """
    Візуалізує розподіл часу першого виходу.
    
    Parameters:
        first_exit_times (numpy.ndarray): Масив часів першого виходу
        dt (float): Часовий крок
        level (float): Рівень виходу
    """
    # Відфільтровуємо NaN значення і конвертуємо індекси в реальний час
    exit_times = first_exit_times[~np.isnan(first_exit_times)] * dt

    plt.figure(figsize=(12, 6))
    sns.histplot(exit_times, bins=30, kde=True, stat="density")
    plt.xlabel('Час першого виходу')
    plt.ylabel('Густина ймовірності')
    plt.title(f'Емпіричний розподіл часу першого виходу за рівень {level}')
    plt.grid(True)
    plt.show()
    
    # Додатково виводимо основні статистики
    print(f"Статистики часу першого виходу:")
    print(f"Середнє значення: {np.mean(exit_times):.4f}")
    print(f"Медіана: {np.median(exit_times):.4f}")
    print(f"Стандартне відхилення: {np.std(exit_times):.4f}")

def plot_sample_paths(t, W, num_paths_to_plot=5):
    """
    Візуалізує вибрані реалізації процесу.
    
    Parameters:
        t (numpy.ndarray): Масив часових точок
        W (numpy.ndarray): Масив реалізацій процесу
        num_paths_to_plot (int): Кількість реалізацій для відображення
    """
    plt.figure(figsize=(12, 6))
    for i in range(min(num_paths_to_plot, W.shape[0])):
        plt.plot(t, W[i, :], label=f'Реалізація {i+1}')
    
    plt.xlabel('Час')
    plt.ylabel('W(t)')
    plt.title('Вибрані реалізації вінерівського процесу')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # Параметри симуляції
    T = 1.0         # Кінцевий час
    dt = 0.001      # Часовий крок
    num_paths = 1000  # Кількість реалізацій
    exit_level = 1.0  # Рівень виходу для аналізу

    # Симуляція процесу
    t, W = simulate_wiener_process(T, dt, num_paths)
    
    # Обчислення та візуалізація статистик
    mean_W, var_W = compute_statistics(W)
    plot_mean_variance(t, mean_W, var_W)
    
    # Візуалізація вибраних реалізацій
    plot_sample_paths(t, W)
    
    # Аналіз часу першого виходу
    first_exit_times = find_first_exit_times(W, level=exit_level)
    plot_first_exit_distribution(first_exit_times, dt, level=exit_level)

if __name__ == "__main__":
    main()