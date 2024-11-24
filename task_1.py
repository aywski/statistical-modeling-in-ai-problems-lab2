import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button


def simulate_poisson_process(lambda_rate, T):
    """
    Симулює Пуассонівський процес.

    :param lambda_rate: Інтенсивність процесу
    :param T: Кінцевий час симуляції
    :return: Масив часів подій
    """
    t = 0
    events = []
    while t < T:
        t += np.random.exponential(1 / lambda_rate)
        if t < T:
            events.append(t)
    return np.array(events)


def plot_process_realization(events, T):
    """
    Будує графік реалізації процесу.

    :param events: Масив часів подій
    :param T: Кінцевий час симуляції
    """
    plt.figure(figsize=(12, 6))
    plt.step(events, range(1, len(events) + 1), where='post')
    plt.xlabel('Час')
    plt.ylabel('Кількість подій')
    plt.title('Реалізація Пуассонівського процесу')
    plt.grid(True)
    plt.show()


def plot_3d_process_realization(events, T, num_realizations=10):
    """
    Будує 3D візуалізацію декількох реалізацій процесу.

    :param events: Масив часів подій для першої реалізації
    :param T: Кінцевий час симуляції
    :param num_realizations: Кількість реалізацій для візуалізації
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(num_realizations):
        events = simulate_poisson_process(lambda_rate, T)
        x = events
        y = np.ones_like(events) * i
        z = np.arange(1, len(events) + 1)
        ax.plot(x, y, z, marker='o')

    ax.set_xlabel('Час')
    ax.set_ylabel('Номер реалізації')
    ax.set_zlabel('Кількість подій')
    ax.set_title('3D візуалізація реалізацій Пуассонівського процесу')
    plt.show()


def plot_real_time_process(lambda_rate, T, update_interval=100):
    """
    Будує графік процесу в реальному часі.

    :param lambda_rate: Інтенсивність процесу
    :param T: Кінцевий час симуляції
    :param update_interval: Інтервал оновлення в мілісекундах
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    line, = ax.plot([], [], 'b-')
    ax.set_xlim(0, T)
    ax.set_ylim(0, lambda_rate * T * 1.5)
    ax.set_xlabel('Час')
    ax.set_ylabel('Кількість подій')
    ax.set_title('Пуассонівський процес в реальному часі')
    ax.grid(True)

    events = []

    def init():
        line.set_data([], [])
        return line,

    def animate(frame):
        t = frame * T / 1000
        if not events or events[-1] <= t:
            new_event = simulate_poisson_process(lambda_rate, t)
            events.extend(new_event[new_event > (events[-1] if events else 0)])
        x = events
        y = range(1, len(events) + 1)
        line.set_data(x, y)
        return line,

    ani = animation.FuncAnimation(fig, animate, frames=1000, init_func=init, blit=True, interval=update_interval)
    plt.show()


def interactive_rate_analysis(T, max_rate=10):
    """
    Інтерактивний аналіз впливу інтенсивності на процес.

    :param T: Кінцевий час симуляції
    :param max_rate: Максимальна інтенсивність для слайдера
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.subplots_adjust(bottom=0.25)

    events = simulate_poisson_process(1, T)
    line, = ax.step(events, range(1, len(events) + 1), where='post')

    ax.set_xlim(0, T)
    ax.set_ylim(0, max_rate * T)
    ax.set_xlabel('Час')
    ax.set_ylabel('Кількість подій')
    ax.set_title('Вплив інтенсивності на Пуассонівський процес')
    ax.grid(True)

    axcolor = 'lightgoldenrodyellow'
    ax_slider = plt.axes([0.1, 0.1, 0.8, 0.03], facecolor=axcolor)
    slider = Slider(ax_slider, 'Інтенсивність', 0.1, max_rate, valinit=1)

    def update(val):
        rate = slider.val
        events = simulate_poisson_process(rate, T)
        line.set_xdata(events)
        line.set_ydata(range(1, len(events) + 1))
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


def analyze_convergence(lambda_rate, T, num_simulations=1000, k=1):
    """
    Аналізує збіжність середнього числа подій до теоретичного значення.

    :param lambda_rate: Інтенсивність процесу
    :param T: Кінцевий час симуляції
    :param num_simulations: Кількість симуляцій
    :param k: Число k для аналізу перших подій
    """
    event_counts = [len(simulate_poisson_process(lambda_rate, T)) for _ in range(num_simulations)]
    cumulative_means = np.cumsum(event_counts) / np.arange(1, num_simulations + 1)

    plt.figure(figsize=(12, 6))
    plt.plot(range(1, num_simulations + 1), cumulative_means, label='Емпіричне середнє')
    plt.axhline(y=lambda_rate * T, color='r', linestyle='--', label='Теоретичне середнє')
    plt.xlabel('Кількість симуляцій')
    plt.ylabel('Середня кількість подій')
    plt.title('Збіжність середнього числа подій')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Аналіз перших k подій
    if k > 1:
        first_k_events = []
        for _ in range(num_simulations):
            events = simulate_poisson_process(lambda_rate, T)
            if len(events) >= k:
                first_k_events.append(events[:k])

        first_k_events = np.array(first_k_events)
        mean_first_k = np.mean(first_k_events, axis=0)

        plt.figure(figsize=(12, 6))
        for i in range(k):
            plt.plot(range(1, len(mean_first_k) + 1), mean_first_k, label=f'Середнє значення для перших {i+1} подій')
        plt.xlabel('Номер події')
        plt.ylabel('Середній час події')
        plt.title('Середнє значення для перших k подій')
        plt.legend()
        plt.grid(True)
        plt.show()


# Параметри симуляції
lambda_rate = 2  # Інтенсивність процесу
T = 10  # Кінцевий час симуляції

# Базова симуляція та візуалізація
events = simulate_poisson_process(lambda_rate, T)
plot_process_realization(events, T)

# 3D візуалізація
plot_3d_process_realization(events, T)

# Графік в реальному часі
plot_real_time_process(lambda_rate, T)

# Інтерактивний аналіз впливу інтенсивності
interactive_rate_analysis(T)

# Аналіз збіжності
analyze_convergence(lambda_rate, T, k=3)

print(f"Кількість змодельованих подій: {len(events)}")
print(f"Середній інтервал між подіями: {np.mean(np.diff(events)):.4f}")
print(f"Теоретичний середній інтервал: {1 / lambda_rate:.4f}")
