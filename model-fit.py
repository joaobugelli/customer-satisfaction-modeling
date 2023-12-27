import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Substitua "seus_dados" pela sua variÃ¡vel com os dados
data = seus_dados  # seus dados aqui

# Lista de distribuiÃ§Ãµes para testar
distributions = [stats.beta, stats.expon, stats.gamma, stats.norm]

# Armazenar resultados
results = []

# Ajustar cada distribuiÃ§Ã£o aos dados e calcular o AIC
for distribution in distributions:
    # Ajustar a distribuiÃ§Ã£o aos dados
    params = distribution.fit(data)
    
    # Separar os parÃ¢metros do ajuste
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]
    
    # Calcular o log-likelihood
    log_likelihood = np.sum(distribution.logpdf(data, *arg, loc=loc, scale=scale))
    
    # Calcular o AIC
    aic = 2 * len(params) - 2 * log_likelihood
    
    # Guardar os resultados
    results.append((distribution.name, aic))

# Ordenar os resultados pelo AIC
results.sort(key=lambda x: x[1])

# Visualizar os dados e os melhores ajustes
plt.hist(data, bins=30, density=True, alpha=0.5, color='g', label='Data Histogram')

# Plotar as melhores distribuiÃ§Ãµes ajustadas
for i, (name, _) in enumerate(results[:2]):  # Mostrando os dois melhores
    distribution = getattr(stats, name)
    params = distribution.fit(data)
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]
    x = np.linspace(min(data), max(data), 100)
    y = distribution.pdf(x, *arg, loc=loc, scale=scale)
    plt.plot(x, y, label=f'{name} fit')

plt.legend()
plt.title('Data with Fitted Distributions')
plt.show()

# Exibir os resultados do AIC
for result in results:
    print(f"Distribution: {result[0]}, AIC: {result[1]}")