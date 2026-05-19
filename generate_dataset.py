import pandas as pd
import numpy as np
import random

# Reprodutibilidade
np.random.seed(42)

# Número de samples
n_samples = 2000

data = []

traffic_sources = ["Organic", "Ads", "Social Media", "Email"]
devices = ["Mobile", "Desktop", "Tablet"]

for _ in range(n_samples):

    age = np.random.randint(18, 65)

    previous_purchases = np.random.poisson(3)

    traffic_source = random.choice(traffic_sources)

    time_on_site = round(np.random.normal(8, 3), 2)
    time_on_site = max(1, time_on_site)

    cart_value = round(np.random.normal(120, 60), 2)
    cart_value = max(5, cart_value)

    viewed_reviews = np.random.choice([0, 1], p=[0.4, 0.6])

    pages_visited = np.random.randint(1, 20)

    device = random.choice(devices)

    added_to_cart = np.random.choice([0, 1], p=[0.3, 0.7])

    # -------------------------
    # Lógica de conversão
    # -------------------------

    conversion_score = 0

    # fatores positivos
    conversion_score += previous_purchases * 0.4
    conversion_score += time_on_site * 0.3
    conversion_score += cart_value * 0.02
    conversion_score += viewed_reviews * 1.5
    conversion_score += pages_visited * 0.2
    conversion_score += added_to_cart * 4

    # influência da origem do tráfego
    if traffic_source == "Email":
        conversion_score += 2

    elif traffic_source == "Organic":
        conversion_score += 1

    # influência do dispositivo
    if device == "Desktop":
        conversion_score += 1

    elif device == "Mobile":
        conversion_score -= 0.5

    # probabilidade final
    probability = 1 / (1 + np.exp(-0.1 * (conversion_score - 10)))

    converted = np.random.choice([0, 1], p=[1 - probability, probability])

    data.append([
        age,
        previous_purchases,
        traffic_source,
        time_on_site,
        cart_value,
        viewed_reviews,
        pages_visited,
        device,
        added_to_cart,
        converted
    ])

# Criar DataFrame
columns = [
    "Age",
    "PreviousPurchases",
    "TrafficSource",
    "TimeOnSite",
    "CartValue",
    "ViewedReviews",
    "PagesVisited",
    "Device",
    "AddedToCart",
    "Converted"
]

df = pd.DataFrame(data, columns=columns)

# Guardar CSV
df.to_csv("ecommerce_data.csv", index=False)

print("Dataset generated successfully!")
print(df.head())