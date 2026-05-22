import pandas as pd
import numpy as np
import random

# Reproducibility

np.random.seed(42)

# Number of samples

n_samples = 3000

data = []

traffic_sources = [
    "Organic",
    "Ads",
    "Social Media",
    "Email"
]

devices = [
    "Mobile",
    "Desktop",
    "Tablet"
]

# Generate dataset

for _ in range(n_samples):

    # Decide if user converts
    converted = np.random.choice([0, 1],p=[0.5, 0.5])

    # USERS WHO CONVERT

    if converted == 1:

        age = np.random.randint(25, 55)

        previous_purchases = np.random.poisson(5)

        traffic_source = np.random.choice(["Email", "Organic", "Social Media"],p=[0.4, 0.4, 0.2])

        time_on_site = round(np.random.normal(12, 3), 2)

        cart_value = round(np.random.normal(180, 50), 2)

        viewed_reviews = np.random.choice([0, 1],p=[0.2, 0.8])

        pages_visited = np.random.randint(8, 20)

        device = np.random.choice(["Desktop", "Mobile", "Tablet"],p=[0.5, 0.3, 0.2])

        added_to_cart = np.random.choice([0, 1],p=[0.25, 0.75])

    # USERS WHO DO NOT CONVERT

    else:

        age = np.random.randint(18, 65)

        previous_purchases = np.random.poisson(1)

        traffic_source = np.random.choice(["Ads", "Social Media", "Organic"],p=[0.5, 0.3, 0.2])

        time_on_site = round(np.random.normal(6, 3), 2)

        cart_value = round(np.random.normal(80, 45), 2)

        viewed_reviews = np.random.choice([0, 1],p=[0.7, 0.3])

        pages_visited = np.random.randint(1, 10)

        device = np.random.choice(["Mobile", "Desktop", "Tablet"],p=[0.6, 0.2, 0.2])

        added_to_cart = np.random.choice([0, 1],p=[0.6, 0.4])

    # Avoid negative values
    time_on_site = max(1, time_on_site)
    cart_value = max(5, cart_value)

    # Realistic noise to balance results out

    if np.random.rand() < 0.15:
        converted = 1 - converted

    # Store row


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

# Create DataFrame

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

df.to_csv("ecommerce_data.csv", index=False)

print("\nCLASS DISTRIBUTION")
print(df["Converted"].value_counts())

print("\nDataset generated successfully!")
print(df.head())