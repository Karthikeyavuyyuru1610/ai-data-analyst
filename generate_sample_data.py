"""Generate deterministic sample sales data for testing."""
import csv
import os
import random

# Deterministic seed for reproducible data
random.seed(42)

PRODUCTS = {
    "Laptop Pro": {"category": "Electronics", "base_price": 1200},
    "Wireless Mouse": {"category": "Electronics", "base_price": 25},
    "Office Chair": {"category": "Furniture", "base_price": 350},
    "Standing Desk": {"category": "Furniture", "base_price": 600},
    "Notebook Set": {"category": "Stationery", "base_price": 15},
}

REGIONS = ["North", "South", "East", "West"]
SALESPERSONS = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"]
CUSTOMERS = [
    "Acme Corp", "Globex Inc", "Initech", "Umbrella Co",
    "Stark Industries", "Wayne Enterprises", "Cyberdyne", "Soylent Corp",
    "Wonka Industries", "Dunder Mifflin",
]

MONTHS = [
    ("2024-01", "sales_january.csv", 31),
    ("2024-02", "sales_february.csv", 29),
    ("2024-03", "sales_march.csv", 31),
]

# Growth factors per month to create meaningful trends
# Product A (Laptop Pro) grows, Wireless Mouse declines
PRODUCT_MONTH_MULTIPLIER = {
    "Laptop Pro":     [1.0, 1.15, 1.30],   # Growing
    "Wireless Mouse": [1.0, 0.90, 0.75],   # Declining
    "Office Chair":   [1.0, 1.05, 1.10],   # Slight growth
    "Standing Desk":  [1.0, 1.00, 1.05],   # Stable
    "Notebook Set":   [1.0, 1.10, 1.20],   # Growing
}

REGION_MULTIPLIER = {
    "North": 1.2,
    "South": 0.9,
    "East": 1.0,
    "West": 1.1,
}


def generate_month_data(month_prefix: str, days_in_month: int, month_index: int) -> list[dict]:
    """Generate sales rows for a single month."""
    rows = []
    row_id = 0

    for day in range(1, days_in_month + 1):
        date_str = f"{month_prefix}-{day:02d}"
        # 3-6 transactions per day
        num_transactions = random.randint(3, 6)

        for _ in range(num_transactions):
            product = random.choice(list(PRODUCTS.keys()))
            info = PRODUCTS[product]
            region = random.choice(REGIONS)
            salesperson = random.choice(SALESPERSONS)
            customer = random.choice(CUSTOMERS)

            quantity = random.randint(1, 10)
            base_revenue = info["base_price"] * quantity
            # Apply growth/decline trend
            trend = PRODUCT_MONTH_MULTIPLIER[product][month_index]
            # Apply region multiplier
            region_mult = REGION_MULTIPLIER[region]
            # Small random variation (+/- 10%)
            variation = random.uniform(0.90, 1.10)

            revenue = round(base_revenue * trend * region_mult * variation, 2)
            # Profit margin: 15-35% of revenue
            margin = random.uniform(0.15, 0.35)
            profit = round(revenue * margin, 2)

            # Introduce some missing profit values (~5% of rows)
            if random.random() < 0.05:
                profit = ""

            rows.append({
                "date": date_str,
                "product": product,
                "category": info["category"],
                "region": region,
                "salesperson": salesperson,
                "quantity": quantity,
                "revenue": revenue,
                "profit": profit,
                "customer": customer,
            })
            row_id += 1

    return rows


def main():
    output_dir = os.path.join(os.path.dirname(__file__), "sample_data")
    os.makedirs(output_dir, exist_ok=True)

    fieldnames = ["date", "product", "category", "region", "salesperson",
                  "quantity", "revenue", "profit", "customer"]

    for month_index, (prefix, filename, days) in enumerate(MONTHS):
        rows = generate_month_data(prefix, days, month_index)
        filepath = os.path.join(output_dir, filename)

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"Created {filename}: {len(rows)} rows")


if __name__ == "__main__":
    main()
