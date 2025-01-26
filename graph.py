import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


with open("results.json", 'r') as file:
    data = json.load(file)


rows = []

for dataset in data:
    users = dataset["header"]["users"]
    items = dataset["header"]["items"]

    for method_result in dataset["body"]:
        rows.append({
            "users": users,
            "items": items,
            "method": method_result["method"],
            "f1_score": method_result["f1_score"],
            "calculate_time": method_result["calculate_time"],
        })


df = pd.DataFrame(rows)


plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x="items", y="f1_score", hue="method", marker="o")
plt.title("F1 Score Results", fontsize=16)
plt.xlabel("Number of Items", fontsize=14)
plt.ylabel("F1 Score", fontsize=14)
plt.legend(title="Method")
plt.grid()
plt.show()


plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x="items", y="calculate_time", hue="method", marker="o")
plt.title("Calculation Time Results", fontsize=16)
plt.xlabel("Number of Items", fontsize=14)
plt.ylabel("Calculation Time (seconds)", fontsize=14)
plt.legend(title="Method")
plt.grid()
plt.show()