import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def save_bar_chart(data, title, xlabel, ylabel, output_path, rotation=0):
    plt.figure(figsize=(10, 6))
    plt.bar(data.index.astype(str), data.values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    print(";;;;;;;;;;;;;;;;;;;;;;;;';;;;;;;;';;;;;;;;;;;;;';")

    project_root = Path(__file__).resolve().parent.parent

    data_dir = project_root / "data"
    figures_dir = project_root / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    filtered_file = data_dir / "filtered_2019-Nov.csv"
    common_file = data_dir / "common_interactions_2019-Nov.csv"
    comparison_file = data_dir / "models_common_comparison_2019-Nov.csv"

    print("\n1. СОЗДАНИЕ ГРАФИКА РАСПРЕДЕЛЕНИЯ СОБЫТИЙ")

    if filtered_file.exists():
        events_df = pd.read_csv(filtered_file, usecols=["event_type"])

        event_counts = events_df["event_type"].value_counts()

        save_bar_chart(
            event_counts,
            "Распределение пользовательских событий",
            "Тип события",
            "Количество событий",
            figures_dir / "event_type_distribution.png"
        )

        print("Сохранено:", figures_dir / "event_type_distribution.png")
    else:
        print("Файл не найден:", filtered_file)

    print("\n2. СОЗДАНИЕ ГРАФИКА ТОП-10 КАТЕГОРИЙ")

    if filtered_file.exists():
        categories_df = pd.read_csv(filtered_file, usecols=["category_code"])

        top_categories = categories_df["category_code"].value_counts().head(10)

        save_bar_chart(
            top_categories,
            "Топ-10 категорий товаров после фильтрации",
            "Категория",
            "Количество событий",
            figures_dir / "top_10_categories.png",
            rotation=45
        )

        print("Сохранено:", figures_dir / "top_10_categories.png")
    else:
        print("Файл не найден:", filtered_file)

    print("\n3. СОЗДАНИЕ ГРАФИКА РАСПРЕДЕЛЕНИЯ interaction_weight")

    if common_file.exists():
        common_df = pd.read_csv(common_file, usecols=["interaction_weight"])

        weights = common_df["interaction_weight"].copy()

        weights_clipped = weights.clip(upper=20)

        weight_counts = weights_clipped.value_counts().sort_index()
        weight_counts.index = [
            "20+" if value == 20 else str(value)
            for value in weight_counts.index
        ]

        save_bar_chart(
            weight_counts,
            "Распределение значений interaction_weight",
            "Значение interaction_weight",
            "Количество взаимодействий",
            figures_dir / "interaction_weight_distribution.png"
        )

        print("Сохранено:", figures_dir / "interaction_weight_distribution.png")
    else:
        print("Файл не найден:", common_file)

    print("\n4. СОЗДАНИЕ ГРАФИКОВ СРАВНЕНИЯ МОДЕЛЕЙ")

    if comparison_file.exists():
        comparison_df = pd.read_csv(comparison_file)

        comparison_df = comparison_df.sort_values(
            by="recall_at_10",
            ascending=False
        )

        recall_data = comparison_df.set_index("model")["recall_at_10"]

        save_bar_chart(
            recall_data,
            "Сравнение моделей по recall@10",
            "Модель",
            "Recall@10",
            figures_dir / "models_recall_at_10.png"
        )

        print("Сохранено:", figures_dir / "models_recall_at_10.png")

        precision_data = comparison_df.set_index("model")["precision_at_10"]

        save_bar_chart(
            precision_data,
            "Сравнение моделей по precision@10",
            "Модель",
            "Precision@10",
            figures_dir / "models_precision_at_10.png"
        )

        print("Сохранено:", figures_dir / "models_precision_at_10.png")

        time_data = comparison_df.set_index("model")["time_seconds"]

        save_bar_chart(
            time_data,
            "Сравнение моделей по времени выполнения",
            "Модель",
            "Время, сек.",
            figures_dir / "models_time_seconds.png"
        )

        print("Сохранено:", figures_dir / "models_time_seconds.png")
    else:
        print("Файл не найден:", comparison_file)

    print("\nГрафики сохранены в папку:")
    print(figures_dir)


if __name__ == "__main__":
    main()