import pandas as pd
from pathlib import Path


def get_metrics(file_path):
    df = pd.read_csv(file_path)

    checked_users = len(df)
    hits = df["hit"].sum()
    precision_at_10 = df["precision_at_10"].mean()
    recall_at_10 = df["recall_at_10"].mean()

    return checked_users, hits, precision_at_10, recall_at_10


def main():
    print("COMPARECOMPARECOMPARECOMPARECOMPARE")

    project_root = Path(__file__).resolve().parent.parent

    als_file = project_root / "data" / "als_common_evaluation_2019-Nov.csv"
    svdpp_file = project_root / "data" / "svdpp_common_evaluation_2019-Nov.csv"
    slim_file = project_root / "data" / "slim_common_evaluation_2019-Nov.csv"

    output_file = project_root / "data" / "models_common_comparison_2019-Nov.csv"

    print("\n==============================")
    print("1. ЗАГРУЗКА COMMON-РЕЗУЛЬТАТОВ МОДЕЛЕЙ")
    print("==============================")

    als_users, als_hits, als_precision, als_recall = get_metrics(als_file)
    svdpp_users, svdpp_hits, svdpp_precision, svdpp_recall = get_metrics(svdpp_file)
    slim_users, slim_hits, slim_precision, slim_recall = get_metrics(slim_file)

    print("\nРезультаты ALS загружены.")
    print("\nРезультаты SVD++ загружены.")
    print("\nРезультаты SLIM загружены.")

    print("\n==============================")
    print("2. СОЗДАНИЕ ИТОГОВОЙ ТАБЛИЦЫ СРАВНЕНИЯ")
    print("==============================")

    comparison_df = pd.DataFrame([
        {
            "model": "ALS",
            "train_rows": 366718,
            "users_count": 48040,
            "items_count": 2500,
            "hidden_users": als_users,
            "hits_at_10": als_hits,
            "precision_at_10": round(als_precision, 4),
            "recall_at_10": round(als_recall, 4),
            "time_seconds": 36.21,
            "comment": "Быстрая модель для неявной обратной связи"
        },
        {
            "model": "SVD++",
            "train_rows": 366718,
            "users_count": 48040,
            "items_count": 2500,
            "hidden_users": svdpp_users,
            "hits_at_10": svdpp_hits,
            "precision_at_10": round(svdpp_precision, 4),
            "recall_at_10": round(svdpp_recall, 4),
            "time_seconds": 97.04,
            "comment": "Работает через условные оценки rating"
        },
        {
            "model": "SLIM",
            "train_rows": 366718,
            "users_count": 48040,
            "items_count": 2500,
            "hidden_users": slim_users,
            "hits_at_10": slim_hits,
            "precision_at_10": round(slim_precision, 4),
            "recall_at_10": round(slim_recall, 4),
            "time_seconds": 97.62,
            "comment": "Item-item модель, лучший результат по recall@10"
        }
    ])

    comparison_df = comparison_df.sort_values(
        by="recall_at_10",
        ascending=False
    )

    print("\nИтоговая таблица сравнения моделей:")
    print(comparison_df.to_string(index=False))

    comparison_df.to_csv(output_file, index=False)

    print("\nТаблица сравнения сохранена в файл:")
    print(output_file)

    print("\n==============================")
    print("3. КРАТКИЙ ВЫВОД")
    print("==============================")

    best_model = comparison_df.iloc[0]["model"]
    best_recall = comparison_df.iloc[0]["recall_at_10"]

    print("\nЛучшая модель по recall@10:")
    print(best_model)

    print("\nЛучший recall@10:")
    print(best_recall)


if __name__ == "__main__":
    main()