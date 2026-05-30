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

    als_file = project_root / "data" / "als_evaluation_2019-Nov.csv"
    svdpp_file = project_root / "data" / "svdpp_evaluation_2019-Nov.csv"
    slim_file = project_root / "data" / "slim_evaluation_2019-Nov.csv"

    print("\n==============================")
    print("1. ЗАГРУЗКА РЕЗУЛЬТАТОВ МОДЕЛЕЙ")
    print("==============================")

    als_users, als_hits, als_precision, als_recall = get_metrics(als_file)
    svdpp_users, svdpp_hits, svdpp_precision, svdpp_recall = get_metrics(svdpp_file)
    slim_users, slim_hits, slim_precision, slim_recall = get_metrics(slim_file)

    print("\nРезультаты ALS загружены.")
    print("\nРезультаты SVD++ загружены.")
    print("\nРезультаты SLIM загружены.")





    print("\n==============================")
    print("2. СОЗДАНИЕ ТАБЛИЦЫ СРАВНЕНИЯ")
    print("==============================")

    comparison_df = pd.DataFrame([
        {
            "model": "ALS",
            "train_users": 150000,
            "checked_users": als_users,
            "hits_at_10": als_hits,
            "precision_at_10": round(als_precision, 4),
            "recall_at_10": round(als_recall, 4),
            "time_seconds": 51.98,
            "comment": "Быстрая модель, хорошо подходит для неявной обратной связи"
        },
        {
            "model": "SVD++",
            "train_users": 25000,
            "checked_users": svdpp_users,
            "hits_at_10": svdpp_hits,
            "precision_at_10": round(svdpp_precision, 4),
            "recall_at_10": round(svdpp_recall, 4),
            "time_seconds": 69.84,
            "comment": "Использует условные оценки, показал более слабое качество"
        },
        {
            "model": "SLIM",
            "train_users": 25000,
            "checked_users": slim_users,
            "hits_at_10": slim_hits,
            "precision_at_10": round(slim_precision, 4),
            "recall_at_10": round(slim_recall, 4),
            "time_seconds": 128.72,
            "comment": "Item-item модель, показала лучший recall@10"
        }
    ])

    comparison_df = comparison_df.sort_values(
        by="recall_at_10",
        ascending=False
    )

    print("\nИтоговая таблица сравнения:")
    print(comparison_df.to_string(index=False))

    output_file = project_root / "data" / "models_comparison_2019-Nov.csv"
    comparison_df.to_csv(output_file, index=False)



    print("\nТаблица сравнения сохранена в файл:")
    print(output_file)


if __name__ == "__main__":
    main()