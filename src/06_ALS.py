import time
import pandas as pd
from pathlib import Path
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares


def main():
    print("ALSALSALSALSALSALSALSALSALSALSALS")

    start_time = time.time()

    project_root = Path(__file__).resolve().parent.parent

    common_file = project_root / "data" / "common_interactions_2019-Nov.csv"
    train_file = project_root / "data" / "common_train_2019-Nov.csv"
    hidden_file = project_root / "data" / "common_hidden_2019-Nov.csv"
    info_file = project_root / "data" / "filtered_2019-Nov.csv"

    print("\n==============================")
    print("1. ЗАГРУЗКА ОБЩИХ ДАННЫХ ДЛЯ ALS")
    print("==============================")

    common_df = pd.read_csv(common_file)
    train_df = pd.read_csv(train_file)
    hidden_df = pd.read_csv(hidden_file)

    print("\nРазмер common_interactions_2019-Nov.csv:")
    print(common_df.shape)

    print("\nРазмер common_train_2019-Nov.csv:")
    print(train_df.shape)

    print("\nРазмер common_hidden_2019-Nov.csv:")
    print(hidden_df.shape)

    print("\nПервые 10 строк common_train:")
    print(train_df.head(10).to_string(index=False))

    print("\nПервые 10 hidden-товаров:")
    print(hidden_df.head(10).to_string(index=False))

    print("\n==============================")
    print("2. ЗАГРУЗКА ИНФОРМАЦИИ О ТОВАРАХ")
    print("==============================")

    info_df = pd.read_csv(
        info_file,
        usecols=["product_id", "category_code", "brand"]
    )

    print("\nРазмер filtered_2019-Nov.csv до удаления дублей:")
    print(info_df.shape)

    info_df = info_df.drop_duplicates(subset=["product_id"])

    print("\nРазмер информации о товарах после удаления дублей:")
    print(info_df.shape)

    print("\n==============================")
    print("3. ПОДГОТОВКА ИНДЕКСОВ")
    print("==============================")

    users_count = common_df["user_index"].nunique()
    items_count = common_df["item_index"].nunique()

    print("\nКоличество пользователей в общем наборе:")
    print(users_count)

    print("\nКоличество товаров в общем наборе:")
    print(items_count)

    item_index_table = common_df[
        ["item_index", "product_id"]
    ].drop_duplicates()

    reverse_item_map = dict(
        zip(
            item_index_table["item_index"],
            item_index_table["product_id"]
        )
    )

    print("\nПервые 10 соответствий item_index -> product_id:")
    print(item_index_table.head(10).to_string(index=False))

    print("\n==============================")
    print("4. СОЗДАНИЕ TRAIN MATRIX")
    print("==============================")

    train_matrix = csr_matrix(
        (
            train_df["interaction_weight"],
            (train_df["user_index"], train_df["item_index"])
        ),
        shape=(users_count, items_count)
    )

    print("\nМатрица train_matrix создана.")
    print("Размер матрицы:", train_matrix.shape)
    print("Количество ненулевых значений:", train_matrix.nnz)

    print("\n==============================")
    print("5. ОБУЧЕНИЕ ALS НА ОБЩИХ ДАННЫХ")
    print("==============================")

    model = AlternatingLeastSquares(
        factors=60,
        regularization=0.1,
        iterations=30
    )

    print("\nПараметры ALS:")
    print("factors = 60")
    print("regularization = 0.1")
    print("iterations = 30")

    print("\nНачинаем обучение ALS:")
    model.fit(train_matrix)

    print("\nОбучение ALS завершено.")

    print("\n==============================")
    print("6. ОЦЕНКА ALS НА HIDDEN-ТОВАРАХ")
    print("==============================")

    recommendations_count = 10

    results = []

    for _, row in hidden_df.iterrows():
        real_user_id = int(row["user_id"])
        user_index = int(row["user_index"])
        hidden_product_id = int(row["hidden_product_id"])

        rec_items, scores = model.recommend(
            user_index,
            train_matrix[user_index],
            N=recommendations_count,
            filter_already_liked_items=True
        )

        recommended_product_ids = [
            int(reverse_item_map[int(item_index)])
            for item_index in rec_items
        ]

        hit = hidden_product_id in recommended_product_ids

        if hit:
            precision_at_10 = 1 / recommendations_count
            recall_at_10 = 1
        else:
            precision_at_10 = 0
            recall_at_10 = 0

        results.append({
            "user_id": real_user_id,
            "user_index": user_index,
            "hidden_product_id": hidden_product_id,
            "recommended_product_ids": recommended_product_ids,
            "hit": hit,
            "precision_at_10": precision_at_10,
            "recall_at_10": recall_at_10
        })

    evaluation_results = pd.DataFrame(results)

    hits_count = evaluation_results["hit"].sum()
    average_precision_at_10 = evaluation_results["precision_at_10"].mean()
    average_recall_at_10 = evaluation_results["recall_at_10"].mean()

    print("\nКоличество проверенных пользователей:")
    print(len(evaluation_results))

    print("\nКоличество попаданий в топ-10:")
    print(hits_count)

    print("\nСредний precision@10:")
    print(round(average_precision_at_10, 4))

    print("\nСредний recall@10:")
    print(round(average_recall_at_10, 4))

    print("\nПервые 10 строк результата оценки:")
    print(evaluation_results.head(10).to_string(index=False))

    evaluation_output_file = project_root / "data" / "als_common_evaluation_2019-Nov.csv"
    evaluation_results.to_csv(evaluation_output_file, index=False)

    print("\nРезультаты оценки ALS сохранены в файл:")
    print(evaluation_output_file)

    print("\n==============================")
    print("7. ВЫБОР ПОЛЬЗОВАТЕЛЯ ДЛЯ ПРИМЕРА")
    print("==============================")

    user_stats = common_df.groupby("user_id").agg(
        actions_count=("product_id", "count"),
        products_count=("product_id", "nunique"),
        total_weight=("interaction_weight", "sum")
    ).reset_index()

    min_products_count = 10
    max_products_count = 50

    good_users = user_stats[
        (user_stats["products_count"] >= min_products_count) &
        (user_stats["products_count"] <= max_products_count)
    ].copy()

    good_users = good_users.sort_values(
        by=["products_count", "total_weight"],
        ascending=False
    )

    print("\nПользователи, подходящие для примера рекомендаций:")
    print(good_users.head(10).to_string(index=False))

    if len(good_users) == 0:
        print("\nНе найдено пользователей с подходящим количеством товаров.")
        real_user_id = common_df.iloc[0]["user_id"]
    else:
        real_user_id = good_users.iloc[0]["user_id"]

    user_id = int(real_user_id)

    user_index = int(
        common_df[
            common_df["user_id"] == real_user_id
        ]["user_index"].iloc[0]
    )

    print("\nВыбран пользователь:")
    print("user_id:", user_id)
    print("user_index:", user_index)

    print("\n==============================")
    print("8. ИСТОРИЯ ВЫБРАННОГО ПОЛЬЗОВАТЕЛЯ")
    print("==============================")

    user_history = common_df[
        common_df["user_id"] == real_user_id
    ][["product_id", "interaction_weight"]]

    user_history_full = user_history.merge(
        info_df,
        on="product_id",
        how="left"
    )

    user_history_full.insert(0, "user_id", user_id)

    user_history_full = user_history_full.sort_values(
        by="interaction_weight",
        ascending=False
    )

    print("\nВсего товаров в истории пользователя:")
    print(len(user_history_full))

    print("\nПервые 20 товаров из истории пользователя:")
    print(user_history_full.head(20).to_string(index=False))

    history_output_file = project_root / "data" / "als_common_user_history_2019-Nov.csv"
    user_history_full.to_csv(history_output_file, index=False)

    print("\nИстория пользователя сохранена в файл:")
    print(history_output_file)

    print("\n==============================")
    print("9. ПОЛУЧЕНИЕ РЕКОМЕНДАЦИЙ ALS ДЛЯ ПРИМЕРА")
    print("==============================")

    rec_items, scores = model.recommend(
        user_index,
        train_matrix[user_index],
        N=recommendations_count,
        filter_already_liked_items=True
    )

    real_product_ids = [
        int(reverse_item_map[int(item_index)])
        for item_index in rec_items
    ]

    rec_table = pd.DataFrame({
        "item_index": rec_items,
        "score": scores,
        "product_id": real_product_ids
    })

    print("\nТаблица рекомендаций до добавления описания товаров:")
    print(rec_table.to_string(index=False))

    rec_table_full = rec_table.merge(
        info_df,
        on="product_id",
        how="left"
    )

    rec_table_full.insert(0, "user_id", user_id)

    print("\nРекомендации ALS с category_code и brand:")
    print(rec_table_full.to_string(index=False))

    output_file = project_root / "data" / "als_common_recommendations_2019-Nov.csv"
    rec_table_full.to_csv(output_file, index=False)

    print("\nРекомендации ALS сохранены в файл:")
    print(output_file)

    print("\n==============================")
    print("10. ЗАВЕРШЕНИЕ РАБОТЫ")
    print("==============================")

    end_time = time.time()
    print(f"\nВремя выполнения скрипта: {end_time - start_time:.2f} секунд")


if __name__ == "__main__":
    main()