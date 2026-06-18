import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.linear_model import ElasticNet
from sklearn.exceptions import ConvergenceWarning


def main():
    print("SLIMSLIMSLIMSLIMSLIMSLIMSLIM")

    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    start_time = time.time()

    project_root = Path(__file__).resolve().parent.parent

    common_file = project_root / "data" / "common_interactions_2019-Nov.csv"
    train_file = project_root / "data" / "common_train_2019-Nov.csv"
    hidden_file = project_root / "data" / "common_hidden_2019-Nov.csv"
    info_file = project_root / "data" / "filtered_2019-Nov.csv"

    print("\n==============================")
    print("1. ЗАГРУЗКА ОБЩИХ ДАННЫХ ДЛЯ SLIM")
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
    print("5. ОБУЧЕНИЕ SLIM НА ОБЩИХ ДАННЫХ")
    print("==============================")

    similarity_matrix = lil_matrix((items_count, items_count))

    slim_model = ElasticNet(
        alpha=0.01,
        l1_ratio=0.1,
        positive=True,
        fit_intercept=False,
        max_iter=150,
        random_state=42,
        selection="random"
    )

    print("\nПараметры SLIM:")
    print("alpha = 0.01")
    print("l1_ratio = 0.1")
    print("positive = True")
    print("max_iter = 150")
    print("selection = random")

    print("\nНачинаем обучение SLIM.")
    print("Модель будет по очереди обучаться для каждого товара.")
    print("Выводим прогресс каждые 100 товаров.")

    matrix_for_training = train_matrix.tocsc(copy=True)

    for item_index in range(items_count):
        if item_index % 100 == 0:
            print("Обработано товаров:", item_index, "из", items_count)

        y = train_matrix[:, item_index].toarray().ravel()

        column_start = matrix_for_training.indptr[item_index]
        column_end = matrix_for_training.indptr[item_index + 1]

        saved_column_values = matrix_for_training.data[column_start:column_end].copy()

        matrix_for_training.data[column_start:column_end] = 0

        slim_model.fit(matrix_for_training, y)

        matrix_for_training.data[column_start:column_end] = saved_column_values

        coefs = slim_model.coef_

        positive_indices = np.where(coefs > 0)[0]

        for related_item_index in positive_indices:
            similarity_matrix[related_item_index, item_index] = coefs[related_item_index]

    similarity_matrix = similarity_matrix.tocsr()

    print("\nОбучение SLIM завершено.")
    print("Размер матрицы похожести товаров:", similarity_matrix.shape)
    print("Количество ненулевых значений:", similarity_matrix.nnz)

    print("\n==============================")
    print("6. ОЦЕНКА SLIM НА HIDDEN-ТОВАРАХ")
    print("==============================")

    recommendations_count = 10

    results = []

    for _, row in hidden_df.iterrows():
        real_user_id = int(row["user_id"])
        user_index = int(row["user_index"])
        hidden_product_id = int(row["hidden_product_id"])

        user_vector = train_matrix[user_index]

        scores = user_vector.dot(similarity_matrix).toarray().ravel()

        user_items = user_vector.indices
        scores[user_items] = -1

        top_item_indices = scores.argsort()[::-1][:recommendations_count]

        recommended_product_ids = [
            int(reverse_item_map[int(item_index)])
            for item_index in top_item_indices
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

    evaluation_output_file = project_root / "data" / "slim_common_evaluation_2019-Nov.csv"
    evaluation_results.to_csv(evaluation_output_file, index=False)

    print("\nРезультаты оценки SLIM сохранены в файл:")
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

    history_output_file = project_root / "data" / "slim_common_user_history_2019-Nov.csv"
    user_history_full.to_csv(history_output_file, index=False)

    print("\nИстория пользователя сохранена в файл:")
    print(history_output_file)

    print("\n==============================")
    print("9. ПОЛУЧЕНИЕ РЕКОМЕНДАЦИЙ SLIM ДЛЯ ПРИМЕРА")
    print("==============================")

    user_vector = train_matrix[user_index]

    scores = user_vector.dot(similarity_matrix).toarray().ravel()

    user_items = user_vector.indices
    scores[user_items] = -1

    top_item_indices = scores.argsort()[::-1][:recommendations_count]

    rec_table = pd.DataFrame({
        "item_index": top_item_indices,
        "score": scores[top_item_indices]
    })

    rec_table["product_id"] = rec_table["item_index"].apply(
        lambda item_index: reverse_item_map[int(item_index)]
    )

    print("\nТаблица рекомендаций до добавления описания товаров:")
    print(rec_table.to_string(index=False))

    rec_table_full = rec_table.merge(
        info_df,
        on="product_id",
        how="left"
    )

    rec_table_full.insert(0, "user_id", user_id)

    print("\nРекомендации SLIM с category_code и brand:")
    print(rec_table_full.to_string(index=False))

    output_file = project_root / "data" / "slim_common_recommendations_2019-Nov.csv"
    rec_table_full.to_csv(output_file, index=False)

    print("\nРекомендации SLIM сохранены в файл:")
    print(output_file)

    print("\n==============================")
    print("10. ЗАВЕРШЕНИЕ РАБОТЫ")
    print("==============================")

    end_time = time.time()
    print(f"\nВремя выполнения скрипта: {end_time - start_time:.2f} секунд")


if __name__ == "__main__":
    main()