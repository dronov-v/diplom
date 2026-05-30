import time
import warnings
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

    data_path = project_root / "data" / "matrix_2019-Nov.csv"
    info_file = project_root / "data" / "filtered_2019-Nov.csv"

    print("\n==============================")
    print("1. ЗАГРУЗКА ДАННЫХ ДЛЯ SLIM")
    print("==============================")










    df = pd.read_csv(data_path)

    print("\nРазмер таблицы matrix_2019-Nov.csv:")
    print(df.shape)

    print("\nПервые 10 строк:")
    print(df.head(10).to_string(index=False))

    print("\n==============================")
    print("2. ЗАГРУЗКА ИНФОРМАЦИИ О ТОВАРАХ")
    print("==============================")

    info_df = pd.read_csv(
        info_file,
        usecols=["product_id", "category_code", "brand"]
    )

    print("\nРазмер таблицы filtered_2019-Nov.csv до удаления дублей:")
    print(info_df.shape)

    info_df = info_df.drop_duplicates(subset=["product_id"])

    print("\nРазмер таблицы с информацией о товарах после удаления дублей:")
    print(info_df.shape)

    print("\n==============================")
    print("3. ОГРАНИЧЕНИЕ ДАННЫХ ДЛЯ SLIM")
    print("==============================")

    users_count = 25000
    popular_items_count = 1500

    first_users = df["user_id"].drop_duplicates().head(users_count)
    new_df = df[df["user_id"].isin(first_users)].copy()

    popular_items = (
        new_df.groupby("product_id")["interaction_weight"]
        .sum()
        .sort_values(ascending=False)
        .head(popular_items_count)
        .index
    )

    new_df = new_df[new_df["product_id"].isin(popular_items)].copy()

    print("\nСколько пользователей берем для SLIM:")
    print(users_count)

    print("\nСколько популярных товаров берем для SLIM:")
    print(popular_items_count)



    print("\nРазмер таблицы после ограничения:")
    print(new_df.shape)

    print("\nКоличество уникальных пользователей:")
    print(new_df["user_id"].nunique())

    print("\nКоличество уникальных товаров:")
    print(new_df["product_id"].nunique())

    print("\nПервые 10 строк после ограничения:")
    print(new_df.head(10).to_string(index=False))

    print("\n==============================")
    print("4. СОЗДАНИЕ ВНУТРЕННИХ ИНДЕКСОВ")
    print("==============================")

    new_users = new_df["user_id"].unique()
    new_items = new_df["product_id"].unique()

    user_map = {user_id: index for index, user_id in enumerate(new_users)}
    item_map = {product_id: index for index, product_id in enumerate(new_items)}
    reverse_item_map = {index: product_id for product_id, index in item_map.items()}

    new_df["user_index"] = new_df["user_id"].map(user_map)
    new_df["item_index"] = new_df["product_id"].map(item_map)

    print("\nКоличество пользователей для модели:")
    print(len(new_users))

    print("\nКоличество товаров для модели:")
    print(len(new_items))

    print("\nТаблица после добавления user_index и item_index:")
    print(
        new_df[
            ["user_id", "user_index", "product_id", "item_index", "interaction_weight"]
        ].head(10).to_string(index=False)
    )

    print("\n==============================")
    print("5. TRAIN/TEST SPLIT ДЛЯ ОЦЕНКИ SLIM")
    print("==============================")

    evaluation_users_count = 200


    user_products_count = new_df.groupby("user_id")["product_id"].nunique()

    users_for_evaluation = user_products_count[
        user_products_count >= 2
    ].index[:evaluation_users_count]







    evaluation_df = new_df[
        new_df["user_id"].isin(users_for_evaluation)
    ].copy()

    test_df = evaluation_df.groupby(
        "user_id",
        group_keys=False
    ).sample(n=1, random_state=42)

    train_df = new_df.drop(test_df.index).copy()

    print("\nСколько пользователей берем для оценки качества SLIM:")
    print(evaluation_users_count)

    print("\nРазмер train_df:")
    print(train_df.shape)

    print("\nРазмер test_df:")
    print(test_df.shape)

    print("\nПервые 10 строк train_df:")
    print(
        train_df[
            ["user_id", "product_id", "interaction_weight", "user_index", "item_index"]
        ].head(10).to_string(index=False)
    )

    print("\nПервые 10 спрятанных товаров test_df:")
    print(
        test_df[
            ["user_id", "product_id", "interaction_weight", "user_index", "item_index"]
        ].head(10).to_string(index=False)
    )

    print("\nПояснение:")
    print("train_df — данные, на которых обучается SLIM.")
    print("test_df — товары, которые мы спрятали для проверки качества.")

    print("\n==============================")
    print("6. СОЗДАНИЕ USER-ITEM MATRIX ДЛЯ ОБУЧЕНИЯ")
    print("==============================")

    train_matrix = csr_matrix(
        (
            train_df["interaction_weight"],
            (train_df["user_index"], train_df["item_index"])
        ),
        shape=(len(new_users), len(new_items))
    )

    print("\nМатрица train_matrix создана.")
    print("Размер матрицы:", train_matrix.shape)
    print("Количество ненулевых значений:", train_matrix.nnz)

    print("\n==============================")
    print("7. ОБУЧЕНИЕ МОДЕЛИ SLIM")
    print("==============================")

    items_count = train_matrix.shape[1]

    similarity_matrix = lil_matrix((items_count, items_count))

    slim_model = ElasticNet(
        alpha=0.01,
        l1_ratio=0.1,
        positive=True,
        fit_intercept=False,
        max_iter=200,
        random_state=42
    )

    print("\nПараметры SLIM:")
    print("alpha = 0.01")
    print("l1_ratio = 0.1")
    print("positive = True")
    print("max_iter = 120")

    print("\nНачинаем обучение SLIM.")
    print("Модель будет по очереди обучаться для каждого товара.")
    print("Выводим прогресс каждые 100 товаров.")

    for item_index in range(items_count):
        if item_index % 100 == 0:
            print("Обработано товаров:", item_index, "из", items_count)

        y = train_matrix[:, item_index].toarray().ravel()

        X = train_matrix.copy().tolil()
        X[:, item_index] = 0
        X = X.tocsr()

        slim_model.fit(X, y)

        coefs = slim_model.coef_

        for related_item_index, weight in enumerate(coefs):
            if weight > 0:
                similarity_matrix[related_item_index, item_index] = weight

    similarity_matrix = similarity_matrix.tocsr()

    print("\nОбучение SLIM завершено.")
    print("Размер матрицы похожести товаров:", similarity_matrix.shape)
    print("Количество ненулевых значений:", similarity_matrix.nnz)

    print("\n==============================")
    print("8. ОЦЕНКА КАЧЕСТВА SLIM")
    print("==============================")

    recommendations_count = 10

    results = []

    for _, row in test_df.iterrows():
        real_user_id = int(row["user_id"])
        hidden_product_id = int(row["product_id"])
        user_index = int(row["user_index"])

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
            "hidden_product_id": hidden_product_id,
            "recommended_product_ids": recommended_product_ids,
            "hit": hit,
            "precision_at_10": precision_at_10,
            "recall_at_10": recall_at_10
        })

    evaluation_results = pd.DataFrame(results)

    average_precision_at_10 = evaluation_results["precision_at_10"].mean()
    average_recall_at_10 = evaluation_results["recall_at_10"].mean()
    hits_count = evaluation_results["hit"].sum()

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

    evaluation_output_file = project_root / "data" / "slim_evaluation_2019-Nov.csv"
    evaluation_results.to_csv(evaluation_output_file, index=False)

    print("\nРезультаты оценки SLIM сохранены в файл:")
    print(evaluation_output_file)

    print("\n==============================")
    print("9. ВЫБОР ПОЛЬЗОВАТЕЛЯ ДЛЯ РЕКОМЕНДАЦИЙ")
    print("==============================")

    user_stats = new_df.groupby("user_id").agg(
        actions_count=("product_id", "count"),
        products_count=("product_id", "nunique"),
        total_weight=("interaction_weight", "sum")
    ).reset_index()

    print("\nСтатистика по первым 10 пользователям:")
    print(user_stats.head(10).to_string(index=False))

    min_products_count = 5
    max_products_count = 50

    good_users = user_stats[
        (user_stats["products_count"] >= min_products_count) &
        (user_stats["products_count"] <= max_products_count)
    ].copy()

    good_users = good_users.sort_values(
        by=["products_count", "total_weight"],
        ascending=False
    )

    print("\nПользователи, подходящие для проверки рекомендаций:")
    print(good_users.head(10).to_string(index=False))

    if len(good_users) == 0:
        print("\nНе найдено пользователей с подходящим количеством товаров.")
        print("Для проверки берем первого пользователя из таблицы.")
        real_user_id = new_df.iloc[0]["user_id"]
    else:
        real_user_id = good_users.iloc[0]["user_id"]

    user_id = int(real_user_id)
    user_index = user_map[real_user_id]

    print("\nВыбран пользователь:")
    print("user_id:", user_id)
    print("user_index:", user_index)

    print("\n==============================")
    print("10. ИСТОРИЯ ВЫБРАННОГО ПОЛЬЗОВАТЕЛЯ")
    print("==============================")

    user_history = new_df[
        new_df["user_id"] == real_user_id
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

    history_output_file = project_root / "data" / "slim_user_history_2019-Nov.csv"
    user_history_full.to_csv(history_output_file, index=False)

    print("\nИстория пользователя сохранена в файл:")
    print(history_output_file)

    print("\n==============================")
    print("11. ПОЛУЧЕНИЕ РЕКОМЕНДАЦИЙ SLIM")
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

    output_file = project_root / "data" / "slim_recommendations_2019-Nov.csv"
    rec_table_full.to_csv(output_file, index=False)

    print("\nРекомендации SLIM сохранены в файл:")
    print(output_file)

    print("\n==============================")
    print("12. ЗАВЕРШЕНИЕ РАБОТЫ")
    print("==============================")

    end_time = time.time()
    print(f"\nВремя выполнения скрипта: {end_time - start_time:.2f} секунд")


if __name__ == "__main__":
    main()