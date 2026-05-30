import time
import pandas as pd
from pathlib import Path
from surprise import Dataset, Reader, SVDpp


def convert_weight_to_rating(weight):
    if weight <= 1:
        return 1
    elif weight <= 2:
        return 2
    elif weight <= 5:
        return 3
    elif weight <= 10:
        return 4
    else:
        return 5


def main():
    print("SVDPPSVDPPSVDPPSVDPPSVDPP")

    start_time = time.time()

    project_root = Path(__file__).resolve().parent.parent

    data_path = project_root / "data" / "matrix_2019-Nov.csv"
    info_file = project_root / "data" / "filtered_2019-Nov.csv"



    print("\n==============================")
    print("1. ЗАГРУЗКА ДАННЫХ ДЛЯ SVD++")
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
    print("3. ОГРАНИЧЕНИЕ КОЛИЧЕСТВА ПОЛЬЗОВАТЕЛЕЙ")
    print("==============================")


    users_count = 25000

    first_users = df["user_id"].drop_duplicates().head(users_count)
    new_df = df[df["user_id"].isin(first_users)].copy()

    print("\nСколько пользователей берем для SVD++:")
    print(users_count)

    print("\nРазмер таблицы после ограничения пользователей:")
    print(new_df.shape)

    print("\nКоличество уникальных пользователей:")
    print(new_df["user_id"].nunique())

    print("\nКоличество уникальных товаров:")
    print(new_df["product_id"].nunique())

    print("\nПервые 10 строк после ограничения:")
    print(new_df.head(10).to_string(index=False))

    print("\n==============================")
    print("4. ПРЕОБРАЗОВАНИЕ interaction_weight В rating")
    print("==============================")

    new_df["rating"] = new_df["interaction_weight"].apply(convert_weight_to_rating)

    print("\nПервые 20 строк после добавления rating:")
    print(
        new_df[
            ["user_id", "product_id", "interaction_weight", "rating"]
        ].head(20).to_string(index=False)
    )

    print("\nРаспределение условных оценок rating:")
    print(new_df["rating"].value_counts().sort_index().to_string())

    print("\nПояснение:")
    print("SVD++ работает с оценками.")
    print("Так как настоящих оценок в датасете нет, мы используем условную оценку интереса.")
    print("Она построена на основе interaction_weight.")

    print("\n==============================")
    print("5. TRAIN/TEST SPLIT ДЛЯ ОЦЕНКИ SVD++")
    print("==============================")

    evaluation_users_count = 200

    print("\nСколько пользователей берем для оценки качества SVD++:")
    print(evaluation_users_count)

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

    print("\nРазмер train_df:")
    print(train_df.shape)

    print("\nРазмер test_df:")
    print(test_df.shape)

    print("\nПервые 10 строк train_df:")
    print(
        train_df[
            ["user_id", "product_id", "interaction_weight", "rating"]
        ].head(10).to_string(index=False)
    )

    print("\nПервые 10 спрятанных товаров test_df:")
    print(
        test_df[
            ["user_id", "product_id", "interaction_weight", "rating"]
        ].head(10).to_string(index=False)
    )

    print("\nПояснение:")
    print("train_df — данные, на которых обучается SVD++.")
    print("test_df — товары, которые мы спрятали для проверки качества.")





    print("\n==============================")
    print("6. ПОДГОТОВКА ДАННЫХ ДЛЯ БИБЛИОТЕКИ SURPRISE")
    print("==============================")

    svd_df = train_df[["user_id", "product_id", "rating"]].copy()

    svd_df["user_id"] = svd_df["user_id"].astype(str)
    svd_df["product_id"] = svd_df["product_id"].astype(str)

    reader = Reader(rating_scale=(1, 5))

    data = Dataset.load_from_df(
        svd_df[["user_id", "product_id", "rating"]],
        reader
    )

    trainset = data.build_full_trainset()

    print("\nДанные для SVD++ подготовлены.")
    print("Количество пользователей в trainset:", trainset.n_users)
    print("Количество товаров в trainset:", trainset.n_items)
    print("Количество оценок в trainset:", trainset.n_ratings)

    print("\n==============================")
    print("7. ОБУЧЕНИЕ МОДЕЛИ SVD++")
    print("==============================")

    model = SVDpp(
    n_factors=50,
    n_epochs=20,
    random_state=42,
    verbose=True,
    cache_ratings=True
)

    print("\nПараметры модели:")
    print("n_factors = 50")
    print("n_epochs = 20")
    print("random_state = 42")
    print("cache_ratings = True")

    print("\nНачинаем обучение SVD++:")
    model.fit(trainset)

    print("\nОбучение SVD++ завершено.")

    print("\n==============================")
    print("8. ОЦЕНКА КАЧЕСТВА SVD++")
    print("==============================")

    recommendations_count = 10
    candidate_limit = 5000

    print("\nКоличество рекомендаций для проверки:")
    print(recommendations_count)

    print("\nКоличество популярных товаров-кандидатов:")
    print(candidate_limit)

    popular_candidates = train_df.groupby("product_id")["interaction_weight"].sum()
    popular_candidates = popular_candidates.sort_values(ascending=False)
    popular_candidates = popular_candidates.head(candidate_limit).index.tolist()

    train_user_history = train_df.groupby("user_id")["product_id"].apply(set).to_dict()

    results = []

    for _, row in test_df.iterrows():
        real_user_id = int(row["user_id"])
        hidden_product_id = int(row["product_id"])

        user_train_products = train_user_history.get(real_user_id, set())

        candidate_products = [
            int(product_id)
            for product_id in popular_candidates
            if product_id not in user_train_products
        ]

        if hidden_product_id not in candidate_products:
            candidate_products.append(hidden_product_id)

        predictions = []

        for product_id in candidate_products:
            prediction = model.predict(
                str(real_user_id),
                str(product_id)
            )

            predictions.append({
                "product_id": product_id,
                "score": prediction.est
            })




        predictions_df = pd.DataFrame(predictions)

        recommendations = predictions_df.sort_values(
            by="score",
            ascending=False
        ).head(recommendations_count)

        recommended_product_ids = recommendations["product_id"].tolist()

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

    evaluation_output_file = project_root / "data" / "svdpp_evaluation_2019-Nov.csv"
    evaluation_results.to_csv(evaluation_output_file, index=False)

    print("\nРезультаты оценки SVD++ сохранены в файл:")
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

    print("\nПользователи, подходящие для проверки рекомендаций:")
    print(good_users.head(10).to_string(index=False))

    if len(good_users) == 0:
        print("\nНе найдено пользователей с подходящим количеством товаров.")
        print("Для проверки берем первого пользователя из таблицы.")

        real_user_id = new_df.iloc[0]["user_id"]

    else:

        real_user_id = good_users.iloc[0]["user_id"]




    user_id = int(real_user_id)

    print("\nВыбран пользователь:")
    print("user_id:", user_id)

    print("\n==============================")
    print("10. ИСТОРИЯ ВЫБРАННОГО ПОЛЬЗОВАТЕЛЯ")
    print("==============================")

    user_history = new_df[
        new_df["user_id"] == real_user_id
    ][["product_id", "interaction_weight", "rating"]]

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

    history_output_file = project_root / "data" / "svdpp_user_history_2019-Nov.csv"
    user_history_full.to_csv(history_output_file, index=False)

    print("\nИстория пользователя сохранена в файл:")
    print(history_output_file)

    print("\n==============================")
    print("11. ПОЛУЧЕНИЕ РЕКОМЕНДАЦИЙ SVD++")
    print("==============================")

    user_history_products = set(user_history["product_id"].unique())

    candidate_products = new_df[
        ~new_df["product_id"].isin(user_history_products)
    ]["product_id"].drop_duplicates()

    print("\nКоличество товаров-кандидатов для рекомендаций:")
    print(len(candidate_products))

    predictions = []

    for product_id in candidate_products:
        prediction = model.predict(
            str(user_id),
            str(int(product_id))
        )




        predictions.append({
            "product_id": int(product_id),
            "score": prediction.est
        })

    rec_table = pd.DataFrame(predictions)

    rec_table = rec_table.sort_values(
        by="score",
        ascending=False
    ).head(recommendations_count)

    print("\nТаблица рекомендаций до добавления описания товаров:")
    print(rec_table.to_string(index=False))

    rec_table_full = rec_table.merge(
        info_df,
        on="product_id",
        how="left"
    )

    rec_table_full.insert(0, "user_id", user_id)

    print("\nРекомендации SVD++ с category_code и brand:")
    print(rec_table_full.to_string(index=False))

    output_file = project_root / "data" / "svdpp_recommendations_2019-Nov.csv"
    rec_table_full.to_csv(output_file, index=False)

    print("\nРекомендации SVD++ сохранены в файл:")
    print(output_file)

    print("\n==============================")
    print("12. ЗАВЕРШЕНИЕ РАБОТЫ")
    print("==============================")

    end_time = time.time()
    print(f"\nВремя выполнения скрипта: {end_time - start_time:.2f} секунд")


if __name__ == "__main__":
    main()