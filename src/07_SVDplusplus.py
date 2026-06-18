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

    common_file = project_root / "data" / "common_interactions_2019-Nov.csv"
    train_file = project_root / "data" / "common_train_2019-Nov.csv"
    hidden_file = project_root / "data" / "common_hidden_2019-Nov.csv"
    info_file = project_root / "data" / "filtered_2019-Nov.csv"

    print("\n==============================")
    print("1. ЗАГРУЗКА ОБЩИХ ДАННЫХ ДЛЯ SVD++")
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
    print("3. ПРЕОБРАЗОВАНИЕ interaction_weight В rating")
    print("==============================")

    common_df["rating"] = common_df["interaction_weight"].apply(convert_weight_to_rating)
    train_df["rating"] = train_df["interaction_weight"].apply(convert_weight_to_rating)

    print("\nПервые 20 строк train_df после добавления rating:")
    print(
        train_df[
            ["user_id", "product_id", "interaction_weight", "rating"]
        ].head(20).to_string(index=False)
    )

    print("\nРаспределение rating в train_df:")
    print(train_df["rating"].value_counts().sort_index().to_string())

    print("\nПояснение:")
    print("SVD++ работает с оценками.")
    print("Так как настоящих оценок нет, interaction_weight переводится в условный rating от 1 до 5.")

    print("\n==============================")
    print("4. ПОДГОТОВКА ДАННЫХ ДЛЯ SURPRISE")
    print("==============================")

    svd_train_df = train_df[["user_id", "product_id", "rating"]].copy()

    svd_train_df["user_id"] = svd_train_df["user_id"].astype(str)
    svd_train_df["product_id"] = svd_train_df["product_id"].astype(str)

    reader = Reader(rating_scale=(1, 5))

    data = Dataset.load_from_df(
        svd_train_df[["user_id", "product_id", "rating"]],
        reader
    )

    trainset = data.build_full_trainset()

    print("\nДанные для SVD++ подготовлены.")
    print("Количество пользователей в trainset:", trainset.n_users)
    print("Количество товаров в trainset:", trainset.n_items)
    print("Количество оценок в trainset:", trainset.n_ratings)

    print("\n==============================")
    print("5. ОБУЧЕНИЕ SVD++ НА ОБЩИХ ДАННЫХ")
    print("==============================")

    model = SVDpp(
        n_factors=50,
        n_epochs=20,
        random_state=42,
        verbose=True,
        cache_ratings=True
    )

    print("\nПараметры SVD++:")
    print("n_factors = 50")
    print("n_epochs = 20")
    print("random_state = 42")
    print("cache_ratings = True")

    print("\nНачинаем обучение SVD++:")
    model.fit(trainset)

    print("\nОбучение SVD++ завершено.")

    print("\n==============================")
    print("6. ОЦЕНКА SVD++ НА HIDDEN-ТОВАРАХ")
    print("==============================")

    recommendations_count = 10

    all_product_ids = common_df["product_id"].drop_duplicates().astype(int).tolist()

    train_user_history = (
        train_df.groupby("user_id")["product_id"]
        .apply(set)
        .to_dict()
    )

    results = []

    for row_number, row in hidden_df.iterrows():
        if row_number % 100 == 0:
            print("Проверено hidden-пользователей:", row_number, "из", len(hidden_df))

        real_user_id = int(row["user_id"])
        user_index = int(row["user_index"])
        hidden_product_id = int(row["hidden_product_id"])

        user_train_products = train_user_history.get(real_user_id, set())

        candidate_products = [
            product_id
            for product_id in all_product_ids
            if product_id not in user_train_products
        ]

        predictions = []

        for product_id in candidate_products:
            prediction = model.predict(
                str(real_user_id),
                str(product_id)
            )

            predictions.append({
                "product_id": int(product_id),
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

    evaluation_output_file = project_root / "data" / "svdpp_common_evaluation_2019-Nov.csv"
    evaluation_results.to_csv(evaluation_output_file, index=False)

    print("\nРезультаты оценки SVD++ сохранены в файл:")
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

    print("\nВыбран пользователь:")
    print("user_id:", user_id)

    print("\n==============================")
    print("8. ИСТОРИЯ ВЫБРАННОГО ПОЛЬЗОВАТЕЛЯ")
    print("==============================")

    user_history = common_df[
        common_df["user_id"] == real_user_id
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

    history_output_file = project_root / "data" / "svdpp_common_user_history_2019-Nov.csv"
    user_history_full.to_csv(history_output_file, index=False)

    print("\nИстория пользователя сохранена в файл:")
    print(history_output_file)

    print("\n==============================")
    print("9. ПОЛУЧЕНИЕ РЕКОМЕНДАЦИЙ SVD++ ДЛЯ ПРИМЕРА")
    print("==============================")

    user_history_products = set(user_history["product_id"].unique())

    candidate_products = [
        product_id
        for product_id in all_product_ids
        if product_id not in user_history_products
    ]

    predictions = []

    for product_id in candidate_products:
        prediction = model.predict(
            str(user_id),
            str(product_id)
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

    output_file = project_root / "data" / "svdpp_common_recommendations_2019-Nov.csv"
    rec_table_full.to_csv(output_file, index=False)

    print("\nРекомендации SVD++ сохранены в файл:")
    print(output_file)

    print("\n==============================")
    print("10. ЗАВЕРШЕНИЕ РАБОТЫ")
    print("==============================")

    end_time = time.time()
    print(f"\nВремя выполнения скрипта: {end_time - start_time:.2f} секунд")


if __name__ == "__main__":
    main()