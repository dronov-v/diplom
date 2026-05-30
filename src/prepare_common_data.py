import time
import pandas as pd
from pathlib import Path


def main():
    print("COMMONCOMMONCOMMONCOMMONCOMMON")

    start_time = time.time()

    project_root = Path(__file__).resolve().parent.parent

    input_file = project_root / "data" / "matrix_2019-Nov.csv"

    common_output_file = project_root / "data" / "common_interactions_2019-Nov.csv"
    train_output_file = project_root / "data" / "common_train_2019-Nov.csv"
    hidden_output_file = project_root / "data" / "common_hidden_2019-Nov.csv"

    print("\n==============================")
    print("1. ЗАГРУЗКА ОСНОВНОЙ ТАБЛИЦЫ")
    print("==============================")

    df = pd.read_csv(input_file)

    print("\nРазмер исходной таблицы matrix_2019-Nov.csv:")
    print(df.shape)

    print("\nПервые 10 строк:")
    print(df.head(10).to_string(index=False))

    print("\n==============================")
    print("2. ОБЩИЕ НАСТРОЙКИ ДЛЯ ВСЕХ АЛГОРИТМОВ")
    print("==============================")

    common_users_count = 50000
    common_popular_items_count = 2500
    hidden_users_count = 1000

    print("\nКоличество пользователей для общего набора:")
    print(common_users_count)

    print("\nКоличество популярных товаров для общего набора:")
    print(common_popular_items_count)

    print("\nКоличество пользователей, у которых спрячем товар:")
    print(hidden_users_count)

    print("\n==============================")
    print("3. ВЫБОР ОБЩИХ ПОЛЬЗОВАТЕЛЕЙ")
    print("==============================")

    selected_users = df["user_id"].drop_duplicates().head(common_users_count)

    common_df = df[
        df["user_id"].isin(selected_users)
    ].copy()

    print("\nРазмер таблицы после выбора пользователей:")
    print(common_df.shape)

    print("\nКоличество уникальных пользователей:")
    print(common_df["user_id"].nunique())

    print("\nКоличество уникальных товаров:")
    print(common_df["product_id"].nunique())

    print("\n==============================")
    print("4. ВЫБОР ОБЩИХ ПОПУЛЯРНЫХ ТОВАРОВ")
    print("==============================")

    popular_items = (
        common_df.groupby("product_id")["interaction_weight"]
        .sum()
        .sort_values(ascending=False)
        .head(common_popular_items_count)
        .index
    )

    common_df = common_df[
        common_df["product_id"].isin(popular_items)
    ].copy()

    print("\nРазмер таблицы после выбора популярных товаров:")
    print(common_df.shape)

    print("\nКоличество уникальных пользователей после выбора товаров:")
    print(common_df["user_id"].nunique())

    print("\nКоличество уникальных товаров после выбора товаров:")
    print(common_df["product_id"].nunique())

    print("\nПервые 10 строк после выбора товаров:")
    print(common_df.head(10).to_string(index=False))

    print("\n==============================")
    print("5. ПЕРЕСОЗДАНИЕ ПОНЯТНЫХ ИНДЕКСОВ")
    print("==============================")

    common_df = common_df[
        ["user_id", "product_id", "interaction_weight"]
    ].copy()

    common_users = common_df["user_id"].unique()
    common_items = common_df["product_id"].unique()

    user_map = {
        user_id: user_index
        for user_index, user_id in enumerate(common_users)
    }

    item_map = {
        product_id: item_index
        for item_index, product_id in enumerate(common_items)
    }

    common_df["user_index"] = common_df["user_id"].map(user_map)
    common_df["item_index"] = common_df["product_id"].map(item_map)

    print("\nКоличество пользователей после пересоздания индексов:")
    print(len(common_users))

    print("\nКоличество товаров после пересоздания индексов:")
    print(len(common_items))

    print("\nПервые 10 строк с новыми индексами:")
    print(common_df.head(10).to_string(index=False))

    print("\n==============================")
    print("6. ВЫБОР HIDDEN-ТОВАРОВ")
    print("==============================")

    user_products_count = common_df.groupby("user_id")["product_id"].nunique()

    users_for_hidden = user_products_count[
        user_products_count >= 2
    ].index[:hidden_users_count]

    hidden_candidates_df = common_df[
        common_df["user_id"].isin(users_for_hidden)
    ].copy()

    hidden_rows = hidden_candidates_df.groupby(
        "user_id",
        group_keys=False
    ).sample(n=1, random_state=42)

    train_df = common_df.drop(hidden_rows.index).copy()

    hidden_df = hidden_rows.copy()

    hidden_df = hidden_df.rename(
        columns={
            "product_id": "hidden_product_id",
            "interaction_weight": "hidden_interaction_weight",
            "item_index": "hidden_item_index"
        }
    )

    hidden_df = hidden_df[
        [
            "user_id",
            "user_index",
            "hidden_product_id",
            "hidden_item_index",
            "hidden_interaction_weight"
        ]
    ].copy()

    print("\nСколько пользователей реально получили hidden-товар:")
    print(hidden_df["user_id"].nunique())

    print("\nРазмер common_interactions:")
    print(common_df.shape)

    print("\nРазмер common_train:")
    print(train_df.shape)

    print("\nРазмер common_hidden:")
    print(hidden_df.shape)

    print("\nПервые 10 строк common_train:")
    print(train_df.head(10).to_string(index=False))

    print("\nПервые 10 hidden-товаров:")
    print(hidden_df.head(10).to_string(index=False))

    print("\nПояснение:")
    print("common_train — данные, на которых будут обучаться ALS, SVD++ и SLIM.")
    print("common_hidden — товары, которые мы специально спрятали для проверки.")
    print("Все алгоритмы будут пытаться угадать одни и те же hidden-товары.")

    print("\n==============================")
    print("7. СОХРАНЕНИЕ ОБЩИХ ФАЙЛОВ")
    print("==============================")

    common_df.to_csv(common_output_file, index=False)
    train_df.to_csv(train_output_file, index=False)
    hidden_df.to_csv(hidden_output_file, index=False)

    print("\nОбщий полный файл сохранен:")
    print(common_output_file)

    print("\nОбщий train-файл сохранен:")
    print(train_output_file)

    print("\nОбщий hidden-файл сохранен:")
    print(hidden_output_file)





    print("\n==============================")
    print("8. ЗАВЕРШЕНИЕ РАБОТЫ")
    print("==============================")

    end_time = time.time()
    print(f"\nВремя выполнения скрипта: {end_time - start_time:.2f} секунд")


if __name__ == "__main__":
    main()