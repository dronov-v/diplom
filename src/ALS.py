import time
import pandas as pd
from pathlib import Path
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares

def main():
    print("ALSALSALSALSALSALSALSALSALSALSALS")
    start_time = time.time()

    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "matrix_2019-Nov.csv"
    info_file = project_root / "data" / "filtered_2019-Nov.csv"

    df = pd.read_csv(data_path)

    info_df = pd.read_csv(info_file, usecols = ["product_id", "category_code", "brand"])
    info_df = info_df.drop_duplicates(subset=["product_id"])

    first_users = df["user_id"].drop_duplicates().head(100000)
    new_df = df[df["user_id"].isin(first_users)].copy()

    new_users = new_df["user_id"].unique()
    new_items = new_df["product_id"].unique()


    user_map = {user_id: index for index, user_id in enumerate(new_users)}
    item_map = {product_id: index for index, product_id in enumerate(new_items)}
    reverse_item_map = {index: product_id for product_id, index in item_map.items()}

    new_df["user_index"] = new_df["user_id"].map(user_map)
    new_df["item_index"] = new_df["product_id"].map(item_map)

    values = new_df["interaction_weight"]
    rows = new_df["user_index"]
    cols = new_df["item_index"]

    matrix = csr_matrix((values, (rows, cols)))

    model = AlternatingLeastSquares(

        factors = 20,
        regularization = 0.1,
        iterations = 10
    )

    print("\nНачинаем обучение модели:")
    model.fit(matrix)

    real_user_id = new_df.iloc[3]["user_id"]
    user_id = int(real_user_id)
    user_index = user_map[user_id] #внутренний индекс пользователя через словарь

    print("\nРеальный user_id проверка ")
    print(user_id)

    print("\nВнутренний user_index")
    print(user_index)


    user_history = new_df[new_df["user_id"] == real_user_id][["product_id", "interaction_weight"]]
    print("\nТовары из истории одного выбранного пользователя:")
    print(user_history.head(15).to_string())


    user_history_full = user_history.merge(info_df, on="product_id", how="left")
    print("\nИстория пользователя с инфой об товарах:")
    print(user_history_full.to_string())

    rec_items, scores = model.recommend(user_index, matrix[user_index], N = 10)

    print("\nРекомендованные товары:")
    print(rec_items)
    print("\nОценка рекомендаций:")
    print(scores)

    real_product_id = [reverse_item_map[item_id] for item_id in rec_items]
    print("\nРеальный индекс товаров с которыми пользователь взаимодействовал:")
    print(real_product_id)

    rec_table = pd.DataFrame({
        "item_index": rec_items,
        "score": scores,
        "product_id": real_product_id
    })


    print("\nТаблица рекомендаций:")
    print(rec_table.to_string())

    rec_table_full = rec_table.merge(info_df, on="product_id", how="left")
    print("\nРекомендации с описанием товара:")
    print(rec_table_full.to_string())

    end_time = time.time()
    print(f"\nВремя выполнения скрипта:{end_time - start_time} seconds")
if __name__ == "__main__":
    main()