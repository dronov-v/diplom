import time
import pandas as pd
from pathlib import Path


def main():
    print("ЫЕФКЕ!!!!!!!!!!")

    start_time = time.time()

    project_root = Path(__file__).resolve().parent.parent

    input_file = project_root / "data" / "interaction_2019-Nov.csv"
    output_file = project_root / "data" / "matrix_2019-Nov.csv"

    print("\n==============================")
    print("1. ЗАГРУЗКА ТАБЛИЦЫ ВЗАИМОДЕЙСТВИЙ")
    print("==============================")

    df = pd.read_csv(input_file)

    print("\nРазмер interaction_2019-Nov.csv:")
    print(df.shape)

    print("\nПервые 10 строк:")
    print(df.head(10).to_string(index=False))

    print("\nНазвания столбцов:")
    print(df.columns.tolist())

    print("\nПроверка пропусков:")
    print(df.isna().sum())

    print("\n==============================")
    print("2. СОЗДАНИЕ ИНДЕКСОВ ПОЛЬЗОВАТЕЛЕЙ И ТОВАРОВ")
    print("==============================")

    users = df["user_id"].unique()
    items = df["product_id"].unique()

    user_map = {
        user_id: user_index
        for user_index, user_id in enumerate(users)
    }

    item_map = {
        product_id: item_index
        for item_index, product_id in enumerate(items)
    }

    df["user_index"] = df["user_id"].map(user_map)
    df["item_index"] = df["product_id"].map(item_map)

    print("\nКоличество уникальных пользователей:")
    print(len(users))

    print("\nКоличество уникальных товаров:")
    print(len(items))

    print("\nПервые 10 строк после добавления индексов:")
    print(df.head(10).to_string(index=False))

    print("\n==============================")
    print("3. СОХРАНЕНИЕ MATRIX-ФАЙЛА")
    print("==============================")

    df = df[
        [
            "user_id",
            "product_id",
            "interaction_weight",
            "user_index",
            "item_index"
        ]
    ].copy()

    df.to_csv(output_file, index=False)

    print("\nФайл matrix_2019-Nov.csv сохранен:")
    print(output_file)

    print("\nРазмер сохраненной таблицы:")
    print(df.shape)

    print("\nПервые 10 строк сохраненной таблицы:")
    print(df.head(10).to_string(index=False))

    print("\n==============================")
    print("4. ЗАВЕРШЕНИЕ РАБОТЫ")
    print("==============================")

    end_time = time.time()
    print(f"\nВремя выполнения скрипта: {end_time - start_time:.2f} секунд")






if __name__ == "__main__":
    main()