#########################создаем матрицу юзер-товар
import time
import pandas as pd
from pathlib import Path

def main():
    print("STARTSTARTSTARTSTARTSTARTSTARTSTARTSTARTSTARTSTART")
    start_time = time.time()

    project_root = Path(__file__).resolve().parent.parent

    data_path = project_root / "data" / "matrix_2019-Nov.csv"
    output_path = project_root / "data" / "100k_matrix_2019-Nov.csv"


    df = pd.read_csv(data_path)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)


    print("\nПервые строки новой таблички:")
    print(df.info())
    print("\nКол-во уник пользователей:")

    print(df["user_id"].nunique())
    first_users = df["user_id"].drop_duplicates().head(100000)


    new_df = df[df["user_id"].isin(first_users)]

    print("\nРазмер таблицы после отбора пользователей")
    print(new_df.shape)

    print("\nПервые строки таблички:")
    print(new_df.head().to_string())

    print("\nКол-во уник пользователей в таблице:")
    print(new_df["user_id"].nunique())
    print("\nКол-во уник товаров в таблице:")
    print(new_df["product_id"].nunique())

    new_df.to_csv(output_path, index=False)

    end_time = time.time()
    print(f"\nВремя выполнения: {end_time - start_time:.2f} секунд")





if __name__ == "__main__":
    main()



