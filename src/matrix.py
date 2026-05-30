#########################создаем матрицу юзер-товар
import time
import pandas as pd
from pathlib import Path
from scipy.sparse import csr_matrix


def main():
    print("STARTSTARTSTARTSTARTSTARTSTARTSTARTSTARTSTARTSTARt")
    start_time = time.time()

    project_root = Path(__file__).resolve().parent.parent

    data_path = project_root / "data" / "100k_matrix_2019-Nov.csv"

    df = pd.read_csv(data_path)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)

    first_users = df["user_id"].drop_duplicates().head(100000)
    new_df = df[df["user_id"].isin(first_users)]

    new_users = new_df["user_id"].unique()
    new_items = new_df["product_id"].unique()

    user_map = {user_id: index for index, user_id in enumerate(new_users)}
    product_map = {product_id: index for index, product_id in enumerate(new_items)}

    new_df["user_index"] = new_df["user_id"].map(user_map)
    new_df["item_index"] = new_df["product_id"].map(product_map)






    print("\n10 строк после пересчета индека:")
    print(new_df.head(10).to_string())




    matrix = csr_matrix((new_df["interaction_weight"], (new_df["user_index"], new_df["item_index"])))#user - строки item - столбики

    print("\nРазмер матрицы")
    print(matrix.shape)#число пользователей и число товаров

    print("\nКол во ненулевых знач в матрице:")
    print(matrix.nnz)

    end_time = time.time()
    print(f"\nВремя выполнения: {end_time - start_time:.2f} секунд")





if __name__ == "__main__":
    main()



