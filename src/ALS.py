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

    df = pd.read_csv(data_path)

    first_users = df["user_id"].drop_duplicates().head(100000)
    new_df = df[df["user_id"].isin(first_users)].copy()

    new_users = new_df["user_id"].unique()
    new_items = new_df["product_id"].unique()



    user_map = {user_id: index for index, user_id in enumerate(new_users)}
    item_map = {product_id: index for index, product_id in enumerate(new_items)}

    new_df["user_index"] = new_df["user_id"].map(user_map)
    new_df["item_index"] = new_df["product_id"].map(item_map)

    values = new_df["interaction_weight"]
    rows = new_df["user_index"]
    cols = new_df["item_index"]

    matrix = csr_matrix((values, (rows, cols)))



    model = AlternatingLeastSquares(

        factors = 20,
        regularization = 0.1,
        iterations = 100

    )


    print("\nНачинаем обучение модели:")
    model.fit(matrix)

    end_time = time.time()

    print(f"\nВремя выполнения скрипта:{end_time - start_time} seconds")
if __name__ == "__main__":
    main()