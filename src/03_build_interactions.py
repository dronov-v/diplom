import time
import pandas as pd
from pathlib import Path

def main():
    print("\nСТАРТ№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№")
    start_time = time.time()

    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "filtered_2019-Nov.csv"
    output_path = project_root / "data" / "interaction_2019-Nov.csv"


    df = pd.read_csv(data_path, usecols=["user_id", "product_id", "event_type"])

    print("\nРазмер таблицы с 3 столбиками")
    print(df.shape)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)

    print("\nПервые строки таблицы")
    print(df.head().to_string())



    #event_weights = {"view": 1, "cart": 3, "purchase": 5}
    event_weights = {"view": 1, "cart": 2, "purchase": 3}

    df["interaction_weight"] = df["event_type"].map(event_weights) #new stolbik

    print("\nПервые 15 строк с новым столбиком")
    print(df.head(15).to_string())

    interactions_df = (df.groupby(["user_id", "product_id"], as_index=False)["interaction_weight"].sum())
    print("\nРазмер таблицы после группировке стобликов:")
    print(interactions_df.shape)

    print("\nПервые 10 строки обновленной таблички:")
    print(interactions_df.head(10).to_string())


    print("\nTop самых больших interaction_weight:")
    print(interactions_df["interaction_weight"].sort_values(ascending=False).head(10).to_string())

    interactions_df.to_csv(output_path, index=False)





    end_time = time.time()
    print(f"\n\nВремя выполнения: {end_time-start_time:.2f} seconds")

if __name__ == "__main__":
    main()

