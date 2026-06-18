import time
import pandas as pd
from pathlib import Path

def main():
    print("ПОЛЕТЕЛИ!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    start_time = time.time()

    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "2019-Nov.csv"
    output_path = project_root / "data" / "filtered_2019-Nov.csv"

    df = pd.read_csv(data_path)

    print("\nРазмер исходной таблицы")
    print(df.shape)


    df = df.dropna(subset=["category_code"])
    print("\n\n\nРазмер таблицы после удаления пустых категори_код")
    print(df.shape)




    electronics_mask = df["category_code"].str.startswith("electronics")
    computers_mask = df["category_code"].str.startswith("computers")

    final_mask = electronics_mask | computers_mask
    filtered_df = df[final_mask]

    print("\nРазмер таблицы после филтра по категориям electronics and computers^")
    print(filtered_df.shape)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)

    print("\nПервые 10 строк после фильтрации:")
    print(filtered_df.head(10).to_string())

    print("\nТоп 30 категорий:")
    print(filtered_df["category_code"].value_counts().head(30).to_string())

    filtered_df.to_csv(output_path, index=False)
    end_time = time.time()
    print(f"\nВремя выполнения: {end_time - start_time:.2f} секунд")

if __name__ == "__main__":
    main()

