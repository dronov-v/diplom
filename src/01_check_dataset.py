import pandas as pd
import time
from pathlib import Path

def main ():
    start_time = time.time()
    print("СТАРТУЕМ!!!!!!!!!!!")
    #data_path = Path("data/2019-Nov.csv")
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "2019-Nov.csv"

    df = pd.read_csv(data_path)

    pd.set_option("display.max_columns", None) #не обрезаем столбцы
    pd.set_option("display.width", None) #не переносим по ширине
    pd.set_option("display.max_colwidth", None) #показывает полное содержимое ячеек


    print("\nПервые 5 строе датасета:")
    print(df.head())

    print("\nСтолбики:")
    print(df.columns.tolist())

    print("\nРазмер таблицы:")
    print(df.shape)

    print("\nОбщая инфа о таблице:")
    df.info()

    unique_event_types = df["event_type"].unique()
    print("\nУникальные event_types:")
    print(unique_event_types)

    event_type_counts = df["event_type"].value_counts()
    print("\nКаждый тип события:")
    print(event_type_counts.to_string())

    missing_category_code = df["category_code"].isna().sum()
    print("\nКол во пропусков в category:")
    print(missing_category_code)

    top_categories = df["category_code"].value_counts().head(30)
    print("\nTOP 30 Самых часто встречающихся категорий")
    print(top_categories)



    end_time = time.time()

    print(f"\nВремя теста: {end_time - start_time:.2f} секунд")


if __name__ == "__main__":

    main()