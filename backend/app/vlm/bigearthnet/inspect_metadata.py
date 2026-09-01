import pandas as pd


PATH = "data/bigearthnet/metadata.parquet"


if __name__ == "__main__":

    df = pd.read_parquet(PATH)

    print("\nCOLUMNS:")
    print(df.columns.tolist())

    print("\nNUMBER OF ROWS:")
    print(len(df))

    print("\nFIRST 5 ROWS:")
    print(df.head())

    print("\nSAMPLE PATCH MATCH:")

    target = "S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_57"

    matches = df[
        df.astype(str)
          .apply(
              lambda col:
              col.str.contains(
                  target,
                  regex=False,
                  na=False
              )
          )
          .any(axis=1)
    ]

    print(matches.to_string())