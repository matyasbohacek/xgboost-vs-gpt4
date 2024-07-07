
import pandas as pd


def load_dataset(path: str, split="") -> pd.DataFrame:
    """
    Loads the Verifee dataset from its CSV path. For more information about the dataset itself, refer to its original
    publication:

        "Czech-ing the News: Article Trustworthiness Dataset for Czech",
            Matyas Bohacek, Michal Bravansky, Filip Trhlík, Vaclav Moravec,
            published in Proceedings of the 13th Workshop on Computational Approaches to Subjectivity, Sentiment, &
            Social Media Analysis

    :param path: (str)
    :param split: (str)

    :return: (pd.DataFrame)
    """

    df = pd.read_csv(path, encoding="utf-8")

    # filter for essential data -- other columns are irrelevant to the experiments in this repository
    df = df[["title", "text", "author", "split", "label"]]

    # filter duplicates -- the published CSV captures records as annotations, and thus contains multiple records for a
    # given unique article (capturing the inter-annotator agreement), which is irrelevant to the experiments in this
    # repository
    df.dropna(subset=["split", "title", "text"], how="all", inplace=True)
    df.drop_duplicates(subset=["title"], keep="last", inplace=True)

    #
    df["label"] = df["label"].apply(lambda cls: cls.lower())

    if split:
        df = df[df["split"] == split]

    return df


if __name__ == "__main__":
    pass
