### Core modules ###
from datetime import (
    datetime,
    timedelta,
    timezone
)
from pathlib import Path
from pandas import (
    read_csv,
    concat
)


### Type hints ###
from pandas import DataFrame


### Internal modules ###


statistic_dir: Path = (Path(__file__).resolve(strict=True).parent.parent.joinpath("data", "statistics"))
statistic_dict: dict[str, int | str | float | datetime] = {
    # NOTE: these are just example data.
    "user_id": 0,
    "email": "domain@example.com",
    "start_date": datetime.now(tz=timezone.utc),
    "last_date": datetime.now(tz=timezone.utc) - timedelta(days=1),
    "average_token_length": 0.00,
    "query_count": 0
}


def update_statistic_table(
    statistic_dict: dict[str, int | str | float | datetime]
) -> None:
    global statistic_dir

    statistic_dir.mkdir(
        mode=0o777,
        parents=False,
        exist_ok=True
    )

    current_time: str = str(object=statistic_dict["last_date"])
    time_split: list[str] = current_time.split(
        sep=",",
        maxsplit=-1
    )[0].split(
        sep="-",
        maxsplit=-1
    )
    current_month_year: str = f"{time_split[1]}-{time_split[2]}"
    statistic_path: Path = statistic_dir.joinpath(f"{current_month_year}.csv")

    if statistic_path.exists(follow_symlinks=True):
        data: DataFrame = read_csv(
            filepath_or_buffer=statistic_path,
            dtype={
                "average_token_length": float
            }
        )
        user_emails: list[str | None] = data["email"].tolist()

        if statistic_dict["email"] in user_emails:
            idx: int = [idx for (idx, user_email) in enumerate(iterable=user_emails, start=0) if user_email == statistic_dict["email"]][0]
            data.loc[idx, "last_date"] = statistic_dict["last_date"]

            history_total_token_length: float = \
                data["average_token_length"][idx] * data["query_count"][idx]

            current_total_token_length: float = \
                statistic_dict["average_token_length"] * statistic_dict["query_count"] # pyright: ignore

            total_query_count: int = \
                data["query_count"][idx] + statistic_dict["query_count"]

            data.loc[idx, "average_token_length"] = \
                (history_total_token_length + current_total_token_length) / total_query_count # pyright: ignore
            data.loc[idx, "query_count"] = total_query_count

        else:
            df: DataFrame = DataFrame(
                data=[
                    {
                        "user_id": statistic_dict["user_id"],
                        "email": statistic_dict["email"],
                        "start_date": statistic_dict["start_date"],
                        "last_date": statistic_dict["last_date"],
                        "average_token_length": statistic_dict["average_token_length"],
                        "query_count": statistic_dict["query_count"]
                    }
                ]
            )

            if len(user_emails) > 0:
                df: DataFrame = concat(
                    objs=[data, df],
                    axis=0
                )

            else:
                data: DataFrame = df

        data.to_csv(
            path_or_buf=statistic_path,
            header=[
                "user_id",
                "email",
                "start_date",
                "last_date",
                "average_token_length",
                "query_count"
            ],
            index=False
        )

        return None

    else:
        user_emails: list[str | None] = []

        return None


def update_statistic_per_query(
    query: list[str],
    user_id: int,
    user_email: str | None,
    current_time: str
) -> None:
    global statistic_dict

    if False:
        # Save for the previous user (when the user_id changed).
        pre_user_id: int = statistic_dict["user_id"]
        pre_query_count: str = statistic_dict["query_count"]
        token_length: int = len(query)

        # Accumulate for the same user.
        if pre_user_id == user_id:
            pre_average_token_length: float = statistic_dict["average_token_length"]

            statistic_dict["last_date"] = current_time

            statistic_dict["average_token_length"] = \
                (pre_average_token_length * pre_query_count + token_length) / (pre_query_count + 1)

            statistic_dict["query_count"] = pre_query_count + 1

        else:
            # Save previous user statistic.
            if pre_user_id != 0:
                update_statistic_table(
                    statistic_dict=statistic_dict
                )

            else:
                # Initialize for a different user.
                statistic_dict["user_id"] = user_id
                statistic_dict["email"] = user_email
                statistic_dict["start_date"] = current_time
                statistic_dict["last_date"] = current_time
                statistic_dict["average_token_length"] = token_length
                statistic_dict["query_count"] = 1

    else:
        # Save every query for the current user.
        token_length: int = len(query)

        statistic_dict["user_id"] = user_id
        statistic_dict["email"] = str(object=user_email)
        statistic_dict["start_date"] = current_time
        statistic_dict["last_date"] = current_time
        statistic_dict["average_token_length"] = token_length
        statistic_dict["query_count"] = 1

        update_statistic_table(
            statistic_dict=statistic_dict
        )

        return None